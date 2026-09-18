# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import datetime
import getopt
import logging
import math
import os
import pathlib
import re
import shlex
import time
from typing import Any, ClassVar, cast

import pydantic
import requests
import tenacity

import cloudai.core
import cloudai.util

from .slurm_metadata import SlurmStepMetadata
from .slurm_node import SlurmNode, SlurmNodeState, parse_node_list

logger = logging.getLogger(__name__)


class SlurmAPIConfig(pydantic.BaseModel):
    """Connection details for a Slurm REST API endpoint."""

    model_config = pydantic.ConfigDict(extra="forbid")

    url: str
    headers: dict[str, str] = pydantic.Field(default_factory=dict)
    verify_certs: bool = True


class SlurmRestClient:
    """Translate CloudAI Slurm operations to slurmrestd v0.0.38 requests."""

    _API_VERSION: ClassVar[str] = "v0.0.38"
    _REQUEST_TIMEOUT_SECONDS: ClassVar[int] = 30
    _TERMINAL_JOB_STATES: ClassVar[frozenset[str]] = frozenset(
        {
            "BOOT_FAIL",
            "CANCELLED",
            "COMPLETED",
            "DEADLINE",
            "FAILED",
            "NODE_FAIL",
            "OUT_OF_MEMORY",
            "PREEMPTED",
            "REVOKED",
            "SPECIAL_EXIT",
            "TIMEOUT",
        }
    )
    _DIRECTIVE_FIELDS: ClassVar[dict[str, str]] = {
        "--job-name": "name",
        "-J": "name",
        "--output": "standard_output",
        "-o": "standard_output",
        "--error": "standard_error",
        "-e": "standard_error",
        "--partition": "partition",
        "-p": "partition",
        "--account": "account",
        "-A": "account",
        "--reservation": "reservation",
        "--distribution": "distribution",
        "--nodelist": "required_nodes",
        "--exclude": "excluded_nodes",
        "--chdir": "current_working_directory",
    }
    _SHORT_DIRECTIVES: ClassVar[str] = "J:o:e:p:A:N:n:D:"
    _LONG_DIRECTIVES: ClassVar[list[str]] = [
        "job-name=",
        "output=",
        "error=",
        "partition=",
        "account=",
        "reservation=",
        "distribution=",
        "nodes=",
        "nodelist=",
        "exclude=",
        "ntasks=",
        "ntasks-per-node=",
        "time=",
        "gres=",
        "gpus-per-node=",
        "chdir=",
    ]

    def __init__(self, config: SlurmAPIConfig, retry_pause_seconds: int) -> None:
        self._config = config
        self._retry_pause_seconds = retry_pause_seconds

    def _headers(self) -> dict[str, str]:
        """Expand environment variables in configured headers."""
        headers: dict[str, str] = {}
        for name, value in self._config.headers.items():
            expanded = os.path.expandvars(value)
            if re.search(r"\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[^}]+\})", expanded):
                raise EnvironmentError(f"Environment variable referenced by Slurm API header '{name}' is not set.")
            headers[name] = expanded
        return headers

    @staticmethod
    def _message(item: object) -> str:
        """Extract useful text from Slurm error/warning objects; e.g. `{"error": "bad"}` becomes `"bad"`."""
        if not isinstance(item, dict):
            return str(item)
        return str(item.get("error") or item.get("description") or item)

    def _request_once(self, method: str, service: str, path: str, payload: dict[str, object] | None) -> dict[str, Any]:
        url = f"{self._config.url.rstrip('/')}/{service}/{self._API_VERSION}/{path.lstrip('/')}"
        response = requests.request(
            method,
            url,
            headers=self._headers(),
            json=payload,
            timeout=self._REQUEST_TIMEOUT_SECONDS,
            verify=self._config.verify_certs,
        )
        try:
            data = response.json()
        except ValueError:
            response.raise_for_status()
            raise
        if not isinstance(data, dict):
            raise RuntimeError(f"Slurm API returned a non-object response from {url}.")
        if errors := data.get("errors"):
            details = "; ".join(self._message(error) for error in errors)
            raise RuntimeError(f"Slurm API request failed: {details}")
        response.raise_for_status()
        for warning in data.get("warnings", []):
            logger.warning("Slurm API warning: %s", self._message(warning))
        return data

    def _request(
        self,
        method: str,
        service: str,
        path: str,
        *,
        payload: dict[str, object] | None = None,
        retry_threshold: int = 1,
    ) -> dict[str, Any]:
        """Call slurmrestd, retrying failures up to `retry_threshold` attempts."""
        if retry_threshold < 1:
            raise ValueError("retry_threshold must be at least 1")

        retrying = tenacity.Retrying(
            stop=tenacity.stop_after_attempt(retry_threshold),
            wait=tenacity.wait_fixed(self._retry_pause_seconds),
            retry=tenacity.retry_if_exception_type((requests.RequestException, ValueError, RuntimeError)),
            before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        try:
            return retrying(self._request_once, method, service, path, payload)
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            raise RuntimeError(f"Slurm API request failed after {retry_threshold} attempt(s): {exc}") from exc

    @classmethod
    def _parse_sbatch_line(cls, line: str) -> list[tuple[str, str]]:
        """Parse one SBATCH line; e.g. `--nodes 2 --time=10` becomes two directives."""
        try:
            directives, values = getopt.gnu_getopt(shlex.split(line), cls._SHORT_DIRECTIVES, cls._LONG_DIRECTIVES)
        except getopt.GetoptError as exc:
            raise ValueError(f"Invalid SBATCH directive: {exc}.") from exc
        if values:
            raise ValueError(f"Unexpected SBATCH value(s): {' '.join(values)}.")
        return directives

    @staticmethod
    def _set_directive(job: dict[str, object], field: str, value: object, option: str) -> None:
        """Merge directives mapped to one REST field; e.g. `--gres=gpu:8` and `--gpus-per-node=8` must agree."""
        existing = job.get(field)
        if existing is not None and existing != value:
            raise ValueError(
                f"Conflicting SBATCH directives for '{option}': REST field '{field}' "
                f"is already {existing!r}, got {value!r}."
            )
        job[field] = value

    def _apply_sbatch_directive(self, job: dict[str, object], option: str, value: str) -> None:
        """Map one SBATCH directive to its v0.0.38 job field; e.g. `--nodes=2` sets `nodes=[2, 2]`."""
        option = {"-N": "--nodes", "-n": "--ntasks", "-D": "--chdir"}.get(option, option)

        if option in self._DIRECTIVE_FIELDS:
            self._set_directive(job, self._DIRECTIVE_FIELDS[option], value, option)
        elif option == "--nodes":
            node_counts = [int(item) for item in value.split("-", 1)]
            if len(node_counts) == 1:
                node_counts.append(node_counts[0])
            self._set_directive(job, "nodes", node_counts, option)
        elif option == "--ntasks":
            self._set_directive(job, "tasks", int(value), option)
        elif option == "--ntasks-per-node":
            self._set_directive(job, "tasks_per_node", int(value), option)
        elif option == "--time":
            minutes = (
                int(value) if value.isdigit() else math.ceil(cloudai.util.parse_time_limit(value).total_seconds() / 60)
            )
            self._set_directive(job, "time_limit", minutes, option)
        elif option in {"--gres", "--gpus-per-node"}:
            gres = value if option == "--gres" else f"gpu:{value}"
            self._set_directive(job, "gres", gres, option)
        else:
            raise ValueError(f"SBATCH directive '{option}' is not supported by CloudAI's Slurm REST transport.")

    def _make_job(self, script: str, script_path: pathlib.Path) -> dict[str, object]:
        """Build REST job properties from leading `#SBATCH` lines; script body remains unchanged."""
        job: dict[str, object] = {}
        for line in script.splitlines():
            stripped = line.strip()
            if not stripped or (stripped.startswith("#") and not stripped.startswith("#SBATCH")):
                continue
            if not stripped.startswith("#SBATCH"):
                break
            for option, value in self._parse_sbatch_line(stripped.removeprefix("#SBATCH").strip()):
                self._apply_sbatch_directive(job, option, value)

        job.setdefault("current_working_directory", str(script_path.parent.absolute()))
        environment = dict(os.environ)
        environment.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
        job["environment"] = environment
        return job

    def submit_sbatch(
        self, script_path: pathlib.Path, operation_name: str, *, wait: bool = False, monitor_interval: int = 1
    ) -> int:
        """Submit an SBATCH file and optionally wait for a terminal job state."""
        try:
            script = script_path.read_text(encoding="utf-8")
            data = self._request(
                "POST",
                "slurm",
                "job/submit",
                payload={"script": script, "job": self._make_job(script, script_path)},
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise cloudai.core.JobIdRetrievalError(
                test_name=operation_name,
                command=f"POST /slurm/{self._API_VERSION}/job/submit",
                stdout="",
                stderr=str(exc),
                message="Failed to submit job through Slurm REST API.",
            ) from exc

        job_id = data.get("job_id")
        if not isinstance(job_id, int):
            raise cloudai.core.JobIdRetrievalError(
                test_name=operation_name,
                command=f"POST /slurm/{self._API_VERSION}/job/submit",
                stdout=str(data),
                stderr="",
                message="Failed to retrieve job ID.",
            )

        if wait:
            while not self.is_job_completed(job_id):
                time.sleep(monitor_interval)
        return job_id

    def _cluster_nodes(self) -> list[dict[str, Any]]:
        nodes = self._request("GET", "slurm", "nodes/").get("nodes")
        if not isinstance(nodes, list):
            raise RuntimeError("Slurm API returned an invalid nodes response.")
        return cast(list[dict[str, Any]], nodes)

    @staticmethod
    def _node_state(state: str, state_flags: list[str]) -> SlurmNodeState:
        """Combine the v0.0.38 node state and flags; e.g. `idle` plus `DRAIN` means `DRAINED`."""
        state = state.upper()
        flags = {flag.upper() for flag in state_flags}

        if "NOT_RESPONDING" in flags:
            return SlurmNodeState.NOT_RESPONDING
        if "DRAIN" in flags:
            if state in {"ALLOCATED", "MIXED"} or "COMPLETING" in flags:
                return SlurmNodeState.DRAINING
            return SlurmNodeState.DRAINED
        if "FAIL" in flags:
            return SlurmNodeState.FAILING if state == "ALLOCATED" or "COMPLETING" in flags else SlurmNodeState.FAIL

        flag_states = {
            "INVALID_REG": SlurmNodeState.INVALID_REGISTRATION,
            "MAINTENANCE": SlurmNodeState.MAINTENANCE,
            "POWER_DOWN": SlurmNodeState.PENDING_POWER_DOWN_STATE,
            "POWER_UP": SlurmNodeState.BEING_POWERED_UP_OR_CONFIGURED,
            "POWERED_DOWN": SlurmNodeState.POWERED_DOWN_STATE,
            "POWERING_DOWN": SlurmNodeState.POWERING_DOWN_STATE,
            "POWERING_UP": SlurmNodeState.POWERING_UP_STATE,
            "REBOOT_REQUESTED": SlurmNodeState.REBOOT_REQUESTED,
            "REBOOT_ISSUED": SlurmNodeState.REBOOT_ISSUED_STATE,
            "PERFCTRS": SlurmNodeState.USING_NETWORK_PERFORMANCE_COUNTERS,
            "PLANNED": SlurmNodeState.PLANNED_STATE,
            "RESERVED": SlurmNodeState.RESERVED,
        }
        for flag in state_flags:
            if node_state := flag_states.get(flag.upper()):
                return node_state

        if "COMPLETING" in flags:
            return SlurmNodeState.ALLOCATED_COMPLETING if state == "ALLOCATED" else SlurmNodeState.COMPLETING

        try:
            return SlurmNodeState(state)
        except ValueError:
            return SlurmNodeState.UNKNOWN_STATE

    def get_nodes(self) -> list[SlurmNode]:
        nodes: list[SlurmNode] = []
        for node in self._cluster_nodes():
            state = self._node_state(node["state"], node["state_flags"])
            nodes.extend(
                SlurmNode(name=node["name"], partition=partition, state=state) for partition in node["partitions"]
            )
        return nodes

    def has_gpus(self) -> bool:
        return any(
            "gpu" in str(node.get(field, "")).lower() for node in self._cluster_nodes() for field in ("gres", "tres")
        )

    def _queue_jobs(self) -> list[dict[str, Any]]:
        jobs = self._request("GET", "slurm", "jobs/").get("jobs")
        if not isinstance(jobs, list):
            raise RuntimeError("Slurm API returned an invalid jobs response.")
        return cast(list[dict[str, Any]], jobs)

    def get_allocated_nodes(self) -> list[SlurmNode]:
        nodes: list[SlurmNode] = []
        for job in self._queue_jobs():
            if job["job_state"].upper().rstrip("+") not in {"RUNNING", "PENDING"}:
                continue

            nodes.extend(
                SlurmNode(
                    name=name,
                    partition=job["partition"] or "",
                    state=SlurmNodeState.ALLOCATED,
                    user=job["user_name"] or "N/A",
                )
                for name in parse_node_list(job["nodes"] or "")
            )
        return nodes

    def _get_job(self, job_id: int, retry_threshold: int = 3) -> dict[str, Any] | None:
        jobs = self._request("GET", "slurm", f"job/{job_id}", retry_threshold=retry_threshold).get("jobs")
        if not isinstance(jobs, list):
            raise RuntimeError("Slurm API returned an invalid jobs response.")
        return cast(dict[str, Any], jobs[0]) if jobs else None

    def get_job_state(self, job_id: int, retry_threshold: int = 3) -> str:
        """Return current job state from slurmctld."""
        job = self._get_job(job_id, retry_threshold)
        if job is None:
            return ""

        return job["job_state"].upper().rstrip("+")

    def is_job_completed(self, job_id: int, retry_threshold: int = 3) -> bool:
        """Return whether slurmctld reports a terminal job state."""
        return self.get_job_state(job_id, retry_threshold) in self._TERMINAL_JOB_STATES

    def get_job_status(self, job_id: int, retry_threshold: int = 3) -> list[SlurmStepMetadata]:
        job = self._get_job(job_id, retry_threshold)
        if job is None:
            return []

        raw_exit_code = job["exit_code"]
        return_code = 0
        signal = 0
        if 0 <= raw_exit_code <= 0xFFFF:
            if os.WIFEXITED(raw_exit_code):
                return_code = os.WEXITSTATUS(raw_exit_code)
            elif os.WIFSIGNALED(raw_exit_code):
                signal = os.WTERMSIG(raw_exit_code)

        start_timestamp = job["start_time"]
        end_timestamp = job["end_time"]
        start_time = (
            datetime.datetime.fromtimestamp(start_timestamp, tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if start_timestamp
            else ""
        )
        end_time = (
            datetime.datetime.fromtimestamp(end_timestamp, tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if end_timestamp
            else ""
        )
        elapsed_seconds = max(end_timestamp - start_timestamp, 0) if start_timestamp and end_timestamp else 0

        return [
            SlurmStepMetadata(
                job_id=job["job_id"],
                step_id="",
                name=job["name"],
                state=job["job_state"].upper().rstrip("+"),
                exit_code=f"{return_code}:{signal}",
                start_time=start_time,
                end_time=end_time,
                elapsed_time_sec=elapsed_seconds,
                submit_line=job.get("command") or "",
            )
        ]

    def get_job_nodes(self, job_id: int) -> str:
        job = self._get_job(job_id)
        if job is None:
            return ""
        return job.get("nodes") or ""

    def cancel(self, job_id: int) -> None:
        """Cancel a Slurm job through slurmctld."""
        self._request("DELETE", "slurm", f"job/{job_id}")

    def validate(self) -> None:
        """Verify access to slurmctld endpoints used by CloudAI."""
        self._request("GET", "slurm", "ping/")
