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
from typing import Any, ClassVar

import pydantic
import requests
import tenacity

import cloudai.core
import cloudai.util

from .slurm_metadata import SlurmStepMetadata

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
        "--nodelist": "nodelist",
        "--exclude": "exclude_nodes",
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
        job["environment"] = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")}
        return job

    def submit_sbatch(
        self, script_path: pathlib.Path, operation_name: str, *, wait: bool = False, monitor_interval: int = 1
    ) -> int:
        """Submit an SBATCH file and optionally wait for a terminal accounting state."""
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

    def cluster_nodes(self) -> list[dict[str, Any]]:
        nodes = self._request("GET", "slurm", "nodes/").get("nodes", [])
        if not isinstance(nodes, list):
            raise RuntimeError("Slurm API returned an invalid nodes response.")
        return [node for node in nodes if isinstance(node, dict)]

    def has_gpus(self) -> bool:
        return any(
            "gpu" in str(node.get(field, "")).lower() for node in self.cluster_nodes() for field in ("gres", "tres")
        )

    def queue_jobs(self) -> list[dict[str, Any]]:
        jobs = self._request("GET", "slurm", "jobs/").get("jobs", [])
        if not isinstance(jobs, list):
            raise RuntimeError("Slurm API returned an invalid jobs response.")
        return [job for job in jobs if isinstance(job, dict)]

    def _get_job(self, job_id: int, retry_threshold: int = 3) -> dict[str, Any] | None:
        jobs = self._request("GET", "slurmdb", f"job/{job_id}", retry_threshold=retry_threshold).get("jobs", [])
        if not isinstance(jobs, list):
            raise RuntimeError("Slurm API returned an invalid jobs response.")

        for job in jobs:
            if not isinstance(job, dict):
                continue
            response_job_id = job.get("job_id")
            if isinstance(response_job_id, dict):
                if response_job_id.get("set") is False:
                    continue
                response_job_id = response_job_id.get("number", 0)
            if not isinstance(response_job_id, (str, int, float)):
                continue
            try:
                if int(response_job_id) == job_id:
                    return job
            except (TypeError, ValueError):
                continue
        return None

    def job_states(self, job_id: int, retry_threshold: int = 3) -> list[str]:
        """Return job and step states from slurmdbd."""
        job = self._get_job(job_id, retry_threshold)
        if job is None:
            return []

        steps = job.get("steps", [])
        records = [job, *(step for step in steps if isinstance(step, dict))] if isinstance(steps, list) else [job]
        states: list[str] = []
        for record in records:
            raw_states = record.get("state", record.get("job_state"))
            if isinstance(raw_states, dict):
                raw_states = raw_states.get("current", [])
            if isinstance(raw_states, list):
                record_states = raw_states
            elif raw_states is None:
                record_states = []
            else:
                record_states = re.split(r"[,+]", str(raw_states))
            states.extend(str(state).upper().rstrip("+") for state in record_states if state)
        return states

    def is_job_completed(self, job_id: int, retry_threshold: int = 3) -> bool:
        """Return whether accounting reports any terminal state and no running state."""
        states = self.job_states(job_id, retry_threshold)
        if "RUNNING" in states:
            return False
        return any(state in self._TERMINAL_JOB_STATES for state in states)

    @staticmethod
    def _make_step_metadata(job_id: int, record: dict[str, Any], *, is_job: bool) -> SlurmStepMetadata:  # noqa: C901
        step = record.get("step", {}) if isinstance(record.get("step"), dict) else {}
        times = record.get("time", {}) if isinstance(record.get("time"), dict) else {}

        raw_states = record.get("state", record.get("job_state"))
        if isinstance(raw_states, dict):
            raw_states = raw_states.get("current", [])
        if isinstance(raw_states, list):
            states = [str(state).upper().rstrip("+") for state in raw_states if state]
        elif raw_states is None:
            states = []
        else:
            states = [state.upper().rstrip("+") for state in re.split(r"[,+]", str(raw_states)) if state]

        exit_code = record.get("exit_code")
        if isinstance(exit_code, str):
            formatted_exit_code = exit_code
        elif isinstance(exit_code, dict):
            return_code = exit_code.get("return_code", 0)
            if isinstance(return_code, dict):
                return_code = return_code.get("number", 0) if return_code.get("set") is not False else 0
            signal = exit_code.get("signal", 0)
            if isinstance(signal, dict):
                signal = signal.get("id", signal.get("signal_id", signal.get("number", 0)))
            try:
                parsed_return_code = int(return_code) if isinstance(return_code, (str, int, float)) else 0
                parsed_signal = int(signal) if isinstance(signal, (str, int, float)) else 0
                formatted_exit_code = f"{parsed_return_code}:{parsed_signal}"
            except (TypeError, ValueError):
                formatted_exit_code = "0:0"
        else:
            formatted_exit_code = "0:0"

        formatted_times: list[str] = []
        for field in ("start", "end"):
            raw_time = times.get(field)
            if isinstance(raw_time, str) and not raw_time.isdigit():
                formatted_times.append(raw_time)
                continue
            if isinstance(raw_time, dict):
                raw_time = raw_time.get("number", 0) if raw_time.get("set") is not False else 0
            try:
                timestamp = int(raw_time) if isinstance(raw_time, (str, int, float)) else 0
            except (TypeError, ValueError):
                timestamp = 0
            formatted_times.append(
                datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                if timestamp
                else ""
            )

        elapsed = times.get("elapsed", 0)
        if isinstance(elapsed, dict):
            elapsed = elapsed.get("number", 0) if elapsed.get("set") is not False else 0
        try:
            elapsed_seconds = int(elapsed) if isinstance(elapsed, (str, int, float)) else 0
        except (TypeError, ValueError):
            elapsed_seconds = 0

        return SlurmStepMetadata(
            job_id=job_id,
            step_id="" if is_job else str(step.get("id", "")),
            name=str(record.get("name", step.get("name", ""))),
            state=states[0] if states else "",
            exit_code=formatted_exit_code,
            start_time=formatted_times[0],
            end_time=formatted_times[1],
            elapsed_time_sec=elapsed_seconds,
            submit_line=str(record.get("submit_line", "")),
        )

    def get_job_status(self, job_id: int, retry_threshold: int = 3) -> list[SlurmStepMetadata]:
        job = self._get_job(job_id, retry_threshold)
        if job is None:
            return []

        response_job_id = job.get("job_id")
        if isinstance(response_job_id, dict):
            response_job_id = response_job_id.get("number", 0) if response_job_id.get("set") is not False else 0
        try:
            metadata_job_id = int(response_job_id) if isinstance(response_job_id, (str, int, float)) else 0
        except (TypeError, ValueError):
            metadata_job_id = 0

        steps = job.get("steps", [])
        records = [job, *(step for step in steps if isinstance(step, dict))] if isinstance(steps, list) else [job]
        return [
            self._make_step_metadata(metadata_job_id, record, is_job=index == 0) for index, record in enumerate(records)
        ]

    def get_job_nodes(self, job_id: int) -> str:
        job = self._get_job(job_id)
        return str(job.get("nodes", "")) if job else ""

    def cancel(self, job_id: int) -> None:
        """Cancel a Slurm job through slurmctld."""
        self._request("DELETE", "slurm", f"job/{job_id}")

    def validate(self) -> None:
        """Verify access to slurmctld and slurmdbd endpoints used by CloudAI."""
        self._request("GET", "slurm", "ping/")
        self._request("GET", "slurmdb", "clusters/")
