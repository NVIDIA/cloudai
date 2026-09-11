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

import logging
import math
import os
import re
import shlex
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import requests
from pydantic import BaseModel, ConfigDict, Field, field_validator
from tenacity import Retrying, before_sleep_log, retry_if_exception_type, stop_after_attempt, wait_fixed

from cloudai.core import JobIdRetrievalError
from cloudai.util import parse_time_limit

from .slurm_metadata import SlurmStepMetadata

logger = logging.getLogger(__name__)


class SlurmAPIConfig(BaseModel):
    """Connection details for a Slurm REST API endpoint."""

    model_config = ConfigDict(extra="forbid")

    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    verify_certs: bool = True

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value:
            raise ValueError("slurm_api.url must be non-blank")
        return value


class SlurmRestClient:
    """Translate CloudAI Slurm operations to slurmrestd v0.0.38 requests."""

    API_VERSION: ClassVar[str] = "v0.0.38"
    REQUEST_TIMEOUT_SECONDS: ClassVar[int] = 30
    TERMINAL_JOB_STATES: ClassVar[frozenset[str]] = frozenset(
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
    DIRECTIVE_FIELDS: ClassVar[dict[str, str]] = {
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
    }

    def __init__(self, config: SlurmAPIConfig, retry_pause_seconds: int) -> None:
        self.config = config
        self.retry_pause_seconds = retry_pause_seconds

    def _headers(self) -> dict[str, str]:
        """Expand environment variables in configured headers."""
        headers: dict[str, str] = {}
        for name, value in self.config.headers.items():
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
        """Send and validate one request; retry policy is applied by `request`."""
        url = f"{self.config.url}/{service}/{self.API_VERSION}/{path.lstrip('/')}"
        response = requests.request(
            method,
            url,
            headers=self._headers(),
            json=payload,
            timeout=self.REQUEST_TIMEOUT_SECONDS,
            verify=self.config.verify_certs,
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

    def request(
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

        retrying = Retrying(
            stop=stop_after_attempt(retry_threshold),
            wait=wait_fixed(self.retry_pause_seconds),
            retry=retry_if_exception_type((requests.RequestException, ValueError, RuntimeError)),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        try:
            return retrying(self._request_once, method, service, path, payload)
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            raise RuntimeError(f"Slurm API request failed after {retry_threshold} attempt(s): {exc}") from exc

    @staticmethod
    def _directive_value(args: list[str], index: int, option: str) -> tuple[str, int]:
        """Read one SBATCH value and next index; e.g. `(["--time", "10"], 0, "--time")` returns `("10", 2)`."""
        token = args[index]
        if "=" in token:
            return token.split("=", 1)[1], index + 1
        if option == "--nodes" and token.startswith("-N") and token != "-N":
            return token[2:], index + 1
        if index + 1 >= len(args):
            raise ValueError(f"SBATCH directive '{option}' requires a value.")
        return args[index + 1], index + 2

    @staticmethod
    def _gpu_gres(value: str, *, from_gres: bool) -> str:
        """Normalize GPU requests; e.g. `--gpus-per-node=8` becomes REST GRES `gpu:8`."""
        return value if from_gres else f"gpu:{value}"

    @staticmethod
    def _set_directive(job: dict[str, object], field: str, value: object, option: str) -> None:
        """Set one REST field, rejecting conflicting aliases; e.g. `--gres` and `--gpus-per-node` must agree."""
        existing = job.get(field)
        if existing is not None and existing != value:
            raise ValueError(f"Conflicting SBATCH directives for '{option}'.")
        job[field] = value

    def _apply_sbatch_args(self, job: dict[str, object], args: list[str]) -> None:  # noqa: C901
        """Map tokenized SBATCH directives into v0.0.38 fields; e.g. `["-N", "2"]` sets `nodes=[2, 2]`."""
        index = 0
        while index < len(args):
            token = args[index]
            option = token.split("=", 1)[0]
            if token.startswith("-N"):
                option = "--nodes"
            elif option == "-n":
                option = "--ntasks"
            elif option == "-D":
                option = "--chdir"
            value, index = self._directive_value(args, index, option)

            if option in self.DIRECTIVE_FIELDS:
                self._set_directive(job, self.DIRECTIVE_FIELDS[option], value, option)
            elif option == "--nodes":
                node_counts = [int(item) for item in str(value).split("-", 1)]
                if len(node_counts) == 1:
                    node_counts.append(node_counts[0])
                self._set_directive(job, "nodes", node_counts, option)
            elif option == "--nodelist":
                self._set_directive(job, "nodelist", str(value), option)
            elif option == "--exclude":
                self._set_directive(job, "exclude_nodes", str(value), option)
            elif option == "--ntasks":
                self._set_directive(job, "tasks", int(value), option)
            elif option == "--ntasks-per-node":
                self._set_directive(job, "tasks_per_node", int(value), option)
            elif option == "--time":
                minutes = (
                    int(value) if str(value).isdigit() else math.ceil(parse_time_limit(str(value)).total_seconds() / 60)
                )
                self._set_directive(job, "time_limit", minutes, option)
            elif option in {"--gres", "--gpus-per-node"}:
                gres = self._gpu_gres(str(value), from_gres=option == "--gres")
                self._set_directive(job, "gres", gres, option)
            elif option == "--chdir":
                self._set_directive(job, "current_working_directory", value, option)
            else:
                raise ValueError(f"SBATCH directive '{option}' is not supported by CloudAI's Slurm REST transport.")

    def _job_description(self, script: str, script_path: Path) -> dict[str, object]:
        """Build REST job properties from leading `#SBATCH` lines; script body remains unchanged."""
        job: dict[str, object] = {}
        for line in script.splitlines():
            stripped = line.strip()
            if not stripped or (stripped.startswith("#") and not stripped.startswith("#SBATCH")):
                continue
            if not stripped.startswith("#SBATCH"):
                break
            args = shlex.split(stripped.removeprefix("#SBATCH").strip())
            self._apply_sbatch_args(job, args)

        job.setdefault("current_working_directory", str(script_path.parent.absolute()))
        job["environment"] = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")}
        return job

    def submit_sbatch(
        self, script_path: Path, operation_name: str, *, wait: bool = False, monitor_interval: int = 1
    ) -> int:
        """Submit an SBATCH file and optionally wait for a terminal accounting state."""
        try:
            script = script_path.read_text(encoding="utf-8")
            data = self.request(
                "POST",
                "slurm",
                "job/submit",
                payload={"script": script, "job": self._job_description(script, script_path)},
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise JobIdRetrievalError(
                test_name=operation_name,
                command=f"POST /slurm/{self.API_VERSION}/job/submit",
                stdout="",
                stderr=str(exc),
                message="Failed to submit job through Slurm REST API.",
            ) from exc

        job_id = data.get("job_id")
        if not isinstance(job_id, int):
            raise JobIdRetrievalError(
                test_name=operation_name,
                command=f"POST /slurm/{self.API_VERSION}/job/submit",
                stdout=str(data),
                stderr="",
                message="Failed to retrieve job ID.",
            )

        if wait:
            while not self.is_job_completed(job_id):
                time.sleep(monitor_interval)
        return job_id

    @staticmethod
    def values(value: object) -> list[str]:
        """Normalize Slurm scalar/list wrappers; e.g. `{"current": "IDLE+DRAIN"}` becomes `["IDLE", "DRAIN"]`."""
        if isinstance(value, dict):
            value = value.get("current", [])
        if isinstance(value, list):
            return [str(item) for item in value]
        if value is None:
            return []
        return [item for item in re.split(r"[,+]", str(value)) if item]

    @classmethod
    def states(cls, record: dict[str, Any]) -> list[str]:
        """Extract normalized states; e.g. `{"job_state": "running+"}` becomes `["RUNNING"]`."""
        value = record.get("state", record.get("job_state"))
        return [state.upper().rstrip("+") for state in cls.values(value)]

    @staticmethod
    def _number(value: object) -> int:
        """Decode Slurm number wrappers; e.g. `{"set": true, "number": 12}` becomes `12`."""
        if isinstance(value, dict):
            if value.get("set") is False:
                return 0
            value = value.get("number", 0)
        if not isinstance(value, (str, int, float)):
            return 0
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _time(cls, value: object) -> str:
        """Normalize Slurm time values; e.g. epoch `100` becomes a UTC ISO-8601 timestamp."""
        if isinstance(value, str) and not value.isdigit():
            return value
        timestamp = cls._number(value)
        if not timestamp:
            return ""
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @classmethod
    def _exit_code(cls, record: dict[str, Any]) -> str:
        """Normalize composite exit status; e.g. return code `1` plus signal `9` becomes `"1:9"`."""
        exit_code = record.get("exit_code")
        if isinstance(exit_code, str):
            return exit_code
        if not isinstance(exit_code, dict):
            return "0:0"
        return_code = cls._number(exit_code.get("return_code"))
        signal_value = exit_code.get("signal")
        if isinstance(signal_value, dict):
            signal_value = signal_value.get("id", signal_value.get("signal_id", signal_value))
        signal = cls._number(signal_value)
        return f"{return_code}:{signal}"

    @staticmethod
    def _records(data: dict[str, Any], field: str) -> list[dict[str, Any]]:
        """Read object records; e.g. `jobs` returns dictionary entries from `data["jobs"]`."""
        records = data.get(field, [])
        if not isinstance(records, list):
            raise RuntimeError(f"Slurm API returned an invalid {field} response.")
        return [record for record in records if isinstance(record, dict)]

    def cluster_nodes(self) -> list[dict[str, Any]]:
        """Return node records from slurmctld."""
        return self._records(self.request("GET", "slurm", "nodes/"), "nodes")

    def queue_jobs(self) -> list[dict[str, Any]]:
        """Return current job records from slurmctld."""
        return self._records(self.request("GET", "slurm", "jobs/"), "jobs")

    def accounting_job(self, job_id: int, retry_threshold: int = 3) -> dict[str, Any] | None:
        """Return one job from slurmdbd, retrying while accounting catches up."""
        data = self.request("GET", "slurmdb", f"job/{job_id}", retry_threshold=retry_threshold)
        return next((job for job in self._records(data, "jobs") if self._number(job.get("job_id")) == job_id), None)

    def job_states(self, job_id: int, retry_threshold: int = 3) -> list[str]:
        """Return job and step states from slurmdbd."""
        job = self.accounting_job(job_id, retry_threshold)
        if job is None:
            return []
        states = self.states(job)
        steps = job.get("steps", [])
        if isinstance(steps, list):
            for step in steps:
                if isinstance(step, dict):
                    states.extend(self.states(step))
        return states

    def is_job_completed(self, job_id: int, retry_threshold: int = 3) -> bool:
        """Return whether accounting reports any terminal state and no running state."""
        states = self.job_states(job_id, retry_threshold)
        if "RUNNING" in states:
            return False
        return any(state in self.TERMINAL_JOB_STATES for state in states)

    @classmethod
    def step_metadata(cls, job: dict[str, Any]) -> list[SlurmStepMetadata]:
        """Convert one accounting job and its steps to CloudAI metadata records."""
        job_id = cls._number(job.get("job_id"))
        steps = job.get("steps", [])
        records = [job, *(step for step in steps if isinstance(step, dict))] if isinstance(steps, list) else [job]
        metadata: list[SlurmStepMetadata] = []
        for index, record in enumerate(records):
            step = record.get("step", {}) if isinstance(record.get("step"), dict) else {}
            times = record.get("time", {}) if isinstance(record.get("time"), dict) else {}
            states = cls.states(record)
            metadata.append(
                SlurmStepMetadata(
                    job_id=job_id,
                    step_id="" if index == 0 else str(step.get("id", "")),
                    name=str(record.get("name", step.get("name", ""))),
                    state=states[0] if states else "",
                    exit_code=cls._exit_code(record),
                    start_time=cls._time(times.get("start")),
                    end_time=cls._time(times.get("end")),
                    elapsed_time_sec=cls._number(times.get("elapsed")),
                    submit_line=str(record.get("submit_line", "")),
                )
            )
        return metadata

    def cancel(self, job_id: int) -> None:
        """Cancel a Slurm job through slurmctld."""
        self.request("DELETE", "slurm", f"job/{job_id}")

    def validate(self) -> None:
        """Verify access to slurmctld and slurmdbd endpoints used by CloudAI."""
        self.request("GET", "slurm", "ping/")
        self.request("GET", "slurmdb", "clusters/")
