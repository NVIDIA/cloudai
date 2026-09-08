# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import shutil
import subprocess
import time
from copy import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Union

import requests
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from cloudai.core import BaseJob, File, Installable, JobIdRetrievalError, System
from cloudai.models.scenario import ReportConfig, parse_reports_spec
from cloudai.util import CommandShell, parse_time_limit

from .slurm_job import SlurmJob
from .slurm_metadata import SlurmStepMetadata
from .slurm_node import SlurmNode, SlurmNodeState


class DataRepositoryConfig(BaseModel):
    """Configuration for a data repository."""

    endpoint: str
    verify_certs: bool = True


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


def parse_node_list(node_list: str) -> List[str]:
    """
    Expand a list of node names (with ranges) into a flat list of individual node names, keeping leading zeroes.

    Args:
        node_list (str): A list of node names, possibly including ranges.

    Returns:
        List[str]: A flat list of expanded node names with preserved zeroes.
    """
    node_list = node_list.strip()
    nodes = []
    if not node_list:
        return []

    components = re.split(r",\s*(?![^[]*\])", node_list)
    for component in components:
        if "[" not in component:
            nodes.append(component)
        else:
            header, node_number = component.split("[")
            node_number = node_number.replace("]", "")
            ranges = node_number.split(",")
            for r in ranges:
                if "-" in r:
                    start_node, end_node = r.split("-")
                    number_of_digits = len(end_node)
                    nodes.extend(
                        [f"{header}{str(i).zfill(number_of_digits)}" for i in range(int(start_node), int(end_node) + 1)]
                    )
                else:
                    nodes.append(f"{header}{r}")

    return nodes


class SlurmGroup(BaseModel):
    """Represents a group of nodes within a partition."""

    model_config = ConfigDict(extra="forbid")
    name: str
    nodes: List[str]


class SlurmPartition(BaseModel):
    """Represents a partition within a Slurm system."""

    model_config = ConfigDict(extra="forbid")
    name: str
    groups: List[SlurmGroup] = []
    slurm_nodes: list[SlurmNode] = Field(default_factory=list[SlurmNode], exclude=True)


class SlurmSystem(System):
    """Represents a Slurm system."""

    def submit_sbatch(self, script_path: Path, operation_name: str, *, wait: bool = False) -> int:
        """Submit an sbatch script without exposing the CLI transport to callers."""
        if self.uses_slurm_api:
            return self._submit_sbatch_rest(script_path, operation_name, wait=wait)

        wait_arg = " --wait" if wait else ""
        command = f"sbatch{wait_arg} {shlex.quote(str(script_path))}"
        return self.submit_job(command, operation_name)

    default_partition: str
    partitions: List[SlurmPartition]
    account: Optional[str] = None
    distribution: Optional[str] = None
    mpi: str = "pmix"
    gpus_per_node: Optional[int] = None
    ntasks_per_node: Optional[int] = None
    cache_docker_images_locally: bool = False
    scheduler: str = "slurm"
    monitor_interval: int = 60
    cmd_shell: CommandShell = Field(default_factory=CommandShell, exclude=True)
    extra_srun_args: Optional[str] = None
    extra_sbatch_args: list[str] = Field(default_factory=list)
    extra_transient_status_errors: list[str] = Field(default_factory=list)
    status_retry_pause_seconds: int = Field(default=10, ge=0)
    supports_gpu_directives_cache: Optional[bool] = Field(default=None, exclude=True)
    container_mount_home: bool = False
    slurm_api: Optional[SlurmAPIConfig] = None

    data_repository: Optional[DataRepositoryConfig] = None
    reports: Optional[dict[str, ReportConfig]] = None

    group_allocated: set[SlurmNode] = Field(default_factory=set, exclude=True)

    _REQUIRED_BINARIES: ClassVar[tuple[str, ...]] = (
        "git",
        "sbatch",
        "sinfo",
        "squeue",
        "srun",
        "scancel",
        "sacct",
    )
    _REQUIRED_SRUN_OPTIONS: ClassVar[tuple[str, ...]] = (
        "--mpi",
        "--gpus-per-node",
        "--ntasks-per-node",
        "--container-image",
        "--container-mounts",
    )
    _REST_API_VERSION: ClassVar[str] = "v0.0.43"
    _REST_TIMEOUT_SECONDS: ClassVar[int] = 30
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
    }

    @field_validator("reports", mode="before")
    @classmethod
    def parse_reports(cls, value: dict[str, Any] | None) -> dict[str, ReportConfig] | None:
        return parse_reports_spec(value)

    @field_validator("extra_transient_status_errors")
    @classmethod
    def _reject_blank_transient_patterns(cls, value: list[str]) -> list[str]:
        if any(not pattern.strip() for pattern in value):
            raise ValueError("extra_transient_status_errors entries must be non-blank")
        return value

    @property
    def uses_slurm_api(self) -> bool:
        """Whether Slurm communication uses slurmrestd instead of local CLI tools."""
        return self.slurm_api is not None

    def _rest_headers(self) -> dict[str, str]:
        assert self.slurm_api is not None
        headers: dict[str, str] = {}
        for name, value in self.slurm_api.headers.items():
            expanded = os.path.expandvars(value)
            if re.search(r"\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[^}]+\})", expanded):
                raise EnvironmentError(f"Environment variable referenced by Slurm API header '{name}' is not set.")
            headers[name] = expanded
        return headers

    @staticmethod
    def _rest_message(item: object) -> str:
        if not isinstance(item, dict):
            return str(item)
        return str(item.get("error") or item.get("description") or item)

    def _rest_request(
        self,
        method: str,
        service: str,
        path: str,
        *,
        payload: dict[str, object] | None = None,
        retry_threshold: int = 1,
    ) -> dict[str, Any]:
        assert self.slurm_api is not None
        url = f"{self.slurm_api.url}/{service}/{self._REST_API_VERSION}/{path.lstrip('/')}"
        last_error = ""

        for attempt in range(retry_threshold):
            try:
                response = requests.request(
                    method,
                    url,
                    headers=self._rest_headers(),
                    json=payload,
                    timeout=self._REST_TIMEOUT_SECONDS,
                    verify=self.slurm_api.verify_certs,
                )
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise RuntimeError(f"Slurm API returned a non-object response from {url}.")
                if errors := data.get("errors"):
                    details = "; ".join(self._rest_message(error) for error in errors)
                    raise RuntimeError(f"Slurm API request failed: {details}")
                for warning in data.get("warnings", []):
                    logging.warning("Slurm API warning: %s", self._rest_message(warning))
                return data
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                last_error = str(exc)
                if attempt + 1 < retry_threshold:
                    logging.warning(
                        "Slurm API request failed; retrying (%d/%d): %s",
                        attempt + 1,
                        retry_threshold,
                        exc,
                    )
                    time.sleep(self.status_retry_pause_seconds)

        raise RuntimeError(f"Slurm API request failed after {retry_threshold} attempt(s): {last_error}")

    @staticmethod
    def _directive_value(args: list[str], index: int, option: str) -> tuple[str, int]:
        token = args[index]
        if "=" in token:
            return token.split("=", 1)[1], index + 1
        if option == "--nodes" and token.startswith("-N") and token != "-N":
            return token[2:], index + 1
        if index + 1 >= len(args):
            raise ValueError(f"SBATCH directive '{option}' requires a value.")
        return args[index + 1], index + 2

    @staticmethod
    def _gpu_tres(value: str, *, from_gres: bool) -> str:
        if from_gres:
            return ",".join(item if item.startswith("gres/") else f"gres/{item}" for item in value.split(","))
        return f"gres/gpu:{value}"

    @staticmethod
    def _set_directive(job: dict[str, object], field: str, value: object, option: str) -> None:
        existing = job.get(field)
        if existing is not None and existing != value:
            raise ValueError(f"Conflicting SBATCH directives for '{option}'.")
        job[field] = value

    def _apply_sbatch_args(self, job: dict[str, object], args: list[str]) -> None:  # noqa: C901
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

            if option in self._DIRECTIVE_FIELDS:
                self._set_directive(job, self._DIRECTIVE_FIELDS[option], value, option)
            elif option == "--nodes":
                self._set_directive(job, "nodes", str(value), option)
            elif option == "--nodelist":
                self._set_directive(job, "required_nodes", str(value).split(","), option)
            elif option == "--exclude":
                self._set_directive(job, "excluded_nodes", str(value).split(","), option)
            elif option == "--ntasks":
                self._set_directive(job, "tasks", int(value), option)
            elif option == "--ntasks-per-node":
                self._set_directive(job, "tasks_per_node", int(value), option)
            elif option == "--time":
                minutes = (
                    int(value) if str(value).isdigit() else math.ceil(parse_time_limit(str(value)).total_seconds() / 60)
                )
                self._set_directive(job, "time_limit", {"set": True, "number": minutes}, option)
            elif option in {"--gres", "--gpus-per-node"}:
                tres = self._gpu_tres(str(value), from_gres=option == "--gres")
                self._set_directive(job, "tres_per_node", tres, option)
            elif option in {"--chdir", "-D"}:
                self._set_directive(job, "current_working_directory", value, option)
            else:
                raise ValueError(f"SBATCH directive '{option}' is not supported by CloudAI's Slurm REST transport.")

    def _rest_job_description(self, script: str, script_path: Path) -> dict[str, object]:
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
        job["environment"] = [f"PATH={os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin')}"]
        return job

    def _submit_sbatch_rest(self, script_path: Path, operation_name: str, *, wait: bool = False) -> int:
        try:
            script = script_path.read_text(encoding="utf-8")
            data = self._rest_request(
                "POST",
                "slurm",
                "job/submit",
                payload={"script": script, "job": self._rest_job_description(script, script_path)},
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise JobIdRetrievalError(
                test_name=operation_name,
                command=f"POST /slurm/{self._REST_API_VERSION}/job/submit",
                stdout="",
                stderr=str(exc),
                message="Failed to submit job through Slurm REST API.",
            ) from exc

        job_id = data.get("job_id")
        if not isinstance(job_id, int):
            raise JobIdRetrievalError(
                test_name=operation_name,
                command=f"POST /slurm/{self._REST_API_VERSION}/job/submit",
                stdout=str(data),
                stderr="",
                message="Failed to retrieve job ID.",
            )

        if wait:
            while not self._is_rest_job_completed(job_id):
                time.sleep(self.monitor_interval)
        return job_id

    @staticmethod
    def _rest_values(value: object) -> list[str]:
        if isinstance(value, dict):
            value = value.get("current", [])
        if isinstance(value, list):
            return [str(item) for item in value]
        if value is None:
            return []
        return [item for item in re.split(r"[,+]", str(value)) if item]

    @classmethod
    def _rest_states(cls, record: dict[str, Any]) -> list[str]:
        value = record.get("state", record.get("job_state"))
        return [state.upper().rstrip("+") for state in cls._rest_values(value)]

    def _rest_node_state(self, node: dict[str, Any]) -> SlurmNodeState:
        states = [self.convert_state_to_enum(state) for state in self._rest_states(node)]
        ordinary_states = {
            SlurmNodeState.ALLOCATED,
            SlurmNodeState.ALLOCATED_COMPLETING,
            SlurmNodeState.COMPLETING,
            SlurmNodeState.IDLE,
            SlurmNodeState.MIXED_ALLOCATION,
        }
        fallback = states[0] if states else SlurmNodeState.UNKNOWN_STATE
        return next((state for state in states if state not in ordinary_states), fallback)

    @staticmethod
    def _rest_number(value: object) -> int:
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
    def _rest_time(cls, value: object) -> str:
        if isinstance(value, str) and not value.isdigit():
            return value
        timestamp = cls._rest_number(value)
        if not timestamp:
            return ""
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @classmethod
    def _rest_exit_code(cls, record: dict[str, Any]) -> str:
        exit_code = record.get("exit_code")
        if isinstance(exit_code, str):
            return exit_code
        if not isinstance(exit_code, dict):
            return "0:0"
        return_code = cls._rest_number(exit_code.get("return_code"))
        signal_value = exit_code.get("signal")
        if isinstance(signal_value, dict):
            signal_value = signal_value.get("id", signal_value)
        signal = cls._rest_number(signal_value)
        return f"{return_code}:{signal}"

    def _rest_accounting_job(self, job_id: int, retry_threshold: int = 3) -> dict[str, Any] | None:
        data = self._rest_request("GET", "slurmdb", f"job/{job_id}", retry_threshold=retry_threshold)
        jobs = data.get("jobs", [])
        if not isinstance(jobs, list):
            raise RuntimeError("Slurm API returned an invalid jobs response.")
        return next(
            (item for item in jobs if isinstance(item, dict) and self._rest_number(item.get("job_id")) == job_id),
            None,
        )

    def _rest_job_states(self, job_id: int, retry_threshold: int = 3) -> list[str]:
        job = self._rest_accounting_job(job_id, retry_threshold)
        if job is None:
            return []
        states = self._rest_states(job)
        for step in job.get("steps", []):
            if isinstance(step, dict):
                states.extend(self._rest_states(step))
        return states

    def _is_rest_job_completed(self, job_id: int, retry_threshold: int = 3) -> bool:
        states = self._rest_job_states(job_id, retry_threshold)
        if "RUNNING" in states:
            return False
        return any(state in self._TERMINAL_JOB_STATES for state in states)

    @classmethod
    def _rest_step_metadata(cls, job: dict[str, Any]) -> list[SlurmStepMetadata]:
        job_id = cls._rest_number(job.get("job_id"))
        records = [job, *(step for step in job.get("steps", []) if isinstance(step, dict))]
        metadata: list[SlurmStepMetadata] = []
        for index, record in enumerate(records):
            step = record.get("step", {}) if isinstance(record.get("step"), dict) else {}
            times = record.get("time", {}) if isinstance(record.get("time"), dict) else {}
            states = cls._rest_states(record)
            metadata.append(
                SlurmStepMetadata(
                    job_id=job_id,
                    step_id="" if index == 0 else str(step.get("id", "")),
                    name=str(record.get("name", step.get("name", ""))),
                    state=states[0] if states else "",
                    exit_code=cls._rest_exit_code(record),
                    start_time=cls._rest_time(times.get("start")),
                    end_time=cls._rest_time(times.get("end")),
                    elapsed_time_sec=cls._rest_number(times.get("elapsed")),
                    submit_line=str(record.get("submit_line", "")),
                )
            )
        return metadata

    @property
    def groups(self) -> Dict[str, Dict[str, List[SlurmNode]]]:
        groups: Dict[str, Dict[str, List[SlurmNode]]] = {}
        for part in self.partitions:
            groups[part.name] = {}
            for group in part.groups:
                node_names = set()
                for group_nodes in group.nodes:
                    node_names.update(set(parse_node_list(group_nodes)))

                groups[part.name][group.name] = []
                for node_name in node_names:
                    node_in_partition = next((node for node in part.slurm_nodes if node.name == node_name), None)
                    if not node_in_partition:
                        logging.error(f"Node '{node_name}' not found in partition '{part.name}'")
                        groups[part.name][group.name].append(
                            SlurmNode(name=node_name, partition=self.name, state=SlurmNodeState.UNKNOWN_STATE)
                        )
                    else:
                        groups[part.name][group.name].append(node_in_partition)

        return groups

    @property
    def supports_gpu_directives(self) -> bool:
        if self.supports_gpu_directives_cache is not None:
            return self.supports_gpu_directives_cache

        if self.uses_slurm_api:
            try:
                data = self._rest_request("GET", "slurm", "nodes/")
            except RuntimeError as exc:
                logging.warning("Error checking GPU support: %s", exc)
                self.supports_gpu_directives_cache = True
                return True

            self.supports_gpu_directives_cache = any(
                "gpu" in str(node.get(field, "")).lower()
                for node in data.get("nodes", [])
                if isinstance(node, dict)
                for field in ("gres", "tres")
            )
            return self.supports_gpu_directives_cache

        stdout, stderr = self.fetch_command_output("scontrol show config")
        if stderr:
            logging.warning(f"Error checking GPU support: {stderr}")
            self.supports_gpu_directives_cache = True
            return True

        for line in stdout.splitlines():
            if "GresTypes" in line and "gpu" in line:
                self.supports_gpu_directives_cache = True
                return True

        self.supports_gpu_directives_cache = False
        return False

    @field_serializer("install_path", "output_path")
    def _path_serializer(self, v: Path) -> str:
        return str(v)

    def update(self) -> None:
        """
        Update the system object for a SLURM system.

        This method updates the system object by querying the current state of each node using the 'sinfo' and 'squeue'
        commands, and correlating this information to determine the state of each node and the user running jobs on
        each node.
        """
        all_nodes = self.nodes_from_sinfo()
        self.update_nodes_state_and_user(all_nodes, insert_new=True)
        self.update_nodes_state_and_user(self.nodes_from_squeue())
        self.update_nodes_state_and_user(self.group_allocated)

    def nodes_from_sinfo(self) -> list[SlurmNode]:
        if self.uses_slurm_api:
            data = self._rest_request("GET", "slurm", "nodes/")
            nodes: list[SlurmNode] = []
            for node in data.get("nodes", []):
                if not isinstance(node, dict) or not node.get("name"):
                    continue
                state = self._rest_node_state(node)
                for partition in self._rest_values(node.get("partitions")):
                    nodes.append(SlurmNode(name=str(node["name"]), partition=partition, state=state))
            return nodes

        sinfo_output, _ = self.fetch_command_output("sinfo --noheader -o '%P|%t|%u|%N'")
        nodes: list[SlurmNode] = []
        for line in sinfo_output.split("\n"):
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            partition, state, user, nodelist = parts[:4]
            partition = partition.rstrip("*").strip()
            node_names = parse_node_list(nodelist)
            logging.debug(f"{partition=}, {state=}, {nodelist=}, {node_names=}")
            for node_name in node_names:
                nodes.append(
                    SlurmNode(name=node_name, partition=partition, state=self.convert_state_to_enum(state), user=user)
                )
        return nodes

    def nodes_from_squeue(self) -> list[SlurmNode]:
        if self.uses_slurm_api:
            data = self._rest_request("GET", "slurm", "jobs/")
            nodes: list[SlurmNode] = []
            for job in data.get("jobs", []):
                if not isinstance(job, dict) or not {"RUNNING", "PENDING"}.intersection(self._rest_states(job)):
                    continue
                partition = str(job.get("partition", ""))
                user = str(job.get("user_name", job.get("user", "N/A")))
                for node_name in parse_node_list(str(job.get("nodes", ""))):
                    nodes.append(
                        SlurmNode(
                            name=node_name,
                            partition=partition,
                            state=SlurmNodeState.ALLOCATED,
                            user=user,
                        )
                    )
            return nodes

        squeue_output, _ = self.fetch_command_output("squeue --states=running,pending --noheader -o '%P|%T|%N|%u'")
        nodes: list[SlurmNode] = []
        for line in squeue_output.split("\n"):
            parts = line.split("|")
            if len(parts) < 4:
                continue
            partition, _, nodelist, user = parts[:4]
            node_names = parse_node_list(nodelist)
            for node in node_names:
                nodes.append(SlurmNode(name=node, partition=partition, state=SlurmNodeState.ALLOCATED, user=user))
        return nodes

    def update_nodes_state_and_user(self, nodes: Iterable[SlurmNode], insert_new: bool = False) -> None:
        for node in nodes:
            for part in self.partitions:
                if part.name != node.partition:
                    continue

                found = False
                for pnode in part.slurm_nodes:
                    if pnode.name != node.name:
                        continue
                    pnode.state = node.state
                    pnode.user = node.user
                    found = True
                    break

                if not found and insert_new:
                    part.slurm_nodes.append(node)

    def _is_transient_status_error(self, stderr: str) -> bool:
        """
        Return True if a job-status query failed with a retryable, transient error.

        Covers slurm's own transient failures plus any site-specific patterns
        configured via ``extra_transient_status_errors`` (e.g. errors emitted
        by a proxy or shim wrapping the slurm CLIs).
        """
        patterns = [
            "Socket timed out",
            "slurm_load_jobs error",
            *(pattern for pattern in self.extra_transient_status_errors if pattern.strip()),
        ]
        return any(p in stderr for p in patterns)

    @staticmethod
    def _parse_submitted_job_id(stdout: str) -> int | None:
        match = re.search(r"Submitted batch job (\d+)", stdout)
        if match:
            return int(match.group(1))

        # Some launchers submit Slurm jobs themselves and use this output format.
        match = re.search(r"submitted with Job ID (\d+)", stdout)
        return int(match.group(1)) if match else None

    def submit_job(self, submission_command: str, test_name: str) -> int:
        """Submit a generated Slurm workload and return its job ID."""
        if self.uses_slurm_api:
            args = shlex.split(submission_command)
            if not args or Path(args[0]).name != "sbatch":
                raise JobIdRetrievalError(
                    test_name=test_name,
                    command=submission_command,
                    stdout="",
                    stderr="Slurm REST mode only supports submission of an sbatch script.",
                    message="Failed to submit job through Slurm REST API.",
                )

            wait = "--wait" in args[1:-1]
            unsupported = [arg for arg in args[1:-1] if arg != "--wait"]
            if unsupported or len(args) < 2:
                raise JobIdRetrievalError(
                    test_name=test_name,
                    command=submission_command,
                    stdout="",
                    stderr=f"Unsupported sbatch command arguments: {' '.join(unsupported)}",
                    message="Failed to submit job through Slurm REST API.",
                )
            return self._submit_sbatch_rest(Path(args[-1]), test_name, wait=wait)

        stdout, stderr = self.cmd_shell.execute(submission_command).communicate()
        job_id = self._parse_submitted_job_id(stdout)
        if job_id is None:
            raise JobIdRetrievalError(
                test_name=test_name,
                command=submission_command,
                stdout=stdout,
                stderr=stderr,
                message="Failed to retrieve job ID.",
            )
        return job_id

    def validate_install_environment(self) -> None:
        """Validate that the configured Slurm environment can run CloudAI workloads."""
        if self.uses_slurm_api:
            if shutil.which("git") is None:
                raise EnvironmentError("Required binary 'git' is not installed.")
            try:
                self._rest_request("GET", "slurm", "ping/")
                self._rest_request("GET", "slurmdb", "ping/")
            except RuntimeError as exc:
                raise EnvironmentError(f"Failed to access the Slurm REST API: {exc}") from exc
            return

        for binary in self._REQUIRED_BINARIES:
            if shutil.which(binary) is None:
                raise EnvironmentError(f"Required binary '{binary}' is not installed.")

        try:
            result = subprocess.run(["srun", "--help"], text=True, capture_output=True, check=True)
        except subprocess.CalledProcessError as exc:
            raise EnvironmentError(f"Failed to execute 'srun --help': {exc}") from exc
        missing_options = [option for option in self._REQUIRED_SRUN_OPTIONS if option not in result.stdout]
        if missing_options:
            raise EnvironmentError(f"Required srun options missing: {', '.join(missing_options)}")

    def is_job_running(self, job: BaseJob, retry_threshold: int = 3) -> bool:
        """
        Determine if a specified Slurm job is currently running by checking its presence and state in the job queue.

        This method queries the Slurm job accounting using 'sacct' to identify if the job with the specified ID is
        running. It handles transient network or system errors by retrying the query a limited number of times.

        Args:
            job (BaseJob): The job to check.
            retry_threshold (int): Maximum number of retry attempts for the query in case of transient errors.

        Returns:
            bool: True if the job is currently running, False otherwise.

        Raises:
            RuntimeError: If an error occurs that prevents determination of the job's running status, or if the status
                        cannot be determined after the specified number of retries.
        """
        if self.uses_slurm_api:
            assert isinstance(job.id, int)
            return "RUNNING" in self._rest_job_states(job.id, retry_threshold)

        retry_count = 0
        command = f"sacct -j {job.id} --format=State --noheader"

        while retry_count < retry_threshold:
            stdout, stderr = self.cmd_shell.execute(command).communicate()
            logging.debug(f"Job running: {command=} {stdout=} {stderr=}")

            if self._is_transient_status_error(stderr):
                retry_count += 1
                logging.warning(
                    f"An error occurred while querying the job status. Retrying... ({retry_count}/{retry_threshold})."
                )
                if retry_count < retry_threshold:
                    time.sleep(self.status_retry_pause_seconds)
                continue

            if stderr:
                error_message = f"Error checking job status: {stderr}"
                logging.error(error_message)
                raise RuntimeError(error_message)

            job_states = stdout.strip().split()
            if "RUNNING" in job_states:
                return True

            break

        if retry_count == retry_threshold:
            error_message = f"Failed to confirm job running status after {retry_threshold} attempts."
            logging.error(error_message)
            raise RuntimeError(error_message)

        return False

    def is_job_completed(self, job: BaseJob, retry_threshold: int = 3) -> bool:
        """
        Check if a Slurm job is completed by querying its status.

        Retries the query a specified number of times if certain errors are encountered.

        Args:
            job (BaseJob): The job to check.
            retry_threshold (int): Maximum number of retries for transient errors.

        Returns:
            bool: True if the job is completed, False otherwise.

        Raises:
            RuntimeError: If unable to determine job status after retries, or if a non-retryable error is encountered.
        """
        if self.uses_slurm_api:
            assert isinstance(job.id, int)
            return self._is_rest_job_completed(job.id, retry_threshold)

        retry_count = 0
        command = f"sacct -j {job.id} --format=State --noheader"

        while retry_count < retry_threshold:
            stdout, stderr = self.cmd_shell.execute(command).communicate()
            logging.debug(f"Job completed: {command=} {stdout=} {stderr=}")

            if self._is_transient_status_error(stderr):
                retry_count += 1
                logging.warning(f"Retrying job status check (attempt {retry_count}/{retry_threshold})")
                if retry_count < retry_threshold:
                    time.sleep(self.status_retry_pause_seconds)
                continue

            if stderr:
                error_message = f"Error checking job status: {stderr}"
                logging.error(error_message)
                raise RuntimeError(error_message)

            job_states = stdout.strip().split()
            if "RUNNING" in job_states:
                return False

            if any(state in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "CANCELLED+"] for state in job_states):
                return True

            break

        if retry_count == retry_threshold:
            error_message = f"Failed to confirm job completion status after {retry_threshold} attempts."
            logging.error(error_message)
            raise RuntimeError(error_message)

        return False

    def get_job_status(self, job: BaseJob, retry_threshold: int = 3) -> list[SlurmStepMetadata]:
        if self.uses_slurm_api:
            assert isinstance(job.id, int)
            rest_job = self._rest_accounting_job(job.id, retry_threshold)
            return self._rest_step_metadata(rest_job) if rest_job else []

        retry_count = 0
        command = (
            f"sacct -j {job.id} --format=JobID,JobName,State,ExitCode,Start,End,ElapsedRAW,SubmitLine "
            "--delimiter='|' -p --noheader"
        )

        while retry_count < retry_threshold:
            stdout, stderr = self.cmd_shell.execute(command).communicate()
            logging.debug(f"Job status: {command=} {stdout=} {stderr=}")

            if self._is_transient_status_error(stderr):
                retry_count += 1
                logging.warning(f"Retrying job status check (attempt {retry_count}/{retry_threshold})")
                if retry_count < retry_threshold:
                    time.sleep(self.status_retry_pause_seconds)
                continue

            if stderr:
                error_message = f"Error checking job status: {stderr}"
                logging.error(error_message)
                raise RuntimeError(error_message)

            return SlurmStepMetadata.from_sacct_output(stdout, delimiter="|")

        return []

    def kill(self, job: BaseJob) -> None:
        """
        Terminate a Slurm job.

        Args:
            job (BaseJob): The job to be terminated.
        """
        assert isinstance(job.id, int)
        self.scancel(job.id)

    @classmethod
    def format_node_list(cls, node_names: List[str]) -> str:
        """
        Format a list of node names into a condensed string representing groups of nodes as ranges.

        Mimicking the compact display found in systems like Slurm's sinfo command output.

        Args:
            node_names: A list of node names, potentially including numerically sequential nodes that can be condensed
                into a range format.

        Returns:
            A string representing the condensed node list, with numerically adjacent nodes shown as ranges.
        """

        def extract_parts(name: str) -> tuple:
            """
            Extract the prefix and numeric part of a node name, along with the length of the numeric part.

            Zero-padding is used.

            Args:
                name: The node name to be parsed.

            Returns:
                A tuple containing the prefix (str), numeric part (int), and
                the length of the numeric part (int).
            """
            match = re.match(r"^(.*?-)(\d+)$", name)
            if not match:
                raise ValueError(f"Cannot extract numeric part from '{name}'")
            prefix, num = match.groups()
            return prefix, int(num), len(num)

        def format_range(lst: List[int], padding: int) -> List[str]:
            """
            Format a list of integers into string ranges, considering zero-padding.

            Args:
                lst: A sorted list of node numbers.
                padding: The number of digits for zero-padding the node numbers.

            Returns:
                A list of formatted string ranges.
            """
            if not lst:
                return []
            lst.sort()
            start = lst[0]
            end = lst[0]
            ranges = []
            for num in lst[1:]:
                if num == end + 1:
                    end = num
                else:
                    range_str = f"{start:0{padding}d}-{end:0{padding}d}" if start != end else f"{start:0{padding}d}"
                    ranges.append(range_str)
                    start = end = num
            range_str = f"{start:0{padding}d}-{end:0{padding}d}" if start != end else f"{start:0{padding}d}"
            ranges.append(range_str)
            return ranges

        nodes_by_prefix = {}
        for name in node_names:
            prefix, num, length = extract_parts(name)
            nodes_by_prefix.setdefault(prefix, {"nums": [], "padding": 0})
            nodes_by_prefix[prefix]["nums"].append(num)
            nodes_by_prefix[prefix]["padding"] = max(nodes_by_prefix[prefix]["padding"], length)

        formatted_ranges = []
        for prefix, details in nodes_by_prefix.items():
            ranges = format_range(details["nums"], details["padding"])
            range_str = f"[{','.join(ranges)}]" if ranges else ""
            formatted_ranges.append(f"{prefix}{range_str}")

        return ", ".join(formatted_ranges)

    def get_available_nodes_from_group(
        self,
        partition_name: str,
        group_name: str,
        number_of_nodes: Union[int, str],
        exclude_nodes: list[str] | None = None,
    ) -> List[SlurmNode]:
        """
        Retrieve a specific number of potentially available nodes from a group within a partition.

        Prioritizes nodes by their current state, preferring idle nodes first, then completing nodes, and finally
        allocated nodes, while excluding nodes that are down and allocated nodes to the current user.

        Args:
            partition_name (str): The name of the partition.
            group_name (str): The name of the group.
            number_of_nodes (Union[int,str]): The number of nodes to retrieve.
                Could also be 'all' to retrieve all the nodes from the group.
            exclude_nodes (list[str] | None): Node names to exclude from the pool before selection.

        Returns:
            List[SlurmNode]: Objects that are potentially available for use.

        Raises:
            ValueError: If the partition or group is not found, or if the requested number of nodes exceeds the
                available nodes.
        """
        self.update()

        self.validate_partition_and_group(partition_name, group_name)

        grouped_nodes = self.group_nodes_by_state(partition_name, group_name, exclude_nodes=exclude_nodes)

        try:
            allocated_nodes = self.allocate_nodes(grouped_nodes, number_of_nodes, group_name)

            logging.info(
                f"Allocated nodes from group '{group_name}' in partition '{partition_name}': "
                f"{[node.name for node in allocated_nodes]}"
            )

            return allocated_nodes

        except ValueError as e:
            logging.error(
                f"Error occurred while allocating nodes from group '{group_name}' in partition '{partition_name}': {e}",
                exc_info=True,
            )

            return []

    def validate_partition_and_group(self, partition_name: str, group_name: str) -> None:
        """
        Validate that the partition and group exist.

        Args:
            partition_name (str): The name of the partition.
            group_name (str): The name of the group.

        Raises:
            ValueError: If the partition or group is not found.

        """
        if partition_name not in self.groups:
            raise ValueError(f"Partition '{partition_name}' not found.")
        if group_name not in self.groups[partition_name]:
            raise ValueError(f"Group '{group_name}' not found in partition '{partition_name}'.")

    def group_nodes_by_state(
        self,
        partition_name: str,
        group_name: str,
        exclude_nodes: list[str] | None = None,
    ) -> Dict[SlurmNodeState, List[SlurmNode]]:
        """
        Group nodes by their states, excluding nodes allocated to the current user.

        Args:
            partition_name (str): The name of the partition.
            group_name (str): The name of the group.
            exclude_nodes (list[str] | None): Node names to exclude from the pool before grouping.

        Returns:
            Dict[SlurmNodeState, List[SlurmNode]]: A dictionary grouping nodes by their state.
        """
        grouped_nodes = {
            SlurmNodeState.IDLE: [],
            SlurmNodeState.COMPLETING: [],
            SlurmNodeState.ALLOCATED: [],
            SlurmNodeState.RESERVED: [],
        }

        for node in self.groups[partition_name][group_name]:
            if exclude_nodes and node.name in exclude_nodes:
                continue
            if node.state in grouped_nodes:
                grouped_nodes[node.state].append(node)

        logging.debug(f"Grouped nodes by state: {grouped_nodes}")

        return grouped_nodes

    def allocate_nodes(
        self, grouped_nodes: Dict[SlurmNodeState, List[SlurmNode]], number_of_nodes: Union[int, str], group_name: str
    ) -> List[SlurmNode]:
        """
        Allocate nodes based on the requested number or maximum availability.

        Args:
            grouped_nodes (Dict[SlurmNodeState, List[SlurmNode]]): Nodes grouped by their state.
            number_of_nodes (Union[int, str]): The number of nodes to allocate, or 'max_avail' to allocate
                all available nodes.
            group_name (str): The name of the group.

        Returns:
            List[SlurmNode]: A list of allocated nodes.

        Raises:
            ValueError: If the requested number of nodes exceeds the available nodes.
        """
        allocated_nodes = []

        if isinstance(number_of_nodes, str) and number_of_nodes == "max_avail":
            allocated_nodes.extend(grouped_nodes[SlurmNodeState.IDLE])
            allocated_nodes.extend(grouped_nodes[SlurmNodeState.COMPLETING])
            allocated_nodes.extend(grouped_nodes[SlurmNodeState.RESERVED])

            if len(allocated_nodes) == 0:
                raise ValueError(
                    f"CloudAI is requesting the maximum available nodes from the group '{group_name}', "
                    f"but no nodes are available. Please review the available nodes in the system and ensure "
                    f"there are sufficient resources to meet the requirements of the test scenario. Additionally, "
                    f"verify that the system is capable of hosting the maximum number of nodes specified in the test "
                    "scenario."
                )

        elif isinstance(number_of_nodes, int):
            for state in grouped_nodes:
                while grouped_nodes[state] and len(allocated_nodes) < number_of_nodes:
                    allocated_nodes.append(grouped_nodes[state].pop(0))

            if len(allocated_nodes) < number_of_nodes:
                raise ValueError(
                    f"CloudAI is requesting {number_of_nodes} nodes from the group '{group_name}', but only "
                    f"{len(allocated_nodes)} nodes are available. Please review the available nodes in the system "
                    f"and ensure there are enough resources to meet the requested node count. Additionally, "
                    f"verify that the system can accommodate the number of nodes required by the test scenario."
                )

        else:
            raise ValueError(
                f"The 'number_of_nodes' argument must be either an integer specifying the number of nodes to allocate,"
                f" or 'max_avail' to allocate all available nodes. Received: '{number_of_nodes}'. "
                "Please correct the input."
            )

        for node in allocated_nodes:
            node.state = SlurmNodeState.ALLOCATED
        self.group_allocated.update(copy(node) for node in allocated_nodes)

        return allocated_nodes

    def scancel(self, job_id: int) -> None:
        """
        Terminates a specified Slurm job by sending a cancellation command.

        Args:
            job_id (int): The ID of the job to cancel.
        """
        if self.uses_slurm_api:
            self._rest_request("DELETE", "slurm", f"job/{job_id}")
            return
        self.cmd_shell.execute(f"scancel {job_id}")

    def fetch_command_output(self, command: str) -> Tuple[str, str]:
        """
        Execute a system command and return its output.

        Args:
            command (str): The command to execute.

        Returns:
            Tuple[str, str]: The stdout and stderr from the command execution.
        """
        logging.debug(f"Executing command: {command}")
        stdout, stderr = self.cmd_shell.execute(command).communicate()
        if stderr:
            logging.error(f"Error executing command '{command}': {stderr}")
        return stdout, stderr

    def convert_state_to_enum(self, state_str: str) -> SlurmNodeState:
        """
        Convert a Slurm node state string to its corresponding enum member.

        Handles both full state names and abbreviated forms. Special handling for states ending with "*", indicating a
        non-responding node. If the state cannot be matched, UNKNOWN_STATE is returned.

        Args:
            state_str (str): State string from Slurm, could be full name, abbreviated code, or with a "*" suffix.

        Returns:
            SlurmNodeState: Corresponding enum member, or UNKNOWN_STATE for unmatched states, NOT_RESPONDING for "*"
                suffix.

        Raises:
            ValueError: If state_str is not a non-empty string.
        """
        if not isinstance(state_str, str) or not state_str:
            raise ValueError("state_str must be a non-empty string")

        # Mapping of abbreviated states to enum members
        state_abbreviations = {
            "alloc": SlurmNodeState.ALLOCATED,
            "comp": SlurmNodeState.COMPLETING,
            "down": SlurmNodeState.DOWN,
            "drain": SlurmNodeState.DRAINED,
            "drng": SlurmNodeState.DRAINING,
            "fail": SlurmNodeState.FAIL,
            "failg": SlurmNodeState.FAILING,
            "futr": SlurmNodeState.FUTURE,
            "idle": SlurmNodeState.IDLE,
            "maint": SlurmNodeState.MAINTENANCE,
            "mix": SlurmNodeState.MIXED_ALLOCATION,
            "npc": SlurmNodeState.USING_NETWORK_PERFORMANCE_COUNTERS,
            "plnd": SlurmNodeState.PLANNED_STATE,
            "pow_dn": SlurmNodeState.PENDING_POWER_DOWN_STATE,
            "pow_up": SlurmNodeState.POWERING_UP_STATE,
            "resv": SlurmNodeState.RESERVED,
            "unk": SlurmNodeState.UNKNOWN_STATE,
        }

        core_state = state_str.split()[0].upper().rstrip("*+~#!%$@^-")

        if state_str.endswith("*"):
            return SlurmNodeState.NOT_RESPONDING

        try:
            return SlurmNodeState(core_state)
        except ValueError:
            abbrev = core_state.lower()
            if abbrev in state_abbreviations:
                return state_abbreviations[abbrev]
            else:
                logging.debug(f"Unknown node state: {core_state}")
                return SlurmNodeState.UNKNOWN_STATE

    def parse_nodes(self, nodes: List[str], exclude_nodes: list[str] | None = None) -> List[str]:
        """
        Parse a list of node specifications into individual node names.

        Supports explicit node names and specifications in "partition:group:num_nodes" format, and also handles ranges
        in node names. This allows for dynamic node allocation based on system state and compact node list
        specifications.

        Args:
            nodes (List[str]): A list containing node names or specifications. Specifications should follow
                "partition:group:num_nodes", where "partition" is the partition name, "group" is a group within that
                partition, and "num_nodes" is the number of nodes requested. Node ranges should be specified with
                square brackets and dashes, e.g., "node[01-03]" for "node01", "node02", "node03".
            exclude_nodes (list[str] | None): Node names (or Slurm range expressions) to exclude from group pools
                before selection. Ranges are expanded internally.

        Returns:
            List[str]: A list of node names. For specifications, it includes names of allocated nodes based on the
                specification, without duplicates. Node ranges are expanded into individual node names.

        Raises:
            ValueError: If a specification is malformed, a specified node is not found, or a node range cannot be
                parsed. This ensures users are aware of incorrect inputs.
        """
        if exclude_nodes:
            exclude_nodes = [n for spec in exclude_nodes for n in parse_node_list(spec)]

        parsed_nodes = []
        for node_spec in nodes:
            if ":" in node_spec:
                parts = node_spec.split(":")
                if len(parts) != 3:
                    raise ValueError("Format should be partition:group:num_nodes")
                partition_name, group_name, num_nodes_spec = parts
                num_nodes = int(num_nodes_spec) if num_nodes_spec != "max_avail" else num_nodes_spec
                group_nodes = self.get_available_nodes_from_group(
                    partition_name, group_name, num_nodes, exclude_nodes=exclude_nodes
                )
                parsed_nodes += [node.name for node in group_nodes]
            else:
                expanded_nodes = parse_node_list(node_spec)
                if exclude_nodes:
                    expanded_nodes = [n for n in expanded_nodes if n not in exclude_nodes]
                parsed_nodes += expanded_nodes

        # Remove duplicates while preserving order
        parsed_nodes = list(dict.fromkeys(parsed_nodes))
        return parsed_nodes

    def get_nodes_by_spec(
        self, num_nodes: int, nodes: list[str], exclude_nodes: list[str] | None = None
    ) -> Tuple[int, list[str]]:
        """
        Retrieve a list of node names based on specifications.

        When nodes is empty, returns `(num_nodes, [])`, otherwise parses the node specifications and returns the number
        of nodes and a list of node names.

        Args:
            num_nodes (int): The number of nodes, can't be `0`.
            nodes (list[str]): A list of node names specifications, slurm format or `PARTITION:GROUP:NUM_NODES`.
            exclude_nodes (list[str] | None): Node names to exclude from group pools before selection.

        Returns:
            Tuple[int, list[str]]: The number of nodes and a list of node names.

        Raises:
            ValueError: If node specifications were provided but resolved to an empty list.
        """
        num_nodes, node_list = num_nodes, []
        parsed_nodes = self.parse_nodes(nodes, exclude_nodes=exclude_nodes)
        if parsed_nodes:
            num_nodes = len(parsed_nodes)
            node_list = parsed_nodes
        elif nodes:
            reason = (
                f"after excluding nodes {exclude_nodes}"
                if exclude_nodes
                else "— no nodes are available (all may be DRAIN/DOWN)"
            )
            raise ValueError(
                f"Node specifications {nodes} resolved to an empty node list {reason}. "
                "Cannot fall back to unconstrained allocation."
            )
        return num_nodes, sorted(node_list)

    def system_installables(self) -> list[Installable]:
        return [File(Path(__file__).parent.absolute() / "slurm-metadata.sh")]

    def complete_job(self, job: SlurmJob) -> list[str]:
        if self.uses_slurm_api:
            assert isinstance(job.id, int)
            rest_job = self._rest_accounting_job(job.id)
            spec = str(rest_job.get("nodes", "")) if rest_job else ""
        else:
            out, _ = self.fetch_command_output(f"sacct -j {job.id} -p --noheader -X --format=NodeList")
            spec = out.splitlines()[0] if out.splitlines() else out
        nodelist = sorted(set(parse_node_list(spec.strip().replace("|", ""))))
        to_unlock = [node for node in self.group_allocated if node.name in nodelist]
        self.group_allocated.difference_update(to_unlock)
        return nodelist
