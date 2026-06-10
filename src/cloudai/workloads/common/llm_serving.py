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

import re
import shlex
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rich.console import Console
from rich.table import Table
from typing_extensions import Self

from cloudai.core import METRIC_ERROR, DockerImage, HFModel, Installable, MetricValue, ReportGenerationStrategy
from cloudai.models.workload import CmdArgs, TestDefinition
from cloudai.systems.slurm import SlurmCommandGenStrategy

TestDefT = TypeVar("TestDefT", bound="LLMServingTestDefinition[Any]")
ReportT = TypeVar("ReportT", bound="LLMServingBenchReport")
LLMServingArgsT = TypeVar("LLMServingArgsT", bound="LLMServingArgs")
LLMServingCmdArgsT = TypeVar("LLMServingCmdArgsT", bound="LLMServingCmdArgs")

CustomBash = str | dict[str, str]


def validate_custom_bash_patterns(custom_bash: CustomBash | None) -> CustomBash | None:
    """Validate regex keys for workload-specific custom bash injection."""
    if isinstance(custom_bash, dict):
        for pattern in custom_bash:
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid custom_bash regex '{pattern}': {e}") from e
    return custom_bash


def parse_gpu_ids(gpu_ids: str | list[str] | None) -> list[int]:
    if not gpu_ids:
        return []
    if isinstance(gpu_ids, list):
        return [int(gpu_id) for gpu_id in gpu_ids]
    return [int(gpu_id) for gpu_id in gpu_ids.split(",")]


def all_gpu_ids(tdef: LLMServingTestDefinition[LLMServingCmdArgsT], system_gpus_per_node: int | None) -> list[int]:
    cuda_devices = str(tdef.extra_env_vars.get("CUDA_VISIBLE_DEVICES", ""))
    if tdef.cmd_args.prefill:
        if tdef.cmd_args.prefill.gpu_ids and tdef.cmd_args.decode.gpu_ids:
            return parse_gpu_ids(tdef.cmd_args.prefill.gpu_ids) + parse_gpu_ids(tdef.cmd_args.decode.gpu_ids)
    elif tdef.cmd_args.decode.gpu_ids:
        return parse_gpu_ids(tdef.cmd_args.decode.gpu_ids)
    if cuda_devices:
        return parse_gpu_ids(cuda_devices)
    return list(range(system_gpus_per_node or 1))


def calculate_prefill_gpu_ids(
    tdef: LLMServingTestDefinition[LLMServingCmdArgsT],
    num_nodes: int,
    system_gpus_per_node: int | None,
) -> list[int]:
    if not tdef.cmd_args.prefill:
        return []
    if tdef.cmd_args.prefill.gpu_ids:
        return parse_gpu_ids(tdef.cmd_args.prefill.gpu_ids)

    gpu_ids = all_gpu_ids(tdef, system_gpus_per_node)
    if num_nodes > 1 or tdef.cmd_args.prefill.num_nodes is not None:
        return gpu_ids
    mid = len(gpu_ids) // 2
    return gpu_ids[:mid]


def calculate_decode_gpu_ids(
    tdef: LLMServingTestDefinition[LLMServingCmdArgsT],
    num_nodes: int,
    system_gpus_per_node: int | None,
) -> list[int]:
    if tdef.cmd_args.decode.gpu_ids:
        return parse_gpu_ids(tdef.cmd_args.decode.gpu_ids)

    gpu_ids = all_gpu_ids(tdef, system_gpus_per_node)
    if not tdef.cmd_args.prefill:
        return gpu_ids
    if num_nodes > 1 or tdef.cmd_args.decode.num_nodes is not None:
        return gpu_ids
    mid = len(gpu_ids) // 2
    return gpu_ids[mid:]


class LLMServingArgs(CmdArgs):
    """Shared serve-argument serialization for LLM serving workloads."""

    gpu_ids: str | list[str] | None = Field(
        default=None, description="Comma-separated GPU IDs. If not set, all available GPUs will be used."
    )
    num_nodes: int | list[int] | None = Field(
        default=None,
        description="Number of Slurm nodes assigned to this role in disaggregated serving mode.",
    )

    @property
    def serve_args_exclude(self) -> set[str]:
        """Fields consumed internally and excluded from generic serve args."""
        return {"gpu_ids", "num_nodes"}

    def serialize_serve_arg(self, key: str, value: Any) -> list[str]:
        """Serialize a single serve argument to CLI tokens."""
        opt = f"--{key.replace('_', '-')}"
        if value == "":
            return [opt]
        return [opt, str(value)]

    @property
    def serve_args(self) -> list[str]:
        args: list[str] = []
        for key, value in self.model_dump(exclude=self.serve_args_exclude, exclude_none=True).items():
            args.extend(self.serialize_serve_arg(key, value))
        return args


class LLMServingCmdArgs(CmdArgs, Generic[LLMServingArgsT]):
    """Shared command-argument shape for LLM serving workloads."""

    docker_image_url: str
    model: str
    port: int = Field(default=8300, ge=1, le=65535)
    host: str = Field(default="0.0.0.0", description="Host/interface for serve or router processes to bind to.")
    bench_host: str | None = Field(
        default=None,
        description="Hostname used by the benchmark client. Defaults to the allocated node hostname.",
    )
    healthcheck: str = Field(default="")
    serve_wait_seconds: int = 300
    prefill: LLMServingArgsT | None = Field(default=None)
    decode: LLMServingArgsT

    @model_validator(mode="after")
    def validate_disaggregated_port(self) -> Self:
        if self.prefill is not None and self.port > 65335:
            raise ValueError("Disaggregated mode requires port <= 65335 because prefill/decode add 100/200.")
        return self


class LLMServingTestDefinition(TestDefinition, Generic[LLMServingCmdArgsT]):
    """Shared test-definition behavior for LLM serving workloads."""

    cmd_args: LLMServingCmdArgsT
    _docker_image: DockerImage | None = None
    _hf_model: HFModel | None = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def hf_model(self) -> HFModel:
        if not self._hf_model:
            self._hf_model = HFModel(model_name=self.cmd_args.model)
        return self._hf_model

    @property
    def extra_installables(self) -> list[Installable]:
        return []

    @property
    def installables(self) -> list[Installable]:
        return [*self.git_repos, self.docker_image, self.hf_model, *self.extra_installables]

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._docker_image = None
        self._hf_model = None

    @model_validator(mode="after")
    def check_gpu_ids_setup(self) -> Self:
        if self.cmd_args.prefill:
            prefill_set = bool(self.cmd_args.prefill.gpu_ids)
            decode_set = bool(self.cmd_args.decode.gpu_ids)
            if prefill_set != decode_set:
                raise ValueError("Both prefill and decode gpu_ids must be set or both must be None.")
        return self


class LLMServingBenchReport(BaseModel, ABC):
    """Shared benchmark result shape for LLM serving workloads."""

    model_config = ConfigDict(extra="ignore")

    num_prompts: int
    completed: int
    mean_ttft_ms: float
    median_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    p99_tpot_ms: float
    max_concurrency: int

    @property
    @abstractmethod
    def throughput(self) -> float:
        """Workload-specific throughput metric."""

    @property
    def concurrency(self) -> int:
        return self.max_concurrency

    @property
    def tps_per_user(self) -> float | None:
        if self.concurrency <= 0:
            return None
        return self.throughput / self.concurrency


class LLMServingReportGenerationStrategy(ReportGenerationStrategy, Generic[TestDefT, ReportT], ABC):
    """Shared report generation strategy for LLM serving workloads."""

    metrics: ClassVar[list[str]] = [
        "default",
        "throughput",
        "tps-per-user",
        "tps-per-gpu",
        "accuracy",
    ]

    @property
    @abstractmethod
    def result_file_name(self) -> str:
        """Benchmark result file name for this workload."""

    @property
    @abstractmethod
    def report_title(self) -> str:
        """User-facing report title."""

    @abstractmethod
    def parse_output(self, path: Path) -> ReportT | None:
        """Parse benchmark output into a report model."""

    @abstractmethod
    def all_gpu_ids(self, tdef: TestDefT, gpus_per_node: int | None) -> list[int]:
        """Return GPU ids used by this workload."""

    def parse_results(self) -> ReportT | None:
        return self.parse_output(self.test_run.output_path / self.result_file_name)

    def parse_semantic_accuracy(self) -> float | None:
        """Parse semantic validation accuracy, if supported by the workload."""
        return None

    def can_handle_directory(self) -> bool:
        return self.parse_results() is not None

    def used_gpus_count(self) -> int:
        tdef = cast(TestDefT, self.test_run.test)
        gpu_ids = self.all_gpu_ids(tdef, getattr(self.system, "gpus_per_node", 1))
        num_nodes = self.test_run.nnodes
        if tdef.cmd_args.prefill is None or num_nodes <= 1:
            return len(gpu_ids)

        prefill_gpu_ids = tdef.cmd_args.prefill.gpu_ids
        decode_gpu_ids = tdef.cmd_args.decode.gpu_ids
        if prefill_gpu_ids and decode_gpu_ids:
            return len(parse_gpu_ids(prefill_gpu_ids)) + len(parse_gpu_ids(decode_gpu_ids))

        return len(gpu_ids) * num_nodes

    def get_metric(self, metric: str) -> MetricValue:
        if metric not in self.metrics:
            return METRIC_ERROR

        if metric == "accuracy":
            tdef = cast(TestDefT, self.test_run.test)
            if getattr(tdef, "semantic_eval_cmd_args", None) is None:
                return METRIC_ERROR
            accuracy = self.parse_semantic_accuracy()
            return accuracy if accuracy is not None else METRIC_ERROR

        results = self.parse_results()
        if results is None:
            return METRIC_ERROR

        if metric == "tps-per-user":
            return results.tps_per_user if results.tps_per_user is not None else METRIC_ERROR
        if metric == "tps-per-gpu":
            return results.throughput / self.used_gpus_count()

        return results.throughput

    def generate_report(self) -> None:
        results = self.parse_results()
        if results is None:
            return

        console = Console()
        table = Table(title=f"{self.report_title} ({self.test_run.output_path})", title_justify="left")
        table.add_column("Successful prompts", justify="right")
        table.add_column("TTFT Mean, ms", justify="right")
        table.add_column("TTFT Median, ms", justify="right")
        table.add_column("TTFT P99, ms", justify="right")
        table.add_column("TPOT Mean, ms", justify="right")
        table.add_column("TPOT Median, ms", justify="right")
        table.add_column("TPOT P99, ms", justify="right")

        row = [
            f"{results.completed / results.num_prompts * 100:.2f}% ({results.completed} of {results.num_prompts})",
            f"{results.mean_ttft_ms:.4f}",
            f"{results.median_ttft_ms:.4f}",
            f"{results.p99_ttft_ms:.4f}",
            f"{results.mean_tpot_ms:.4f}",
            f"{results.median_tpot_ms:.4f}",
            f"{results.p99_tpot_ms:.4f}",
        ]

        accuracy = self.get_metric("accuracy")
        if accuracy != METRIC_ERROR:
            table.add_column("Accuracy", justify="right")
            row.append(f"{accuracy:.4f}")

        table.add_row(*row)
        console.print(table)


class LLMServingSlurmCommandGenStrategy(SlurmCommandGenStrategy, Generic[LLMServingCmdArgsT], ABC):
    """Shared Slurm helpers for LLM serving workloads."""

    @property
    @abstractmethod
    def tdef(self) -> LLMServingTestDefinition[LLMServingCmdArgsT]:
        """Typed access to the workload test definition."""

    @property
    def mpi(self) -> str:
        return "none"

    @property
    @abstractmethod
    def workload_name(self) -> str:
        """User-facing workload name for diagnostics."""

    @property
    def workload_slug(self) -> str:
        """Filesystem-friendly workload identifier used for generated artifact names."""
        return self.workload_name.lower().replace(" ", "-")

    def _container_mounts(self) -> list[str]:
        return [f"{self.system.hf_home_path.absolute()}:/root/.cache/huggingface"]

    def image_path(self) -> str | None:
        return str(self.tdef.docker_image.installed_path)

    @property
    def gpu_ids(self) -> list[int]:
        return all_gpu_ids(self.tdef, self.system.gpus_per_node)

    @property
    def is_disaggregated(self) -> bool:
        return self.tdef.cmd_args.prefill is not None

    @staticmethod
    def _role_num_nodes(value: int | list[int] | None, role: str) -> int | None:
        if isinstance(value, list):
            raise ValueError(f"{role}.num_nodes must be a single integer for command generation.")
        return value

    @property
    def aggregated_node_count(self) -> int:
        num_nodes, _ = self.get_cached_nodes_spec()
        return num_nodes

    def disaggregated_role_node_counts(self) -> tuple[int, int]:
        if not self.is_disaggregated or self.tdef.cmd_args.prefill is None:
            return (0, 0)

        num_nodes, _ = self.get_cached_nodes_spec()
        prefill_nodes = self._role_num_nodes(self.tdef.cmd_args.prefill.num_nodes, "prefill")
        decode_nodes = self._role_num_nodes(self.tdef.cmd_args.decode.num_nodes, "decode")

        if prefill_nodes is None and decode_nodes is None:
            if num_nodes in (1, 2):
                return (1, 1)
            raise ValueError(
                f"Disaggregated {self.workload_name} over more than 2 nodes requires both "
                "prefill.num_nodes and decode.num_nodes."
            )
        if prefill_nodes is None or decode_nodes is None:
            raise ValueError("Both prefill.num_nodes and decode.num_nodes must be set or both must be omitted.")
        if prefill_nodes <= 0 or decode_nodes <= 0:
            raise ValueError("prefill.num_nodes and decode.num_nodes must be positive integers.")
        if prefill_nodes + decode_nodes != num_nodes:
            raise ValueError(
                f"prefill.num_nodes + decode.num_nodes must equal allocated nodes ({num_nodes}), "
                f"got {prefill_nodes + decode_nodes}."
            )
        return (prefill_nodes, decode_nodes)

    def role_node_count(self, role: str) -> int:
        if role == "serve":
            return self.aggregated_node_count
        prefill_nodes, decode_nodes = self.disaggregated_role_node_counts()
        if role == "prefill":
            return prefill_nodes
        if role == "decode":
            return decode_nodes
        raise ValueError(f"Unknown serving role: {role}")

    @property
    def prefill_gpu_ids(self) -> list[int]:
        return calculate_prefill_gpu_ids(self.tdef, self.test_run.nnodes, self.system.gpus_per_node)

    @property
    def decode_gpu_ids(self) -> list[int]:
        return calculate_decode_gpu_ids(self.tdef, self.test_run.nnodes, self.system.gpus_per_node)

    def _role_srun_prefix(self, nodelist_expr: str, node_count: int = 1, task_count: int = 1) -> str:
        srun_command_parts = self.gen_srun_prefix(with_num_nodes=False)
        srun_command_parts.extend(
            [
                "--overlap",
                f'--nodelist="{nodelist_expr}"',
                f"--nodes={node_count}",
                f"--ntasks={task_count}",
                "--ntasks-per-node=1",
            ]
        )
        return " ".join(srun_command_parts)

    def _single_role_srun_prefix(self, node_var: str) -> str:
        return self._role_srun_prefix(f"${{{node_var}}}")

    @staticmethod
    def _with_env(command: list[str], env_vars: dict[str, str]) -> str:
        if not env_vars:
            return " ".join(command)
        env_parts = ["env", *[f'{key}="{value}"' for key, value in env_vars.items()], *command]
        return " ".join(env_parts)

    def _custom_bash_for_command(self, command_tail: str) -> str | None:
        custom_bash = getattr(self.tdef, "custom_bash", None)
        if isinstance(custom_bash, str):
            return custom_bash or None
        if isinstance(custom_bash, dict):
            for pattern, bash in custom_bash.items():
                if re.search(pattern, command_tail):
                    return bash or None
        return None

    def _with_custom_bash(self, command_tail: str) -> str:
        custom_bash = self._custom_bash_for_command(command_tail)
        if not custom_bash:
            return command_tail

        return "bash -c " + shlex.quote(f"{custom_bash}; exec {command_tail}")

    def disaggregated_role_host(self, role: str) -> str:
        if role == "prefill":
            return "${PREFILL_NODE}"
        if role == "decode":
            return "${DECODE_NODE}"
        raise ValueError(f"Unknown disaggregated role: {role}")

    @property
    def bind_host(self) -> str:
        return self.tdef.cmd_args.host

    @property
    def bench_host(self) -> str:
        configured_host = self.tdef.cmd_args.bench_host
        if configured_host:
            return configured_host
        if self.is_disaggregated:
            return "${PREFILL_NODE}"
        return "${NODE}"

    def generate_aggregated_node_setup(self, node_count: int) -> str:
        if node_count <= 1:
            return ""
        return f"""\
NODES=( $(scontrol show hostname $SLURM_JOB_NODELIST) )
SERVE_NODES=( "${{NODES[@]:0:{node_count}}}" )
if [ "${{#SERVE_NODES[@]}}" -ne {node_count} ]; then
    echo "Expected {node_count} allocated nodes for {self.workload_name}, got: ${{NODES[*]}}"
    exit 1
fi
export SERVE_NODE=${{SERVE_NODES[0]}}
export NODE=$SERVE_NODE
SERVE_NODELIST=$(IFS=,; echo "${{SERVE_NODES[*]}}")
echo "Node roles: serve=${{SERVE_NODES[*]}}"

"""

    def generate_disaggregated_node_setup(self) -> str:
        if not self.is_disaggregated:
            return ""
        allocated_nodes, _ = self.get_cached_nodes_spec()
        prefill_nodes, decode_nodes = self.disaggregated_role_node_counts()
        decode_start = 0 if allocated_nodes == 1 and prefill_nodes == 1 and decode_nodes == 1 else prefill_nodes
        role_error = (
            f"Expected {prefill_nodes} prefill and {decode_nodes} decode nodes for disaggregated {self.workload_name}"
        )
        return f"""\
NODES=( $(scontrol show hostname $SLURM_JOB_NODELIST) )
PREFILL_NODES=( "${{NODES[@]:0:{prefill_nodes}}}" )
DECODE_NODES=( "${{NODES[@]:{decode_start}:{decode_nodes}}}" )
if [ "${{#PREFILL_NODES[@]}}" -ne {prefill_nodes} ] || [ "${{#DECODE_NODES[@]}}" -ne {decode_nodes} ]; then
    echo "{role_error}, got: ${{NODES[*]}}"
    exit 1
fi
export PREFILL_NODE=${{PREFILL_NODES[0]}}
export DECODE_NODE=${{DECODE_NODES[0]}}
PREFILL_NODELIST=$(IFS=,; echo "${{PREFILL_NODES[*]}}")
DECODE_NODELIST=$(IFS=,; echo "${{DECODE_NODES[*]}}")
if [ -z "$PREFILL_NODE" ] || [ -z "$DECODE_NODE" ]; then
    echo "Failed to resolve allocated nodes for disaggregated {self.workload_name}"
    exit 1
fi
echo "Node roles: prefill=${{PREFILL_NODES[*]}} decode=${{DECODE_NODES[*]}}"

"""

    def generate_wait_for_health_function(self) -> str:
        timeout = self.tdef.cmd_args.serve_wait_seconds
        return f"""\
wait_for_health() {{
    local endpoint="$1"
    local timeout={timeout}
    local interval=5
    local end_time=$(($(date +%s) + timeout))

    while [ "$(date +%s)" -lt "$end_time" ]; do
        if curl -sf "$endpoint" > /dev/null 2>&1; then
            echo "Health check passed: $endpoint"
            return 0
        fi
        sleep "$interval"
    done

    echo "Timeout waiting for: $endpoint"
    return 1
}}"""

    @staticmethod
    def generate_cleanup_function(pid_vars: list[str], timeout: int = 15) -> str:
        if len(pid_vars) == 1:
            pid_var = pid_vars[0]
            return f"""\
cleanup() {{
    echo "Cleaning up PIDs: {pid_var}=${pid_var}"
    kill -TERM "${pid_var}" 2>/dev/null
    i=0
    while kill -0 "${pid_var}" 2>/dev/null; do
        [ "$i" -ge {timeout} ] && echo "PID did not exit in time" && return 1
        sleep 1
        i=$((i+1))
    done
}}
trap cleanup EXIT"""

        pid_values = " ".join(f"{pid_var}=${pid_var}" for pid_var in pid_vars)
        pid_array = " ".join(f'"${p}"' for p in pid_vars)
        return f"""\
cleanup() {{
    echo "Cleaning up PIDs: {pid_values}"

    for pid in {pid_array}; do
        [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null
    done

    for pid in {pid_array}; do
        [ -z "$pid" ] && continue
        i=0
        while kill -0 "$pid" 2>/dev/null; do
            [ "$i" -ge {timeout} ] && echo "PID $pid did not exit in time" && return 1
            sleep 1
            i=$((i+1))
        done
    done
}}
trap cleanup EXIT"""

    @staticmethod
    def generate_wait_for_health_block(
        service_name: str,
        endpoints: list[str],
        *,
        host_setup: str = "NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)\n",
        host_display: str = "$NODE",
    ) -> str:
        waits = "\n".join(f'wait_for_health "{endpoint}" || exit 1' for endpoint in endpoints)
        return f"""\
{host_setup}echo "Waiting for {service_name} on {host_display} to be ready..."
{waits}"""

    @property
    def prefill_port(self) -> int:
        """Prefill service port in disaggregated mode."""
        return self.tdef.cmd_args.port + 100

    @property
    def decode_port(self) -> int:
        """Decode service port in disaggregated mode."""
        return self.tdef.cmd_args.port + 200

    @property
    def prefill_log_file(self) -> str:
        """Prefill log file name in disaggregated mode."""
        return f"{self.workload_slug}-prefill.log"

    @property
    def decode_log_file(self) -> str:
        """Decode log file name in disaggregated mode."""
        return f"{self.workload_slug}-decode.log"

    @property
    def proxy_router_name(self) -> str:
        return "router"

    @property
    def proxy_router_pid_var(self) -> str:
        """Shell variable holding helper PID."""
        return "HELPER_PID"

    @property
    def proxy_router_log_file(self) -> str:
        """Helper process log file name."""
        return f"{self.workload_slug}-{self.proxy_router_name}.log"

    @property
    def proxy_router_healthcheck(self) -> str:
        """Healthcheck endpoint for the helper/proxy process in disaggregated mode."""
        return self.tdef.cmd_args.healthcheck

    @property
    def bench_log_file(self) -> str:
        """Benchmark log file name."""
        return f"{self.workload_slug}-bench.log"

    @property
    def semantic_eval_log_file(self) -> str:
        """Semantic validation log file name."""
        return f"{self.workload_slug}-semantic-eval.log"

    @property
    def serve_pid_var(self) -> str:
        """Shell variable holding the aggregated serve PID."""
        return "SERVE_PID"

    @property
    def serve_log_file(self) -> str:
        """Serve log file name in aggregated mode."""
        return f"{self.workload_slug}-serve.log"

    @property
    def serve_port(self) -> int:
        """Serve port in aggregated mode."""
        return self.tdef.cmd_args.port

    def disaggregated_script_preamble(self) -> str:
        return ""

    def aggregated_script_preamble(self) -> str:
        return ""

    def aggregated_cleanup_pid_vars(self) -> list[str]:
        return [self.serve_pid_var]

    def disaggregated_cleanup_pid_vars(self) -> list[str]:
        return ["PREFILL_PID", "DECODE_PID", self.proxy_router_pid_var]

    def aggregated_serve_env(self) -> dict[str, str]:
        return {}

    def disaggregated_role_env(self, role: str, gpu_ids: list[int]) -> dict[str, str]:
        return {"CUDA_VISIBLE_DEVICES": ",".join(str(gpu_id) for gpu_id in gpu_ids)}

    @abstractmethod
    def get_serve_commands(self) -> list[list[str]]:
        """Return workload serve commands."""

    @abstractmethod
    def get_bench_command(self) -> list[str]:
        """Return workload benchmark command."""

    @abstractmethod
    def get_helper_command(self) -> list[str]:
        """Return the helper process command for disaggregated mode."""

    def get_semantic_eval_command(self) -> list[str] | None:
        """Return the optional semantic validation command."""
        return None

    def render_serve_launch(
        self,
        role: str,
        command_tail: str,
        pid_var: str,
        log_file: str,
        node_count: int,
        head_node_var: str,
        nodelist_var: str,
    ) -> str:
        del role, node_count, nodelist_var
        return f"""\
{self._single_role_srun_prefix(head_node_var)} \\
    --output={self.test_run.output_path.absolute()}/{log_file} \\
    {self._with_custom_bash(command_tail)} &
{pid_var}=$!"""

    def _expand_semantic_eval_args(self, args: str, *, host: str) -> str:
        replacements = {
            "{model}": self.tdef.cmd_args.model,
            "{host}": host,
            "{port}": str(self.serve_port),
            "{url}": f"{host}:{self.serve_port}",
            "{output_path}": str(self.test_run.output_path.absolute()),
            "{result_dir}": str(self.test_run.output_path.absolute()),
        }
        for placeholder, value in replacements.items():
            args = args.replace(placeholder, value)
        return args

    def _gen_semantic_eval_block(self, srun_prefix: str) -> str:
        semantic_cmd = self.get_semantic_eval_command()
        if not semantic_cmd:
            return ""
        semantic_cmd_full = self._with_custom_bash(" ".join(semantic_cmd))

        return f"""\

echo "Running semantic validation..."
{srun_prefix} \\
    --output={(self.test_run.output_path / self.semantic_eval_log_file).absolute()} \\
    {semantic_cmd_full}"""

    def _gen_srun_command(self) -> str:
        serve_commands = self.get_serve_commands()
        srun_command = self._gen_llm_serving_srun_command(serve_commands)
        srun_command += "\n\ncleanup\n"
        return srun_command

    def _gen_llm_serving_srun_command(self, serve_commands: list[list[str]]) -> str:
        bench_cmd = " ".join(self.get_bench_command())
        if len(serve_commands) == 1:
            return self._gen_aggregated_script(serve_commands[0], bench_cmd)
        return self._gen_disaggregated_script(serve_commands, bench_cmd)

    def _gen_aggregated_script(self, serve_cmd: list[str], bench_cmd: str) -> str:
        serve_node_count = self.role_node_count("serve")
        legacy_single_node = serve_node_count == 1
        srun_prefix = (
            " ".join(self.gen_srun_prefix()) if legacy_single_node else self._single_role_srun_prefix("SERVE_NODE")
        )
        host_setup = (
            "" if not legacy_single_node else "NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)\n"
        )
        serve_cmd_with_env = self._with_env(serve_cmd, self.aggregated_serve_env())
        health_func = self.generate_wait_for_health_function()
        wait_block = self.generate_wait_for_health_block(
            self.workload_name,
            [f"http://${{NODE}}:{self.serve_port}{self.tdef.cmd_args.healthcheck}"],
            host_setup=host_setup,
        )
        node_setup = self.generate_aggregated_node_setup(serve_node_count)
        preamble = self.aggregated_script_preamble()
        if legacy_single_node:
            serve_launch = f"""\
{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={(self.test_run.output_path / self.serve_log_file).absolute()} \\
    {self._with_custom_bash(serve_cmd_with_env)} &
{self.serve_pid_var}=$!"""
        else:
            serve_launch = self.render_serve_launch(
                "serve",
                serve_cmd_with_env,
                self.serve_pid_var,
                self.serve_log_file,
                serve_node_count,
                "SERVE_NODE",
                "SERVE_NODELIST",
            )
        semantic_prefix = (
            f"{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1"
            if legacy_single_node
            else self._single_role_srun_prefix("SERVE_NODE")
        )
        bench_prefix = semantic_prefix
        return f"""\
{self.generate_cleanup_function(self.aggregated_cleanup_pid_vars())}

{health_func}

{preamble}{node_setup}\
echo "Starting {self.workload_name} instances..."
{serve_launch}

{wait_block}

echo "Running benchmark..."
{bench_prefix} \\
    --output={(self.test_run.output_path / self.bench_log_file).absolute()} \\
    {self._with_custom_bash(bench_cmd)}

{self._gen_semantic_eval_block(semantic_prefix)}""".strip()

    def _gen_disaggregated_script(self, serve_commands: list[list[str]], bench_cmd: str) -> str:
        prefill_cmd, decode_cmd = serve_commands
        health_func = self.generate_wait_for_health_function()
        prefill_cmd_with_env = self._with_env(prefill_cmd, self.disaggregated_role_env("prefill", self.prefill_gpu_ids))
        decode_cmd_with_env = self._with_env(decode_cmd, self.disaggregated_role_env("decode", self.decode_gpu_ids))
        prefill_nodes, decode_nodes = self.disaggregated_role_node_counts()
        prefill_srun_prefix = self._single_role_srun_prefix("PREFILL_NODE")
        helper_cmd = self.get_helper_command()
        node_setup = self.generate_disaggregated_node_setup()
        wait_block = self.generate_wait_for_health_block(
            self.workload_name,
            [
                f"http://{self.disaggregated_role_host('prefill')}:{self.prefill_port}/health",
                f"http://{self.disaggregated_role_host('decode')}:{self.decode_port}/health",
            ],
            host_setup="",
            host_display="$PREFILL_NODE and $DECODE_NODE",
        )
        wait_block_helper = self.generate_wait_for_health_block(
            self.workload_name,
            [f"http://{self.disaggregated_role_host('prefill')}:{self.serve_port}{self.proxy_router_healthcheck}"],
            host_setup="",
            host_display="$PREFILL_NODE server",
        )
        preamble = self.disaggregated_script_preamble()
        prefill_launch = self.render_serve_launch(
            "prefill",
            prefill_cmd_with_env,
            "PREFILL_PID",
            self.prefill_log_file,
            prefill_nodes,
            "PREFILL_NODE",
            "PREFILL_NODELIST",
        )
        decode_launch = self.render_serve_launch(
            "decode",
            decode_cmd_with_env,
            "DECODE_PID",
            self.decode_log_file,
            decode_nodes,
            "DECODE_NODE",
            "DECODE_NODELIST",
        )

        return f"""\
{self.generate_cleanup_function(self.disaggregated_cleanup_pid_vars())}

{health_func}

{preamble}{node_setup}\
echo "Starting {self.workload_name} instances..."
{prefill_launch}

{decode_launch}

{wait_block}

echo "Starting {self.proxy_router_name}..."
{prefill_srun_prefix} \\
    --output={self.test_run.output_path.absolute()}/{self.proxy_router_log_file} \\
    {self._with_custom_bash(" ".join(helper_cmd))} &
{self.proxy_router_pid_var}=$!

{wait_block_helper}

echo "Running benchmark..."
{prefill_srun_prefix} \\
    --output={(self.test_run.output_path / self.bench_log_file).absolute()} \\
    {self._with_custom_bash(bench_cmd)}

{self._gen_semantic_eval_block(prefill_srun_prefix)}""".strip()
