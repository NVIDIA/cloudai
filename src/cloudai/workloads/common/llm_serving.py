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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rich.console import Console
from rich.table import Table
from typing_extensions import Self

from cloudai.core import METRIC_ERROR, DockerImage, HFModel, Installable, ReportGenerationStrategy
from cloudai.models.workload import CmdArgs, TestDefinition
from cloudai.systems.slurm import SlurmCommandGenStrategy

TestDefT = TypeVar("TestDefT")
ReportT = TypeVar("ReportT", bound="LLMServingBenchReport")
LLMServingArgsT = TypeVar("LLMServingArgsT", bound="LLMServingArgs")
LLMServingCmdArgsT = TypeVar("LLMServingCmdArgsT", bound="LLMServingCmdArgs")


def all_gpu_ids(tdef: LLMServingTestDefinition[LLMServingCmdArgsT], system_gpus_per_node: int | None) -> list[int]:
    cuda_devices = str(tdef.extra_env_vars.get("CUDA_VISIBLE_DEVICES", ""))
    if (tdef.cmd_args.prefill and tdef.cmd_args.prefill.gpu_ids) and tdef.cmd_args.decode.gpu_ids:
        cuda_devices = f"{tdef.cmd_args.prefill.gpu_ids},{tdef.cmd_args.decode.gpu_ids}"
    if cuda_devices:
        return [int(gpu_id) for gpu_id in cuda_devices.split(",")]
    return list(range(system_gpus_per_node or 1))


class LLMServingArgs(CmdArgs):
    """Shared serve-argument serialization for LLM serving workloads."""

    gpu_ids: str | list[str] | None = Field(
        default=None, description="Comma-separated GPU IDs. If not set, all available GPUs will be used."
    )

    @property
    def serve_args_exclude(self) -> set[str]:
        """Fields consumed internally and excluded from generic serve args."""
        return {"gpu_ids"}

    @property
    def serve_args(self) -> list[str]:
        args: list[str] = []
        for key, value in self.model_dump(exclude=self.serve_args_exclude, exclude_none=True).items():
            opt = f"--{key.replace('_', '-')}"
            if value == "":
                args.append(opt)
            else:
                args.extend([opt, str(value)])
        return args


class LLMServingCmdArgs(CmdArgs, Generic[LLMServingArgsT]):
    """Shared command-argument shape for LLM serving workloads."""

    docker_image_url: str
    model: str
    port: int = 8000
    serve_wait_seconds: int = 300
    prefill: LLMServingArgsT | None = Field(default=None)
    decode: LLMServingArgsT


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

    def can_handle_directory(self) -> bool:
        return self.parse_results() is not None

    def used_gpus_count(self) -> int:
        return len(self.all_gpu_ids(cast(TestDefT, self.test_run.test), getattr(self.system, "gpus_per_node", 1)))

    def get_metric(self, metric: str) -> float:
        if metric not in self.metrics:
            return METRIC_ERROR

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
        table.add_row(
            f"{results.completed / results.num_prompts * 100:.2f}% ({results.completed} of {results.num_prompts})",
            f"{results.mean_ttft_ms:.4f}",
            f"{results.median_ttft_ms:.4f}",
            f"{results.p99_ttft_ms:.4f}",
            f"{results.mean_tpot_ms:.4f}",
            f"{results.median_tpot_ms:.4f}",
            f"{results.p99_tpot_ms:.4f}",
        )
        console.print(table)


class LLMServingSlurmCommandGenStrategy(SlurmCommandGenStrategy, Generic[LLMServingCmdArgsT], ABC):
    """Shared Slurm helpers for LLM serving workloads."""

    @property
    @abstractmethod
    def tdef(self) -> LLMServingTestDefinition[LLMServingCmdArgsT]:
        """Typed access to the workload test definition."""

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

    @property
    def is_two_node_disaggregated(self) -> bool:
        if not self.is_disaggregated:
            return False

        num_nodes, _ = self.get_cached_nodes_spec()
        if num_nodes not in (1, 2):
            raise ValueError(f"Disaggregated {self.workload_name} supports only 1 or 2 nodes, got {num_nodes}.")
        return num_nodes == 2

    @property
    def prefill_gpu_ids(self) -> list[int]:
        if self.tdef.cmd_args.prefill and self.tdef.cmd_args.prefill.gpu_ids:
            return [int(gpu_id) for gpu_id in str(self.tdef.cmd_args.prefill.gpu_ids).split(",")]
        if self.is_two_node_disaggregated:
            return self.gpu_ids
        mid = len(self.gpu_ids) // 2
        return self.gpu_ids[:mid]

    @property
    def decode_gpu_ids(self) -> list[int]:
        if self.tdef.cmd_args.decode.gpu_ids:
            return [int(gpu_id) for gpu_id in str(self.tdef.cmd_args.decode.gpu_ids).split(",")]
        if self.is_two_node_disaggregated:
            return self.gpu_ids
        mid = len(self.gpu_ids) // 2
        return self.gpu_ids[mid:]

    def _disagg_srun_prefix(self, relative: int | None = None) -> str:
        srun_command_parts = self.gen_srun_prefix(with_num_nodes=(relative is None))
        srun_command_parts.extend(["--overlap", "--ntasks-per-node=1", "--ntasks=1"])
        if relative is not None:
            srun_command_parts.extend([f"--relative={relative}", "-N1"])
        return " ".join(srun_command_parts)

    @staticmethod
    def _with_env(command: list[str], env_vars: dict[str, str]) -> str:
        if not env_vars:
            return " ".join(command)
        env_parts = ["env", *[f'{key}="{value}"' for key, value in env_vars.items()], *command]
        return " ".join(env_parts)

    def disaggregated_role_host(self, role: str) -> str:
        if role == "prefill":
            return "${PREFILL_NODE}"
        if role == "decode":
            return "${DECODE_NODE}"
        raise ValueError(f"Unknown disaggregated role: {role}")

    def disaggregated_bench_host(self) -> str:
        return "127.0.0.1"

    def generate_disaggregated_node_setup(self) -> str:
        if not self.is_disaggregated:
            return ""
        decode_node_check = ""
        if self.is_two_node_disaggregated:
            decode_node_check = f"""\
if [ -z "${{NODES[1]}}" ]; then
    echo "Expected 2 allocated nodes for disaggregated {self.workload_name}, got: ${{NODES[*]}}"
    exit 1
fi
"""
        return f"""\
NODES=( $(scontrol show hostname $SLURM_JOB_NODELIST) )
PREFILL_NODE=${{NODES[0]}}
DECODE_NODE=${{NODES[1]:-${{PREFILL_NODE}}}}
if [ -z "$PREFILL_NODE" ]; then
    echo "Failed to resolve allocated nodes for disaggregated {self.workload_name}"
    exit 1
fi
{decode_node_check}\
echo "Node roles: prefill=$PREFILL_NODE decode=$DECODE_NODE"

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
    def generate_cleanup_function(pid_vars: list[str]) -> str:
        pid_values = " ".join(f"{pid_var}=${pid_var}" for pid_var in pid_vars)
        kill_lines = "\n".join(f'    [ -n "${pid_var}" ] && kill -9 ${pid_var} 2>/dev/null' for pid_var in pid_vars)
        return f"""\
cleanup() {{
    echo "Cleaning up PIDs: {pid_values}"
{kill_lines}
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
    def bench_log_file(self) -> str:
        """Benchmark log file name."""
        return f"{self.workload_slug}-bench.log"

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
    def get_helper_command(self, prefill_host: str, decode_host: str) -> list[str]:
        """Return the helper process command for disaggregated mode."""

    def get_proxy_router_command(self) -> list[str]:
        return self.get_helper_command(
            prefill_host=self.disaggregated_role_host("prefill"),
            decode_host=self.disaggregated_role_host("decode"),
        )

    def _gen_srun_command(self) -> str:
        serve_commands = self.get_serve_commands()
        return self._gen_llm_serving_srun_command(serve_commands)

    def _gen_llm_serving_srun_command(self, serve_commands: list[list[str]]) -> str:
        bench_cmd = " ".join(self.get_bench_command())
        if len(serve_commands) == 1:
            return self._gen_aggregated_script(serve_commands[0], bench_cmd)
        _ = self.is_two_node_disaggregated
        return self._gen_disaggregated_script(serve_commands)

    def _gen_aggregated_script(self, serve_cmd: list[str], bench_cmd: str) -> str:
        srun_prefix = " ".join(self.gen_srun_prefix())
        serve_cmd_with_env = self._with_env(serve_cmd, self.aggregated_serve_env())
        health_func = self.generate_wait_for_health_function()
        wait_block = self.generate_wait_for_health_block(
            self.workload_name, [f"http://${{NODE}}:{self.serve_port}/health"]
        )
        return f"""\
{self.generate_cleanup_function([self.serve_pid_var])}

{health_func}

echo "Starting {self.workload_name} instances..."
{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={(self.test_run.output_path / self.serve_log_file).absolute()} \\
    {serve_cmd_with_env} &
{self.serve_pid_var}=$!

{wait_block}

echo "Running benchmark..."
{srun_prefix} --overlap --ntasks-per-node=1 --ntasks=1 \\
    --output={(self.test_run.output_path / self.bench_log_file).absolute()} \\
    {bench_cmd}"""

    def _gen_disaggregated_script(self, serve_commands: list[list[str]]) -> str:
        prefill_cmd, decode_cmd = serve_commands
        health_func = self.generate_wait_for_health_function()
        prefill_cmd_with_env = self._with_env(prefill_cmd, self.disaggregated_role_env("prefill", self.prefill_gpu_ids))
        decode_cmd_with_env = self._with_env(decode_cmd, self.disaggregated_role_env("decode", self.decode_gpu_ids))
        prefill_srun_prefix = self._disagg_srun_prefix(0 if self.is_two_node_disaggregated else None)
        decode_srun_prefix = self._disagg_srun_prefix(1 if self.is_two_node_disaggregated else None)
        prefill_local_srun_prefix = self._disagg_srun_prefix(0 if self.is_two_node_disaggregated else None)
        helper_cmd = self.get_proxy_router_command()
        bench_cmd = " ".join(self.get_bench_command())
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
        preamble = self.disaggregated_script_preamble()

        return f"""\
{self.generate_cleanup_function(["PREFILL_PID", "DECODE_PID", self.proxy_router_pid_var])}

{health_func}

{preamble}{node_setup}\
echo "Starting {self.workload_name} instances..."
{prefill_srun_prefix} \\
    --output={self.test_run.output_path.absolute()}/{self.prefill_log_file} \\
    {prefill_cmd_with_env} &
PREFILL_PID=$!

{decode_srun_prefix} \\
    --output={self.test_run.output_path.absolute()}/{self.decode_log_file} \\
    {decode_cmd_with_env} &
DECODE_PID=$!

{wait_block}

echo "Starting {self.proxy_router_name}..."
{prefill_local_srun_prefix} \\
    --output={self.test_run.output_path.absolute()}/{self.proxy_router_log_file} \\
    {" ".join(helper_cmd)} &
{self.proxy_router_pid_var}=$!

echo "Running benchmark..."
{prefill_local_srun_prefix} \\
    --output={(self.test_run.output_path / self.bench_log_file).absolute()} \\
    {bench_cmd}"""
