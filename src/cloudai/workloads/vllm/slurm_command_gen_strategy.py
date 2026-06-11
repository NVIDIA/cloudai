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

import json
import shlex
from typing import Any, cast

from cloudai.workloads.common.llm_serving import LLMServingSlurmCommandGenStrategy

from .vllm import (
    VLLM_BENCH_JSON_FILE,
    VllmCmdArgs,
    VllmSemanticEvalCmdArgs,
    VllmTestDefinition,
)


class VllmSlurmCommandGenStrategy(LLMServingSlurmCommandGenStrategy[VllmCmdArgs]):
    """Command generation strategy for vLLM on Slurm systems."""

    @property
    def tdef(self) -> VllmTestDefinition:
        return cast(VllmTestDefinition, self.test_run.test)

    @property
    def workload_name(self) -> str:
        return "vLLM"

    @staticmethod
    def _to_json_str_arg(config: dict) -> str:
        return "'" + json.dumps(config, separators=(",", ":")) + "'"

    @staticmethod
    def _with_ray_backend(command: list[str], enabled: bool) -> list[str]:
        if not enabled or "--distributed-executor-backend" in command:
            return command
        return [*command, "--distributed-executor-backend", "ray"]

    def _needs_ray(self, role: str) -> bool:
        return self.role_node_count(role) > 1

    def get_serve_commands(self) -> list[list[str]]:
        tdef: VllmTestDefinition = cast(VllmTestDefinition, self.test_run.test)
        cmd_args: VllmCmdArgs = tdef.cmd_args

        base_cmd = ["vllm", "serve", cmd_args.model, "--host", self.bind_host]
        if not tdef.cmd_args.prefill:
            return [
                self._with_ray_backend(
                    [*base_cmd, *tdef.cmd_args.decode.serve_args, "--port", str(self.serve_port)],
                    self._needs_ray("serve"),
                )
            ]

        commands: list[list[str]] = []
        for port, role, kv_role, args in [
            (self.prefill_port, "prefill", "kv_producer", tdef.cmd_args.prefill),
            (self.decode_port, "decode", "kv_consumer", tdef.cmd_args.decode),
        ]:
            kv_transfer_config: dict[str, Any] = {"kv_connector": "NixlConnector", "kv_role": kv_role}
            if args.nixl_threads is not None:
                kv_transfer_config["kv_connector_extra_config"] = {"num_threads": cast(int, args.nixl_threads)}
            commands.append(
                self._with_ray_backend(
                    [
                        *base_cmd,
                        "--port",
                        str(port),
                        "--kv-transfer-config",
                        self._to_json_str_arg(kv_transfer_config),
                        *args.serve_args,
                    ],
                    self._needs_ray(role),
                )
            )

        return commands

    def aggregated_script_preamble(self) -> str:
        if not self._needs_ray("serve"):
            return ""
        return """\
export PORT_OFFSET=$((SLURM_JOB_ID % 1000))
export SERVE_RAY_PORT=$((6379 + PORT_OFFSET))
"""

    def disaggregated_script_preamble(self) -> str:
        ray_preamble = ""
        if self._needs_ray("prefill") or self._needs_ray("decode"):
            ray_preamble = """\
export PREFILL_RAY_PORT=$((6379 + PORT_OFFSET))
export DECODE_RAY_PORT=$((7379 + PORT_OFFSET))
"""
        return f"""\
export PORT_OFFSET=$((SLURM_JOB_ID % 1000))
export PREFILL_NIXL_PORT=$((5557 + PORT_OFFSET))
export DECODE_NIXL_PORT=$((5557 + PORT_OFFSET + {len(self.gpu_ids)}))

{ray_preamble}"""

    def aggregated_cleanup_pid_vars(self) -> list[str]:
        if not self._needs_ray("serve"):
            return super().aggregated_cleanup_pid_vars()
        return ["SERVE_RAY_PID", self.serve_pid_var]

    def disaggregated_cleanup_pid_vars(self) -> list[str]:
        pid_vars = super().disaggregated_cleanup_pid_vars()
        if self._needs_ray("prefill"):
            pid_vars.insert(0, "PREFILL_RAY_PID")
        if self._needs_ray("decode"):
            insert_at = 1 if self._needs_ray("prefill") else 0
            pid_vars.insert(insert_at, "DECODE_RAY_PID")
        return pid_vars

    @property
    def proxy_router_healthcheck(self) -> str:
        fields_set = self.tdef.cmd_args.model_fields_set
        if "proxy_healthcheck" not in fields_set and "healthcheck" in fields_set:
            return self.tdef.cmd_args.healthcheck
        return self.tdef.cmd_args.proxy_healthcheck

    @staticmethod
    def _compat_health_args(endpoint: str) -> str:
        endpoints = [endpoint]
        if endpoint.endswith("/health"):
            endpoints.append(endpoint[: -len("/health")] + "/healthcheck")
        elif endpoint.endswith("/healthcheck"):
            endpoints.append(endpoint[: -len("/healthcheck")] + "/health")
        return " ".join(f'"{value}"' for value in endpoints)

    def generate_wait_for_health_function(self) -> str:
        return f"""\
wait_for_health() {{
    local endpoints=("$@")
    local timeout={self.tdef.cmd_args.serve_wait_seconds}
    local interval=5
    local end_time=$(($(date +%s) + timeout))

    while [ "$(date +%s)" -lt "$end_time" ]; do
        for endpoint in "${{endpoints[@]}}"; do
            if curl -sf "$endpoint" > /dev/null 2>&1; then
                echo "Health check passed: $endpoint"
                return 0
            fi
        done
        sleep "$interval"
    done

    echo "Timeout waiting for: ${{endpoints[*]}}"
    return 1
}}"""

    @staticmethod
    def generate_wait_for_health_block(
        service_name: str,
        endpoints: list[str],
        *,
        host_setup: str = "NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)\n",
        host_display: str = "$NODE",
    ) -> str:
        waits = "\n".join(
            f"wait_for_health {VllmSlurmCommandGenStrategy._compat_health_args(endpoint)} || exit 1"
            for endpoint in endpoints
        )
        return f"""\
{host_setup}echo "Waiting for {service_name} on {host_display} to be ready..."
{waits}"""

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
        if node_count <= 1:
            return super().render_serve_launch(
                role, command_tail, pid_var, log_file, node_count, head_node_var, nodelist_var
            )

        role_prefix = role.upper()
        ray_pid_var = f"{role_prefix}_RAY_PID"
        ray_port_var = f"{role_prefix}_RAY_PORT"
        node_array_var = f"{role_prefix}_NODES"
        ray_head_log = f"{self.workload_slug}-{role}-ray-head.log"
        ray_worker_log = f"{self.workload_slug}-{role}-ray-worker-%N.log"
        serve_log = f"{self.test_run.output_path.absolute()}/{log_file}"
        head_node_expr = f"${{{head_node_var}}}"
        worker_prefix = self._role_srun_prefix("$node")
        head_prefix = self._single_role_srun_prefix(head_node_var)
        serve_cmd = self._with_custom_bash(f'env RAY_ADDRESS="{head_node_expr}:${{{ray_port_var}}}" {command_tail}')
        ray_head_command = shlex.quote(
            f"""\
ray stop --force >/dev/null 2>&1 || true
ray start --head --port="${{{ray_port_var}}}"

active_nodes=0
for (( i=0; i < {self.tdef.cmd_args.serve_wait_seconds}; i+=5 )); do
    active_nodes=$(python3 -c 'import ray; ray.init(); print(sum(node["Alive"] for node in ray.nodes()))')
    if [ "$active_nodes" -eq "{node_count}" ]; then
        echo "All Ray workers are active: $active_nodes/{node_count}"
        ray status || true
        exec {serve_cmd}
    fi
    echo "Waiting for Ray workers: $active_nodes/{node_count} active"
    sleep 5
done

echo "Waiting for Ray workers timed out: $active_nodes/{node_count} active"
exit 1"""
        )
        ray_worker_command = shlex.quote(
            f"""\
ray stop --force >/dev/null 2>&1 || true
for (( i=0; i < {self.tdef.cmd_args.serve_wait_seconds}; i+=5 )); do
    if ray start --address={head_node_expr}:${{{ray_port_var}}} --block; then
        echo "Ray worker connected to {head_node_expr}:${{{ray_port_var}}}"
        exit 0
    fi
    echo "Waiting until the Ray worker can connect to {head_node_expr}:${{{ray_port_var}}}..."
    sleep 5
done
echo "Ray worker startup timed out for {head_node_expr}:${{{ray_port_var}}}"
exit 1"""
        )

        return f"""\
(
    trap 'kill -TERM $(jobs -pr) 2>/dev/null' TERM EXIT
    for node in "${{{node_array_var}[@]:1}}"; do
        {worker_prefix} \\
            --output={self.test_run.output_path.absolute()}/{ray_worker_log} \\
            bash -lc {ray_worker_command} &
    done
    wait
) &
{ray_pid_var}=$!
{head_prefix} \\
    --output={serve_log} \\
    --error={self.test_run.output_path.absolute()}/{ray_head_log} \\
    bash -lc {ray_head_command} &
{pid_var}=$!"""

    def disaggregated_role_env(self, role: str, gpu_ids: list[int]) -> dict[str, str]:
        env = super().disaggregated_role_env(role, gpu_ids)
        env["VLLM_NIXL_SIDE_CHANNEL_HOST"] = self.disaggregated_role_host(role)
        env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "$PREFILL_NIXL_PORT" if role == "prefill" else "$DECODE_NIXL_PORT"
        return env

    def get_helper_command(self) -> list[str]:
        return [
            "python3",
            self.tdef.cmd_args.proxy_script,
            "--host",
            self.bind_host,
            "--port",
            str(self.serve_port),
            "--prefiller-hosts",
            self.disaggregated_role_host("prefill"),
            "--prefiller-ports",
            str(self.prefill_port),
            "--decoder-hosts",
            self.disaggregated_role_host("decode"),
            "--decoder-ports",
            str(self.decode_port),
        ]

    def get_bench_command(self) -> list[str]:
        bench_args = self.tdef.bench_cmd_args
        extra_args = self.tdef.bench_cmd_args.model_extra or {}
        extras = ["--" + k.replace("_", "-") + " " + str(v) for k, v in extra_args.items()]
        return [
            "vllm",
            "bench",
            "serve",
            f"--model {self.tdef.cmd_args.model}",
            f"--base-url http://{self.bench_host}:{self.serve_port}",
            f"--random-input-len {bench_args.random_input_len}",
            f"--random-output-len {bench_args.random_output_len}",
            f"--max-concurrency {bench_args.max_concurrency}",
            f"--num-prompts {bench_args.num_prompts}",
            f"--result-dir {self.test_run.output_path.absolute()}",
            f"--result-filename {VLLM_BENCH_JSON_FILE}",
            "--save-result",
            *extras,
        ]

    def get_semantic_eval_command(self) -> list[str] | None:
        eval_args: VllmSemanticEvalCmdArgs | None = self.tdef.semantic_eval_cmd_args
        if eval_args is None:
            return None

        host = self.bench_host
        http_host = host if host.startswith("http://") or host.startswith("https://") else f"http://{host}"
        cli = self._expand_semantic_eval_args(eval_args.cli, host=http_host)
        return [eval_args.entrypoint, cli] if cli else [eval_args.entrypoint]
