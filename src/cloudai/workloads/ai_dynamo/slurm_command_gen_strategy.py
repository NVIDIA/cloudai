# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import shlex
import textwrap
from pathlib import Path
from typing import Any, List, cast

import yaml
from pydantic import BaseModel, TypeAdapter, ValidationError

import cloudai.util
from cloudai.core import File, GitRepo
from cloudai.systems.slurm import SlurmCommandGenStrategy

from .ai_dynamo import (
    LMCACHE_CONFIG_BACKUP_FILE_NAME,
    LMCACHE_CONFIG_FILE_NAME,
    AIDynamoTestDefinition,
    AIPerf,
    AIPerfPhase,
)

AIPERF_SCRIPT_FILE_NAME = "aiperf.sh"


class AIDynamoSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for AI Dynamo on Slurm systems."""

    @property
    def td(self) -> AIDynamoTestDefinition:
        return cast(AIDynamoTestDefinition, self.test_run.test)

    def _container_mounts(self) -> list[str]:
        result = [f"{self.system.hf_home_path.absolute()}:{self.CONTAINER_MOUNT_HF_HOME}"]

        logging.info(f"storage_cache_dir: {self.td.cmd_args.storage_cache_dir}")
        if self.td.cmd_args.storage_cache_dir:
            result.append(f"{self.td.cmd_args.storage_cache_dir}:{self.td.cmd_args.storage_cache_dir}")

        return result

    @property
    def final_env_vars(self) -> dict[str, str | list[str]]:
        env_vars = super().final_env_vars
        if self.td.cmd_args.lmcache is not None:
            env_vars["LMCACHE_CONFIG_FILE"] = f"{self.CONTAINER_MOUNT_OUTPUT}/{LMCACHE_CONFIG_FILE_NAME}"
        return env_vars

    @final_env_vars.setter
    def final_env_vars(self, value: dict[str, str | list[str]]) -> None:
        self._final_env_vars = value

    def image_path(self) -> str | None:
        if self.td.docker_image and self.td.docker_image.installed_path:
            return str(self.td.docker_image.installed_path)
        return None

    def _gen_dcgm_srun_prefix(self, image_path: str) -> list[str]:
        srun_parts = ["srun", "--export=ALL", f"--mpi={self.mpi}", f"--container-image={image_path}"]
        mounts = self.container_mounts()
        if mounts:
            srun_parts.append(f"--container-mounts={','.join(mounts)}")
        if not self.system.container_mount_home:
            srun_parts.append("--no-container-mount-home")
        if self.system.extra_srun_args:
            srun_parts.append(self.system.extra_srun_args)
        if self.test_run.extra_srun_args:
            srun_parts.append(self.test_run.extra_srun_args)
        return srun_parts

    def _get_toml_args(self, base_model: BaseModel, prefix: str, exclude: List[str] | None = None) -> List[str]:
        args = []
        exclude = exclude or []
        git_repo_adapter = TypeAdapter(GitRepo)
        file_adapter = TypeAdapter(File)
        toml_args = base_model.model_dump(by_alias=True, exclude=set(exclude), exclude_none=True)
        for k, v in toml_args.items():
            if isinstance(v, dict):
                try:
                    repo = git_repo_adapter.validate_python(v)
                    if repo.installed_path:
                        args.extend([f'{prefix}{k} "{self.CONTAINER_MOUNT_INSTALL}/{repo.repo_name}"'])
                    continue
                except ValidationError:
                    pass
                try:
                    file_obj = file_adapter.validate_python(v)
                    if file_obj.installed_path:
                        args.extend([f'{prefix}{k} "{self.CONTAINER_MOUNT_INSTALL}/{file_obj.src.name}"'])
                    continue
                except ValidationError:
                    pass
            str_v = str(v)
            if str_v.startswith("{") and str_v.endswith("}"):
                args.append(f"{prefix}{k} '{str_v}'")
            elif any(char in str_v for char in ['"', "'", "\n"]):
                args.append(f"{prefix}{k} {shlex.quote(str_v)}")
            else:
                args.append(f'{prefix}{k} "{v}"')

        return args

    def _get_nested_toml_args(self, base_model: BaseModel, prefix: str) -> List[str]:
        result = self._get_toml_args(base_model, prefix, exclude=["args"])

        if (nested_args := getattr(base_model, "args", None)) is not None:
            result.extend(self._get_toml_args(nested_args, prefix + "args-"))

        return result

    def _prepare_lmcache_config(self):
        if self.td.cmd_args.lmcache is None:
            return

        self.test_run.output_path.mkdir(parents=True, exist_ok=True)
        config = yaml.safe_dump(self.td.cmd_args.lmcache, sort_keys=False)
        (self.test_run.output_path / LMCACHE_CONFIG_FILE_NAME).write_text(config)
        (self.test_run.output_path / LMCACHE_CONFIG_BACKUP_FILE_NAME).write_text(config)

    def _render_aiperf_args(self, args: dict[str, Any]) -> str:
        parts: list[str] = []
        for key, value in args.items():
            if value is None or value is False:
                continue
            if isinstance(value, list | dict):
                raise ValueError(
                    f"AIPerf argument {key!r} must be a scalar value. "
                    "Use a string with AIPerf CLI syntax for multi-value arguments."
                )

            parts.append(f"--{key}")
            if value is True:
                continue

            parts.append(shlex.quote(str(value)))
        return " ".join(parts)

    def _runtime_result_path(self, path: str) -> str:
        if Path(path).is_absolute():
            return path
        return f"{self.CONTAINER_MOUNT_OUTPUT}/{path}"

    def _aiperf_phase_args(self, resolved_phase: AIPerf, artifact_dir: str) -> dict[str, Any]:
        args: dict[str, Any] = {
            "model": self.td.cmd_args.dynamo.model,
            "endpoint-type": "chat",
            "streaming": True,
        }
        args.update(resolved_phase.args.model_dump(by_alias=True, exclude_none=True))
        args["artifact-dir"] = artifact_dir

        if "server-metrics" not in args and "no-server-metrics" not in args:
            args["no-server-metrics"] = True

        return args

    def _render_aiperf_phase_args(self, resolved_phase: AIPerf, artifact_dir: str) -> str:
        args = self._aiperf_phase_args(resolved_phase, artifact_dir)
        url = args.pop("url", None)
        server_metrics_auto = args.get("server-metrics") == "auto"
        if server_metrics_auto:
            args.pop("server-metrics")

        parts = []
        for key in ("model", "endpoint-type", "streaming"):
            if key in args:
                parts.append(self._render_aiperf_args({key: args.pop(key)}))
        if url is None:
            parts.append('--url "$FRONTEND_URL"')
        else:
            parts.append(self._render_aiperf_args({"url": url}))
        parts.append(self._render_aiperf_args(args))
        if server_metrics_auto:
            parts.append('--server-metrics "$AIPERF_SERVER_METRICS_URLS"')
        return " ".join(part for part in parts if part)

    def _resolve_aiperf_phase(self, phase: AIPerfPhase) -> AIPerf:
        base_data = self.td.cmd_args.aiperf.model_dump(by_alias=True, exclude_none=True)
        phase_data = phase.model_dump(by_alias=True, exclude_none=True, exclude_unset=True)
        single_phase = self.td.cmd_args.aiperf_phases is None or len(self.td.cmd_args.aiperf_phases) == 1

        if "artifact-dir-name" not in phase_data and not single_phase:
            phase_data["artifact-dir-name"] = f"{self.td.cmd_args.aiperf.artifact_dir_name}/{phase.name}"
        if "report-name" not in phase_data and not single_phase:
            phase_data["report-name"] = f"aiperf_{phase.name}_report.csv"

        return AIPerf.model_validate(cloudai.util.deep_merge(base_data, phase_data))

    def _render_aiperf_setup_blocks(self, log_message: str, setup_cmd: str | None) -> list[str]:
        if not setup_cmd:
            return []

        setup_argv = ["bash", "-lc", setup_cmd]
        return [
            textwrap.dedent(
                f"""\
                log {shlex.quote(f"{log_message}: {shlex.join(setup_argv)}")}
                {shlex.join(setup_argv)}
                """
            ).rstrip()
        ]

    def _render_between_aiperf_phases_block(
        self,
        phase_name: str,
        cmd: str | None,
    ) -> list[str]:
        if not cmd:
            return []

        cleanup_argv = ["bash", "-lc", cmd]
        return (
            textwrap.dedent(
                f"""\
            log {shlex.quote(f"Running AIPerf between-phase command after {phase_name}: {shlex.join(cleanup_argv)}")}
            {shlex.join(cleanup_argv)}
            """
            )
            .rstrip()
            .splitlines()
        )

    def _render_aiperf_script(self) -> str:
        phases = self.td.cmd_args.aiperf_phases or [AIPerfPhase.model_validate({"name": "aiperf"})]
        single_phase = len(phases) == 1
        blocks = [
            textwrap.dedent(
                f"""\
                #!/usr/bin/env bash
                set -Eeuo pipefail

                log() {{ echo "[$(date +%F\\ %T) $(hostname)]: $*"; }}

                : "${{FRONTEND_URL:?FRONTEND_URL is not set}}"
                : "${{AIPERF_MODEL:={self.td.cmd_args.dynamo.model}}}"
                : "${{AIPERF_ENDPOINT:={self.td.cmd_args.dynamo.endpoint}}}"
                : "${{AIPERF_FAILURE_MARKER:={self.CONTAINER_MOUNT_OUTPUT}/{self.td.failure_marker}}}"
                : "${{AIPERF_HEALTH_TIMEOUT:=120}}"
                """
            ).rstrip()
        ]

        blocks.extend(self._render_aiperf_setup_blocks("Running aiperf setup", self.td.cmd_args.aiperf.setup_cmd))

        write_phase_logs = not single_phase
        for idx, phase in enumerate(phases):
            resolved_phase = self._resolve_aiperf_phase(phase)
            artifact_dir = self._runtime_result_path(resolved_phase.artifact_dir_name)
            report_source = f"{artifact_dir}/profile_export_aiperf.csv"
            report_file = self._runtime_result_path(resolved_phase.report_name)
            if isinstance(resolved_phase.extra_args, list):
                raise ValueError("AIPerf extra_args must be a string with explicit CLI syntax")
            cmd_parts = [
                shlex.join(shlex.split(resolved_phase.cmd)),
                self._render_aiperf_phase_args(resolved_phase, artifact_dir),
                resolved_phase.extra_args or "",
            ]
            cmd = " ".join(part for part in cmd_parts if part)
            if write_phase_logs:
                log_file = self._runtime_result_path(f"aiperf_{phase.name}.log")
                run_cmd = f"{cmd} > {shlex.quote(log_file)} 2>&1"
            else:
                run_cmd = cmd
            log_message = f"Running {phase.name}: {cmd}"
            phase_setup = phase.setup_cmd if "setup_cmd" in phase.model_fields_set else None
            phase_lines = self._render_aiperf_setup_blocks(f"Running AIPerf phase setup for {phase.name}", phase_setup)
            phase_lines.append(
                textwrap.dedent(
                    f"""\
                    rm -rf {shlex.quote(artifact_dir)}
                    mkdir -p {shlex.quote(artifact_dir)}
                    log {shlex.quote(log_message)}
                    phase_status=0
                    set +e
                    {run_cmd}
                    phase_status=$?
                    set -e
                    if [[ "$phase_status" -ne 0 ]]; then
                      log {shlex.quote(f"AIPerf phase {phase.name} failed")}
                    """
                ).rstrip()
            )
            if not resolved_phase.continue_on_phase_failure:
                phase_lines.append('  exit "$phase_status"')
            phase_lines.extend(
                [
                    "fi",
                    textwrap.dedent(
                        f"""\
                        if [[ "$phase_status" -eq 0 ]]; then
                          mkdir -p {shlex.quote(str(Path(report_file).parent))}
                        """
                    ).rstrip(),
                ]
            )
            if report_source != report_file:
                phase_lines.append(f"  cp {shlex.quote(report_source)} {shlex.quote(report_file)}")
            phase_lines.append(f"  log {shlex.quote(f'AIPerf report saved to {report_file}')}")

            if not single_phase and idx == len(phases) - 1:
                final_report_file = self._runtime_result_path("aiperf_report.csv")
                phase_lines.append(f"  mkdir -p {shlex.quote(str(Path(final_report_file).parent))}")
                if report_file != final_report_file:
                    phase_lines.append(f"  cp {shlex.quote(report_file)} {shlex.quote(final_report_file)}")
                phase_lines.append(f"  log {shlex.quote(f'Final AIPerf report saved to {final_report_file}')}")

            if not single_phase and idx < len(phases) - 1:
                phase_lines.extend(
                    "  " + line
                    for line in self._render_between_aiperf_phases_block(
                        phase_name=phase.name,
                        cmd=resolved_phase.between_phase_cmd,
                    )
                )

            if not single_phase and idx < len(phases) - 1 and resolved_phase.health_check_between_phases:
                health_probe_cmd = (
                    '  until curl -fsS -X POST "${FRONTEND_URL}/${AIPERF_ENDPOINT}" '
                    "-H 'Content-Type: application/json' "
                    '-d "{\\"model\\":\\"${AIPERF_MODEL}\\",\\"messages\\":[{\\"role\\":\\"user\\",'
                    '\\"content\\":\\"ping\\"}],\\"stream\\":false,\\"max_tokens\\":1}" '
                    ">/dev/null; do"
                )
                phase_lines.extend(
                    [
                        "  health_deadline=$((SECONDS + AIPERF_HEALTH_TIMEOUT))",
                        '  if [[ -f "$AIPERF_FAILURE_MARKER" ]]; then',
                        "    log 'FATAL: failure marker found between AIPerf phases'",
                        "    exit 1",
                        "  fi",
                        health_probe_cmd,
                        '    if [[ -f "$AIPERF_FAILURE_MARKER" ]]; then',
                        "      log 'FATAL: failure marker found while waiting for frontend between AIPerf phases'",
                        "      exit 1",
                        "    fi",
                        "    if (( SECONDS >= health_deadline )); then",
                        "      log 'FATAL: frontend health probe failed between AIPerf phases'",
                        "      exit 1",
                        "    fi",
                        "    sleep 1",
                        "  done",
                    ]
                )
            phase_lines.append("fi")
            blocks.append("\n".join(phase_lines))

        return "\n\n".join(blocks)

    def _prepare_aiperf_script(self) -> str | None:
        if "aiperf.sh" not in self.td.cmd_args.workloads_list:
            return None

        self.test_run.output_path.mkdir(parents=True, exist_ok=True)

        script_path = self.test_run.output_path / AIPERF_SCRIPT_FILE_NAME
        script_path.write_text(self._render_aiperf_script() + "\n")
        script_path.chmod(0o755)
        return f"{self.CONTAINER_MOUNT_OUTPUT}/{AIPERF_SCRIPT_FILE_NAME}"

    def _gen_script_args(self, td: AIDynamoTestDefinition) -> List[str]:
        self._prepare_lmcache_config()
        aiperf_script = self._prepare_aiperf_script()
        if not td.repo.installed_path:
            raise ValueError("Dynamo repo is not installed")
        args = [
            "--user $USER",
            f"--install-dir {self.CONTAINER_MOUNT_INSTALL}",
            f"--results-dir {self.CONTAINER_MOUNT_OUTPUT}",
            f"--dynamo-repo {self.CONTAINER_MOUNT_INSTALL}/{td.repo.repo_name}",
            f"--hf-home {self.CONTAINER_MOUNT_HF_HOME}",
            f"--workloads {td.cmd_args.workloads}",
            f"--failure-marker {self.CONTAINER_MOUNT_OUTPUT}/{td.failure_marker}",
            f"--success-marker {self.CONTAINER_MOUNT_OUTPUT}/{td.success_marker}",
        ]

        if td.cmd_args.storage_cache_dir:
            args.append(f"--storage-cache-dir {td.cmd_args.storage_cache_dir}")
        if td.cmd_args.lmcache_controller:
            args.append(f"--lmcache-controller-cmd {shlex.quote(td.cmd_args.lmcache_controller.cmd)}")

        args.extend(
            self._get_toml_args(
                td.cmd_args.dynamo,
                "--dynamo-",
                exclude=[
                    "prefill_worker",
                    "decode_worker",
                    "dcgm_exporter",
                    "dcgm-exporter",
                ],
            )
        )
        if td.cmd_args.dynamo.dcgm_exporter.enabled:
            args.append('--dynamo-dcgm-exporter-enabled "True"')
            args.append(f'--dynamo-dcgm-exporter-port "{td.cmd_args.dynamo.dcgm_exporter.port}"')

        if td.cmd_args.dynamo.prefill_worker:
            args.extend(self._get_nested_toml_args(td.cmd_args.dynamo.prefill_worker, "--prefill-"))
        args.extend(self._get_nested_toml_args(td.cmd_args.dynamo.decode_worker, "--decode-"))

        args.extend(self._get_nested_toml_args(td.cmd_args.genai_perf, "--genai_perf-"))
        if aiperf_script:
            args.append(f'--aiperf-name "{td.cmd_args.aiperf.name}"')
            args.append(f"--aiperf-script {aiperf_script}")
        else:
            args.extend(self._get_nested_toml_args(td.cmd_args.aiperf, "--aiperf-"))
        if td.cmd_args.aiperf_accuracy is not None:
            args.extend(self._get_nested_toml_args(td.cmd_args.aiperf_accuracy, "--aiperf_accuracy-"))

        return args

    def _gen_srun_command(self) -> str:
        num_nodes, node_list = self.get_cached_nodes_spec()

        out_dir = str(self.test_run.output_path.absolute())

        srun_cmd = self.gen_srun_prefix()
        srun_cmd.extend(
            [
                f"--nodes={num_nodes}",
                *([] if not node_list else [f"--nodelist={','.join(node_list)}"]),
                f"--ntasks={num_nodes}",
                "--ntasks-per-node=1",
                f"--output={out_dir}/node-%n-stdout.txt",
                f"--error={out_dir}/node-%n-stderr.txt",
                "bash",
                f"{self.CONTAINER_MOUNT_INSTALL}/{self.td.script.src.name}",
            ]
        )
        srun_cmd.extend(self._gen_script_args(self.td))
        return " \\\n  ".join(srun_cmd) + "\n"

    def _gen_dcgm_launcher_block(self) -> list[str]:
        dcgm_image = self.td.dcgm_exporter_image
        if not dcgm_image:
            return []

        num_nodes, node_list = self.get_cached_nodes_spec()
        out_dir = self.test_run.output_path.absolute()
        port = self.td.cmd_args.dynamo.dcgm_exporter.port
        dcgm_cmd = f"DCGM_EXPORTER_LISTEN=:{port} dcgm-exporter"
        dcgm_step_name = "cloudai-dcgm-exporter"
        srun_parts = [
            *self._gen_dcgm_srun_prefix(str(dcgm_image.installed_path)),
            "--overlap",
            f"--job-name={dcgm_step_name}",
            f"-N{num_nodes}",
            *([] if not node_list else [f"--nodelist={','.join(node_list)}"]),
            f"--ntasks={num_nodes}",
            "--ntasks-per-node=1",
            f"--output={out_dir / 'dcgm-node-%n-stdout.txt'}",
            f"--error={out_dir / 'dcgm-node-%n-stderr.txt'}",
            "bash",
            "-lc",
            shlex.quote(dcgm_cmd),
        ]

        block = [
            "# Start DCGM exporter on each node.",
            'echo "Starting DCGM exporter..."',
            " ".join(srun_parts) + " &",
            "DCGM_EXPORTER_SRUN_PID=$!",
            'echo "DCGM exporter srun PID: ${DCGM_EXPORTER_SRUN_PID}"',
            "DCGM_EXPORTER_STEP_ID=",
            "for _ in {1..10}; do",
            '    DCGM_EXPORTER_STEP_ID=$(squeue --noheader --steps --job "$SLURM_JOB_ID" '
            f'--format="%i %j" | awk \'$2 == "{dcgm_step_name}" {{ print $1; exit }}\')',
            '    if [[ -n "${DCGM_EXPORTER_STEP_ID}" ]]; then break; fi',
            "    sleep 1",
            "done",
            'echo "DCGM exporter step ID: ${DCGM_EXPORTER_STEP_ID:-unknown}"',
            "function stop_dcgm_exporter()",
            "{",
            '    if [[ -n "${DCGM_EXPORTER_STEP_ID:-}" ]]; then',
            '        scancel --signal=TERM "${DCGM_EXPORTER_STEP_ID}" 2>/dev/null || true',
            "    fi",
            '    if [[ -n "${DCGM_EXPORTER_SRUN_PID:-}" ]]; then',
            '        wait "${DCGM_EXPORTER_SRUN_PID}" 2>/dev/null || true',
            "    fi",
            "}",
            "sleep 5",
            'echo "Checking DCGM exporter metrics endpoints..."',
            "DCGM_EXPORTER_STARTUP_TIMEOUT=${DCGM_EXPORTER_STARTUP_TIMEOUT:-60}",
        ]
        if node_list:
            block.extend(
                [
                    "dcgm_nodes=(" + " ".join(shlex.quote(node) for node in node_list) + ")",
                ]
            )
        else:
            block.append('mapfile -t dcgm_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")')
        endpoints_file = shlex.quote(str(out_dir / "dcgm_endpoints.txt"))
        block.extend(
            [
                f": > {endpoints_file}",
                "dcgm_failed=0",
                'for node in "${dcgm_nodes[@]}"; do',
                f'    dcgm_url="http://${{node}}:{port}/metrics"',
                f'    echo "  ${{dcgm_url}}" >> {endpoints_file}',
                "    deadline=$((SECONDS + DCGM_EXPORTER_STARTUP_TIMEOUT))",
                '    until curl -fsS --max-time 2 "${dcgm_url}" >/dev/null; do',
                "        if (( SECONDS >= deadline )); then",
                '            echo "FATAL: DCGM exporter metrics endpoint is unreachable: ${dcgm_url}" >&2',
                "            dcgm_failed=1",
                "            break",
                "        fi",
                "        sleep 2",
                "    done",
                "    if (( dcgm_failed != 0 )); then break; fi",
                '    echo "DCGM exporter reachable: ${dcgm_url}"',
                "done",
                "if (( dcgm_failed != 0 )); then",
                "    stop_dcgm_exporter",
                "    exit 1",
                "fi",
                "",
            ]
        )
        return block

    def _gen_dcgm_cleanup_command(self) -> str | None:
        if not self.td.cmd_args.dynamo.dcgm_exporter.enabled:
            return None

        return "stop_dcgm_exporter"

    def gen_exec_command(self) -> str:
        srun_command = self._gen_srun_command()
        command_list = []
        indent = ""

        if self.test_run.pre_test:
            pre_test_command = self.gen_pre_test(self.test_run.pre_test, self.test_run.output_path)
            command_list.extend([pre_test_command, "if [ $PRE_TEST_SUCCESS -eq 1 ]; then"])
            indent = "    "

        dcgm_block = self._gen_dcgm_launcher_block()
        if dcgm_block:
            command_list.extend(f"{indent}{line}" for line in dcgm_block)

        command_list.append(f"{indent}{srun_command}")

        dcgm_cleanup = self._gen_dcgm_cleanup_command()
        if dcgm_cleanup:
            command_list.append(f"{indent}# Stop DCGM exporter when test finishes")
            command_list.append(f"{indent}{dcgm_cleanup}")

        if self.test_run.post_test:
            post_test_command = self.gen_post_test(self.test_run.post_test, self.test_run.output_path)
            command_list.append(f"{indent}{post_test_command}")

        if self.test_run.pre_test:
            command_list.append("fi")

        full_command = "\n".join(command_list).strip()
        return self._write_sbatch_script(full_command)

    def _validate_worker_nodes(
        self, node_list: list[str], worker_nodes: str | None, num_nodes: int, worker_type: str
    ) -> None:
        """Validate node list for a specific worker type."""
        if not worker_nodes:
            return

        worker_node_list = worker_nodes.split(",")
        if len(worker_node_list) != num_nodes:
            raise ValueError(
                f"Number of {worker_type} nodes ({len(worker_node_list)}) does not match num_nodes ({num_nodes})"
            )
        if not all(node in node_list for node in worker_node_list):
            raise ValueError(f"Some {worker_type} nodes are not in the allocated node list")

    def _validate_node_overlap(self, prefill_nodes: str, decode_nodes: str) -> None:
        """Validate that there is no overlap between prefill and decode nodes."""
        prefill_set = set(prefill_nodes.split(","))
        decode_set = set(decode_nodes.split(","))
        if prefill_set & decode_set:
            raise ValueError("Overlap found between prefill and decode node lists")

    def get_cached_nodes_spec(self) -> tuple[int, list[str]]:
        cache_key = ":".join(
            [
                self.test_run.name,
                str(self.test_run.current_iteration),
                str(self.test_run.step),
                str(self.test_run.num_nodes),
                ",".join(self.test_run.nodes),
            ]
        )

        if cache_key in self._node_spec_cache:
            return self._node_spec_cache[cache_key]

        prefill_n, prefill_nodes = 0, ""
        if self.td.cmd_args.dynamo.prefill_worker:
            prefill_n = cast(int, self.td.cmd_args.dynamo.prefill_worker.num_nodes)
            prefill_nodes = self.td.cmd_args.dynamo.prefill_worker.nodes
        decode_n = self.td.cmd_args.dynamo.decode_worker.num_nodes
        decode_nodes = self.td.cmd_args.dynamo.decode_worker.nodes

        assert isinstance(prefill_n, int), "prefill_worker.num_nodes must be an integer"
        assert isinstance(decode_n, int), "decode_worker.num_nodes must be an integer"

        if prefill_nodes and decode_nodes:
            self.test_run.nodes = prefill_nodes.split(",") + decode_nodes.split(",") + self.test_run.nodes
            self.test_run.num_nodes = len(self.test_run.nodes)
            prefill_n = len(prefill_nodes.split(","))
            decode_n = len(decode_nodes.split(","))
        else:
            self.test_run.num_nodes = prefill_n + decode_n

        total_nodes = prefill_n + decode_n

        logging.info("Setting num_nodes from %d to %d", self.test_run.num_nodes, total_nodes)

        self.test_run.num_nodes = total_nodes

        requested_nodes, node_list = self.system.get_nodes_by_spec(self.test_run.nnodes, self.test_run.nodes)

        if prefill_nodes or decode_nodes:
            self._validate_worker_nodes(node_list, prefill_nodes, prefill_n, "prefill")
            self._validate_worker_nodes(node_list, decode_nodes, decode_n, "decode")
            if prefill_nodes and decode_nodes:
                self._validate_node_overlap(prefill_nodes, decode_nodes)

        if total_nodes > requested_nodes:
            raise ValueError(
                f"Not enough nodes requested: need {total_nodes} total nodes "
                f"({prefill_n} prefill + {decode_n} decode), "
                f"but only got {requested_nodes}"
            )

        result = (total_nodes, node_list)
        self._node_spec_cache[cache_key] = result
        return result

    def gen_dynamo_cmd(self, module: str, config: Path) -> str:
        """
        Generate the dynamo command for serving a module with a config.

        Args:
            module: The module to serve.
            config: The path to the config file.

        Returns:
            The dynamo command string.
        """
        return f"dynamo serve {module} -f {config}"
