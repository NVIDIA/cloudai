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
from pathlib import Path
from typing import Any, List, cast

import yaml
from pydantic import BaseModel, TypeAdapter, ValidationError

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

    def _aiperf_args_argv(self, args: dict[str, Any]) -> list[str]:
        result = []
        for key, value in args.items():
            result.append(f"--{key}")
            if value is not None:
                result.append(str(value))
        return result

    def _runtime_result_path(self, path: str) -> str:
        if Path(path).is_absolute():
            return path
        return f"{self.CONTAINER_MOUNT_OUTPUT}/{path}"

    def _split_extra_args(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        return shlex.split(str(value))

    def _aiperf_phase_has_explicit_value(self, phase: AIPerfPhase, field_name: str, *extra_aliases: str) -> bool:
        if field_name in phase.model_fields_set and getattr(phase, field_name) is not None:
            return True

        extra = phase.model_extra or {}
        return any(extra.get(alias) is not None for alias in extra_aliases)

    def _resolve_aiperf_phase(self, phase: AIPerfPhase) -> AIPerf:
        resolved = self.td.cmd_args.aiperf.model_copy(deep=True)
        resolved.name = phase.name
        single_phase = self.td.cmd_args.aiperf_phases is None or len(self.td.cmd_args.aiperf_phases) == 1

        for field_name in ("cmd", "setup_cmd", "report_name", "artifact_dir_name", "extra_args"):
            if self._aiperf_phase_has_explicit_value(phase, field_name, field_name.replace("_", "-")):
                setattr(resolved, field_name, getattr(phase, field_name))

        if not self._aiperf_phase_has_explicit_value(phase, "artifact_dir_name", "artifact-dir-name"):
            base_artifact_dir = resolved.artifact_dir_name
            resolved.artifact_dir_name = base_artifact_dir if single_phase else f"{base_artifact_dir}/{phase.name}"
        if not self._aiperf_phase_has_explicit_value(phase, "report_name", "report-name"):
            base_report_name = resolved.report_name
            resolved.report_name = base_report_name if single_phase else f"aiperf_{phase.name}_report.csv"

        resolved.args = resolved.args.model_copy(
            update=phase.args.model_dump(by_alias=True, exclude_none=True, exclude_unset=True)
        )
        return resolved

    def _render_aiperf_script(self) -> str:
        phases = self.td.cmd_args.aiperf_phases or [AIPerfPhase.model_validate({"name": "aiperf"})]
        single_phase = len(phases) == 1
        setup_cmd = self._resolve_aiperf_phase(phases[0]).setup_cmd
        lines = [
            "#!/usr/bin/env bash",
            "set -Eeuo pipefail",
            "",
            'log() { echo "[$(date +%F\\ %T) $(hostname)]: $*"; }',
            "",
            ': "${FRONTEND_URL:?FRONTEND_URL is not set}"',
            "",
        ]

        if setup_cmd:
            setup_argv = ["bash", "-lc", setup_cmd]
            lines.extend(
                [
                    f"log {shlex.quote(f'Running aiperf setup: {shlex.join(setup_argv)}')}",
                    shlex.join(setup_argv),
                    "",
                ]
            )

        write_phase_logs = not single_phase
        for idx, phase in enumerate(phases):
            resolved_phase = self._resolve_aiperf_phase(phase)
            artifact_dir = self._runtime_result_path(resolved_phase.artifact_dir_name)
            report_source = f"{artifact_dir}/profile_export_aiperf.csv"
            report_file = self._runtime_result_path(resolved_phase.report_name)
            argv = [
                *shlex.split(resolved_phase.cmd),
                "--model",
                self.td.cmd_args.dynamo.model,
                "--endpoint-type",
                "chat",
                "--streaming",
                "--artifact-dir",
                artifact_dir,
                "--no-server-metrics",
                *self._aiperf_args_argv(resolved_phase.args.model_dump(by_alias=True, exclude_none=True)),
                *self._split_extra_args(resolved_phase.extra_args),
            ]
            cmd = f'{shlex.join(argv)} --url "$FRONTEND_URL"'
            log_message = f"Running {phase.name}: {cmd}"
            lines.append(f"rm -rf {shlex.quote(artifact_dir)}")
            lines.append(f"mkdir -p {shlex.quote(artifact_dir)}")
            lines.append(f"log {shlex.quote(log_message)}")
            if write_phase_logs:
                log_file = self._runtime_result_path(f"aiperf_{phase.name}.log")
                lines.append(f"{cmd} > {shlex.quote(log_file)} 2>&1")
            else:
                lines.append(cmd)

            lines.append(f"mkdir -p {shlex.quote(str(Path(report_file).parent))}")
            if report_source != report_file:
                lines.append(f"cp {shlex.quote(report_source)} {shlex.quote(report_file)}")
            lines.append(f"log {shlex.quote(f'AIPerf report saved to {report_file}')}")

            if not single_phase and idx == len(phases) - 1:
                final_report_file = self._runtime_result_path("aiperf_report.csv")
                lines.append(f"mkdir -p {shlex.quote(str(Path(final_report_file).parent))}")
                if report_file != final_report_file:
                    lines.append(f"cp {shlex.quote(report_file)} {shlex.quote(final_report_file)}")
                lines.append(f"log {shlex.quote(f'Final AIPerf report saved to {final_report_file}')}")
            lines.append("")

        return "\n".join(lines)

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
                ],
            )
        )

        if td.cmd_args.dynamo.prefill_worker:
            args.extend(self._get_nested_toml_args(td.cmd_args.dynamo.prefill_worker, "--prefill-"))
        args.extend(self._get_nested_toml_args(td.cmd_args.dynamo.decode_worker, "--decode-"))

        args.extend(self._get_nested_toml_args(td.cmd_args.genai_perf, "--genai_perf-"))
        args.extend(self._get_nested_toml_args(td.cmd_args.aiperf, "--aiperf-"))
        if aiperf_script:
            args.append(f"--aiperf-script {aiperf_script}")
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
