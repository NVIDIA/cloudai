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

from __future__ import annotations

import logging
import shlex
import stat
from pathlib import Path
from typing import Any, Optional, cast

import toml

from cloudai.models.scenario import TestRunDetails
from cloudai.systems.slurm import SlurmCommandGenStrategy

from .megatron_bridge import MegatronBridgeCmdArgs, MegatronBridgeTestDefinition


class MegatronBridgeSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """
    Slurm strategy (like `nemo_launcher`): execute the Megatron-Bridge launcher on the submit node.

    The launcher submits the actual training sbatch job; CloudAI tracks that job ID via SlurmRunner parsing.
    """

    CONTAINER_RUNTIME_ENV_VARS: frozenset[str] = frozenset(
        {
            "MELLANOX_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "NVIDIA_DRIVER_CAPABILITIES",
        }
    )

    def _container_mounts(self) -> list[str]:
        # This workload submits its own sbatch job and passes mounts via `-cm`.
        return []

    def gen_exec_command(self) -> str:
        tdef: MegatronBridgeTestDefinition = cast(MegatronBridgeTestDefinition, self.test_run.test)
        args: MegatronBridgeCmdArgs = tdef.cmd_args

        mbridge_repo_path = (
            tdef.megatron_bridge_repo.installed_path.absolute() if tdef.megatron_bridge_repo.installed_path else None
        )
        if not mbridge_repo_path:
            logging.warning(
                f"Local clone of git repo {tdef.megatron_bridge_repo} does not exist. "
                "Please ensure to run installation before running the test."
            )
            mbridge_repo_path = self.system.install_path / tdef.megatron_bridge_repo.repo_name  # dry-run compatibility

        venv_path = tdef.python_executable.venv_path
        if not venv_path:
            logging.warning(
                f"The virtual environment for git repo {tdef.python_executable.git_repo} does not exist. "
                "Please ensure to run installation before running the test."
            )
            venv_path = self.system.install_path / tdef.python_executable.venv_name  # dry-run compatibility

        launcher_py = (mbridge_repo_path / "scripts" / "performance" / "setup_experiment.py").absolute()

        pre_hook_sbatch_path: Optional[Path] = None
        base_slurm_params: str = ""
        capture_nodelist: bool = False
        if self.test_run.pre_test:
            pre_hook_sbatch_path = self._gen_pre_hook_sbatch()
            parts = self._build_launcher_parts(args, tdef, mbridge_repo_path, launcher_py, include_slurm_params=False)
            base_slurm_params = ";".join(self._collect_additional_slurm_params())
            _, node_list = self.get_cached_nodes_spec()
            capture_nodelist = not node_list
        else:
            parts = self._build_launcher_parts(args, tdef, mbridge_repo_path, launcher_py)

        launcher_python = str((venv_path / "bin" / "python").absolute())
        full_cmd = self._wrap_launcher_for_job_id_and_quiet_output(
            " ".join(parts),
            launcher_python,
            pre_hook_sbatch_path=pre_hook_sbatch_path,
            base_slurm_params=base_slurm_params,
            capture_nodelist=capture_nodelist,
        )

        self._write_command_to_file(full_cmd, self.test_run.output_path)
        return full_cmd

    def _collect_additional_slurm_params(self) -> list[str]:
        """Return the additional_slurm_params list (without dependency)."""
        params: list[str] = []
        if self.system.gpus_per_node and self.system.supports_gpu_directives:
            params.append(f"gpus-per-node={self.system.gpus_per_node}")
            params.append(f"gres=gpu:{self.system.gpus_per_node}")
        _, node_list = self.get_cached_nodes_spec()
        if node_list:
            params.append(f"nodelist={','.join(node_list)}")
        elif self.test_run.exclude_nodes:
            params.append(f"exclude={','.join(self.test_run.exclude_nodes)}")
        for source in (self.system.extra_srun_args, self.test_run.extra_srun_args):
            if source:
                params.extend(self._parse_srun_args_as_slurm_params(source))
        return params

    def _gen_pre_hook_sbatch(self) -> Path:
        """
        Generate a standalone sbatch script running per-node independent pre-hook tests.

        Each node runs its own alltoall among its local GPUs (1 srun per node in parallel),
        so the tests are truly independent — no cross-node NCCL communicator is formed.
        """
        pre_hook_output = self.test_run.output_path / "pre_hook"
        pre_hook_output.mkdir(parents=True, exist_ok=True)

        sbatch_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name=pre_hook_{self.job_name()}",
            f"#SBATCH --output={pre_hook_output.absolute() / 'stdout.txt'}",
            f"#SBATCH --error={pre_hook_output.absolute() / 'stderr.txt'}",
            f"#SBATCH --partition={self.system.default_partition}",
        ]
        if self.system.account:
            sbatch_lines.append(f"#SBATCH --account={self.system.account}")
        self._append_resource_directives(sbatch_lines, self.test_run.time_limit)
        if self.test_run.extra_srun_args:
            for param in self._parse_srun_args_as_slurm_params(self.test_run.extra_srun_args):
                key, _, val = param.partition("=")
                sbatch_lines.append(f"#SBATCH --{key}={val}" if val else f"#SBATCH --{key}")
        sbatch_lines.append("")

        success_vars = []
        for idx, tr in enumerate(self.test_run.pre_test.test_runs):
            tr.num_nodes = 1
            strategy = self._get_cmd_gen_strategy(tr)
            self._set_hook_output_path(tr, self.test_run.output_path / "pre_test")
            tr.output_path.mkdir(parents=True, exist_ok=True)

            node_out = tr.output_path.absolute()
            srun_cmd = strategy.gen_srun_command()
            # Inject per-node output paths and --nodelist so each node runs independently
            node_srun = srun_cmd.replace(
                "srun ",
                f"srun --nodelist=$_node --output={node_out}/stdout_$_node.txt --error={node_out}/stderr_$_node.txt ",
                1,
            )

            success_var = f"SUCCESS_{idx}"
            success_vars.append(success_var)

            min_busbw = getattr(tr.test.cmd_args, "min_busbw", None)
            if min_busbw is not None:
                check_cmd = (
                    f"awk '/Avg bus bandwidth/ {{ if ($NF+0 >= {min_busbw}) found=1 }}"
                    f" END {{ exit !found }}' {node_out}/stdout_$_node.txt 2>/dev/null"
                )
            else:
                check_cmd = f'grep -q "Avg bus bandwidth" {node_out}/stdout_$_node.txt 2>/dev/null'

            sbatch_lines.extend(
                [
                    f"# {tr.test.name}",
                    f"mkdir -p {node_out}",
                    "for _node in $(scontrol show hostnames $SLURM_JOB_NODELIST); do",
                    f"    {node_srun} &",
                    "done",
                    "wait",
                    f"{success_var}=1",
                    "for _node in $(scontrol show hostnames $SLURM_JOB_NODELIST); do",
                    f"    if ! {check_cmd}; then",
                    f"        {success_var}=0",
                    "    fi",
                    "done",
                    "",
                ]
            )

        combined = " && ".join([f"[ ${v} -eq 1 ]" for v in success_vars])
        sbatch_lines.extend(
            [
                f"PRE_TEST_SUCCESS=$( {combined} && echo 1 || echo 0 )",
                'if [ "$PRE_TEST_SUCCESS" -ne 1 ]; then',
                '  echo "Pre-hook tests failed. Blocking training job." >&2',
                "  exit 1",
                "fi",
            ]
        )

        sbatch_path = self.test_run.output_path / "pre_hook_sbatch_script.sh"
        sbatch_path.write_text("\n".join(sbatch_lines))
        sbatch_path.chmod(sbatch_path.stat().st_mode | stat.S_IXUSR)
        return sbatch_path

    def store_test_run(self) -> None:
        test_cmd = self.gen_exec_command()
        trd = TestRunDetails.from_test_run(self.test_run, test_cmd=test_cmd, full_cmd=test_cmd)
        with (self.test_run.output_path / self.TEST_RUN_DUMP_FILE_NAME).open("w") as f:
            toml.dump(trd.model_dump(), f)

    def _write_command_to_file(self, command: str, output_path: Path) -> None:
        log_file = output_path / "cloudai_generated_command.sh"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w") as f:
            f.write(f"{command}\n")

    def _build_custom_bash_env_exports(self) -> list[str]:
        """
        Build repeated -cb entries that export env vars inside the launched Slurm job shell.

        We quote each full `export KEY=value` command so `$SLURM_*` and commas survive
        argument parsing on the submit node and are expanded/interpreted in the job shell.
        """
        exports: list[str] = []
        for key, value in sorted(self.final_env_vars.items()):
            exports.extend(["-cb", shlex.quote(f"export {key}={value}")])
        return exports

    def _container_runtime_env_exports(self) -> list[str]:
        """
        Build ``export`` lines for container-runtime env vars.

        Variables like ``MELLANOX_VISIBLE_DEVICES`` and ``NVIDIA_VISIBLE_DEVICES``
        are consumed by the NVIDIA container toolkit / enroot at container-creation
        time to decide which devices to mount.  They must be present in the process
        environment **before** the Megatron-Bridge launcher calls ``sbatch`` so that
        Slurm inherits them into the job and ``srun`` passes them to the container
        runtime.  Exporting them in the wrapper script (which runs on the submit
        node) achieves this.  The same variables are still passed via ``-cb`` as
        well, so they are also set inside the container for any runtime readers.
        """
        lines: list[str] = []
        for key, value in sorted(self.final_env_vars.items()):
            if key in self.CONTAINER_RUNTIME_ENV_VARS:
                lines.append(f"export {key}={shlex.quote(str(value))}")
        return lines

    def _normalize_recompute_modules(self, val: Any) -> str:
        if isinstance(val, list):
            items = [str(x).strip().strip("\"'") for x in val if str(x).strip()]
            joined = ",".join(items)
            return f'"{joined}"'
        s = str(val).strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        items = [seg.strip().strip("\"'") for seg in s.split(",") if seg.strip()]
        joined = ",".join(items)
        return f'"{joined}"'

    @staticmethod
    def _parse_srun_args_as_slurm_params(srun_args: str) -> list[str]:
        """
        Convert ``--key value`` pairs from extra_srun_args into ``key=value`` for --additional_slurm_params.

        Standalone boolean flags (e.g. ``--exclusive``) are emitted as bare
        key names without a ``=value`` suffix.
        """
        params: list[str] = []
        tokens = shlex.split(srun_args)
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.startswith("--") and "=" in tok:
                key, val = tok[2:].split("=", 1)
                params.append(f"{key}={val}")
            elif tok.startswith("--") and i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                params.append(f"{tok[2:]}={tokens[i + 1]}")
                i += 1
            elif tok.startswith("--"):
                params.append(tok[2:])
            i += 1
        return params

    def _normalize_cuda_graph_scope_arg(self, val: Any) -> str:
        s = str(val).strip().strip("\"'")
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        parts = [p.strip().strip("\"'") for p in s.split(",") if p.strip()]
        return ",".join(parts)

    def _wrap_launcher_for_job_id_and_quiet_output(
        self,
        launcher_cmd: str,
        launcher_python: str,
        pre_hook_sbatch_path: Optional[Path] = None,
        base_slurm_params: str = "",
        capture_nodelist: bool = False,
    ) -> str:
        """
        Run the Megatron-Bridge launcher quietly and ensure CloudAI can parse a job ID.

        CloudAI's SlurmRunner expects stdout to include "Submitted batch job <id>".
        This writes a readable wrapper script (with section breaks) into the test output directory, then runs it.

        If pre_hook_sbatch_path is provided, the pre-hook sbatch is submitted first and its job ID is used as
        a Slurm dependency (afterok) for the main training job, so training only starts if the pre-hook passed.
        """
        output_dir = self.test_run.output_path.absolute()
        output_dir.mkdir(parents=True, exist_ok=True)

        wrapper_path = output_dir / "cloudai_megatron_bridge_submit_and_parse_jobid.sh"
        log_path = output_dir / "cloudai_megatron_bridge_launcher.log"

        container_runtime_exports = self._container_runtime_env_exports()

        pre_hook_lines: list[str] = []
        launch_line: str
        if pre_hook_sbatch_path is not None:
            nodelist_lines: list[str] = []
            if capture_nodelist:
                nodelist_lines = [
                    "# Wait for pre-hook to reach RUNNING; capture its nodelist or exit on terminal state",
                    "PRE_HOOK_NODES=''",
                    "while true; do",
                    '    _state=$(squeue -j "$PRE_HOOK_JOB_ID" -h -o "%T" 2>/dev/null || true)',
                    '    if [ "$_state" = "RUNNING" ]; then',
                    '        PRE_HOOK_NODES=$(squeue -j "$PRE_HOOK_JOB_ID" -h -o "%N" 2>/dev/null | head -1 || true)',
                    "        break",
                    '    elif [ -z "$_state" ] || [ "$_state" = "FAILED" ] || [ "$_state" = "CANCELLED" ] || [ "$_state" = "COMPLETED" ] || [ "$_state" = "TIMEOUT" ]; then',  # noqa: E501
                    "        echo \"Pre-hook job $PRE_HOOK_JOB_ID ended in state '${_state:-gone}' before reaching RUNNING.\" >&2",  # noqa: E501
                    "        exit 1",
                    "    fi",
                    "    sleep 10",
                    "done",
                    'ADDITIONAL_SLURM_PARAMS="${ADDITIONAL_SLURM_PARAMS};nodelist=${PRE_HOOK_NODES}"',
                    "",
                ]
            pre_hook_lines = [
                f'PRE_HOOK_SBATCH="{pre_hook_sbatch_path.absolute()}"',
                'PRE_HOOK_OUTPUT=$(sbatch "$PRE_HOOK_SBATCH" 2>&1)',
                'PRE_HOOK_JOB_ID=$(echo "$PRE_HOOK_OUTPUT" | grep -Eo "Submitted batch job [0-9]+" | grep -Eo "[0-9]+" | tail -n1 || true)',  # noqa: E501
                'if [ -z "$PRE_HOOK_JOB_ID" ]; then',
                '  echo "Failed to submit pre-hook job: $PRE_HOOK_OUTPUT" >&2',
                "  exit 1",
                "fi",
                'echo "Submitted pre-hook batch job $PRE_HOOK_JOB_ID"',
                f'ADDITIONAL_SLURM_PARAMS="{base_slurm_params}"',
                'ADDITIONAL_SLURM_PARAMS="${ADDITIONAL_SLURM_PARAMS};dependency=afterok:${PRE_HOOK_JOB_ID}"',
                "",
                *nodelist_lines,
            ]
            launch_line = (
                f'{launcher_cmd} --additional_slurm_params "$ADDITIONAL_SLURM_PARAMS" >>"$LOG" 2>&1 || LAUNCH_RC=$?'
            )
        else:
            launch_line = f'{launcher_cmd} >>"$LOG" 2>&1 || LAUNCH_RC=$?'

        script_lines = [
            "#!/usr/bin/env bash",
            "set -o pipefail",
            "",
            f'export NEMORUN_HOME="{output_dir}"',
            'mkdir -p "$NEMORUN_HOME"',
            f'LOG="{log_path}"',
            f'WRAPPER_STDOUT="{output_dir / "cloudai_megatron_bridge_wrapper.stdout"}"',
            f'WRAPPER_STDERR="{output_dir / "cloudai_megatron_bridge_wrapper.stderr"}"',
            # Mirror wrapper stdout/stderr to files for debugging while still emitting to the parent process.
            'exec > >(tee -a "$WRAPPER_STDOUT") 2> >(tee -a "$WRAPPER_STDERR" >&2)',
            "",
            *container_runtime_exports,
            "",
            *pre_hook_lines,
            ': >"$LOG"',
            "WANDB_INSTALL_RC=0",
            f'{shlex.quote(launcher_python)} -m pip install wandb numpy==1.26.4 >>"$LOG" 2>&1 || WANDB_INSTALL_RC=$?',
            'if [ "${WANDB_INSTALL_RC}" -ne 0 ]; then',
            '  echo "Failed to install runtime deps (wandb, numpy==1.26.4) in launcher venv (exit ${WANDB_INSTALL_RC})." >&2',  # noqa: E501
            '  tail -n 40 "$LOG" >&2 || true',
            '  exit "${WANDB_INSTALL_RC}"',
            "fi",
            "",
            "LAUNCH_RC=0",
            launch_line,
            "",
            # Parse job id from Megatron-Bridge output (multiple possible formats)
            # Patterns: "Submitted batch job 694112", "Job id: 694112", "- Job id: 694112", "Job ID: 694112"
            "",
            'JOB_ID=""',
            'JOB_ID=$(grep -Eio "(Submitted batch job[ ]+[0-9]+|Job id[: ]+[0-9]+|-[ ]*Job id[: ]+[0-9]+|Job ID[: ]+[0-9]+)" "$LOG" | tail -n1 | grep -Eo "[0-9]+" | tail -n1 || true)',  # noqa: E501
            "",
            # Emit a canonical line for CloudAI to parse
            "",
            'if [ -n "${JOB_ID}" ]; then',
            '  if [ "${LAUNCH_RC}" -ne 0 ]; then',
            '    echo "Megatron-Bridge launcher exited non-zero (${LAUNCH_RC}) after submitting job ${JOB_ID}." >&2',
            '    tail -n 40 "$LOG" >&2 || true',
            "  fi",
            '  echo "Submitted batch job ${JOB_ID}"',
            "else",
            '  echo "Failed to retrieve job ID." >&2',
            '  if [ "${LAUNCH_RC}" -ne 0 ]; then',
            '    echo "Launcher exit code: ${LAUNCH_RC}" >&2',
            "  fi",
            '  tail -n 40 "$LOG" >&2 || true',
            "  exit 1",
            "fi",
            "",
        ]

        wrapper_path.write_text("\n".join(script_lines))
        wrapper_path.chmod(wrapper_path.stat().st_mode | stat.S_IXUSR)

        return f"bash {wrapper_path}"

    def _list_or_comma_str(self, val: str | list[str] | None) -> Optional[str]:
        """Normalize list or comma-separated string; return None if `val` is empty or None."""
        if val is None:
            return None
        elif isinstance(val, str):
            return val.strip() or None
        else:
            raise RuntimeError("Unexpected sweeps list. At this point code expects scalars only")

    def _add_extra_cmd_args(self, extra_cmd_args: dict[str, str]) -> list[str]:
        """Hydra overrides: defaults merged with the user's extra_cmd_args, which take precedence."""
        overrides = {
            "logger.log_timers_to_tensorboard": "true",
            "logger.log_throughput_to_tensorboard": "true",
            "logger.log_memory_to_tensorboard": "true",
        }
        overrides.update(extra_cmd_args)
        return [shlex.quote(f"{key}={value}" if value else key) for key, value in overrides.items()]

    def _build_launcher_parts(  # noqa: C901
        self,
        args: MegatronBridgeCmdArgs,
        tdef: MegatronBridgeTestDefinition,
        repo_path: Path,
        launcher_py: Path,
        include_slurm_params: bool = True,
    ) -> list[str]:
        fields_set = args.model_fields_set
        force_fields = {
            "model_family_name",
            "model_recipe_name",
            "num_gpus",
            "gpus_per_node",
            "hf_token",
            "save_config_filepath",
        }

        container_path = ""
        if args.container_image and "container_image" in fields_set:
            installed = tdef.docker_image.installed_path

            def _installed_container_path() -> str:
                if not installed:
                    raise RuntimeError(
                        "cmd_args.container_image was provided, but CloudAI has no installed/cached image path "
                        "(docker_image.installed_path is empty). Please run `cloudai install` first, or provide "
                        "a valid local .sqsh path in cmd_args.container_image."
                    )
                return str(Path(installed).absolute())

            ci = str(args.container_image).strip()
            if ci.startswith("/") or ci.startswith("."):
                ci_path = Path(ci).expanduser()
                container_path = str(ci_path.absolute()) if ci_path.exists() else _installed_container_path()
            else:
                container_path = _installed_container_path()

        mounts: list[str] = [str(m).strip() for m in (tdef.extra_container_mounts or []) if str(m).strip()]

        # When the user sets mount_as on the Megatron-Bridge git repo, bind-mount the
        # installed clone into the container to override the image's built-in copy.
        mb_repo = tdef.megatron_bridge_repo
        if mb_repo.mount_as:
            mb_host = mb_repo.installed_path.absolute() if mb_repo.installed_path else repo_path
            mounts.append(f"{mb_host}:{mb_repo.mount_as}")

        venv_path = tdef.python_executable.venv_path or (self.system.install_path / tdef.python_executable.venv_name)
        python_bin = (venv_path / "bin" / "python").absolute()
        parts: list[str] = [
            f'NEMORUN_HOME="{self.test_run.output_path.absolute()}"',
            str(python_bin),
            str(Path(launcher_py).absolute()),
        ]

        def add(flag: str, value: Any) -> None:
            if value is None:
                return
            if isinstance(value, bool):
                parts.extend([flag, "true" if value else "false"])
            elif isinstance(value, (list, tuple)):
                if not value:
                    return
                if flag == "--dataset_paths":
                    parts.extend([flag, *[str(x) for x in value]])
                elif flag == "--profiling_ranks":
                    parts.extend([flag, ",".join(str(x) for x in value)])
                else:
                    parts.extend([flag, str(value[0]) if len(value) == 1 else ",".join(str(x) for x in value)])
            else:
                sv = str(value)
                if sv != "":
                    parts.extend([flag, sv])

        def add_field(field: str, flag: str, value: Any) -> None:
            if field not in fields_set and field not in force_fields:
                return
            add(flag, value)

        # Base launcher flags
        if self.system.account:
            add("-a", self.system.account)
        add("-p", self.system.default_partition)
        add_field("gpu_type", "-g", args.gpu_type)
        add_field("log_dir", "-l", args.log_dir)
        add("-t", self.test_run.time_limit)
        if container_path:
            add_field("container_image", "-i", container_path)
        add_field("compute_dtype", "-c", args.compute_dtype)
        add_field("task", "--task", args.task)
        add_field("hf_token", "-hf", args.hf_token)
        add_field("nemo_home", "-nh", args.nemo_home)
        add_field("wandb_key", "-wdk", args.wandb_key)
        add_field("wandb_project_name", "-wdp", args.wandb_project_name)
        add_field("wandb_entity_name", "-wde", args.wandb_entity_name)
        add_field("wandb_experiment_name", "-wdj", args.wandb_experiment_name)
        add_field("wandb_save_dir", "-wds", args.wandb_save_dir)
        add_field("max_retries", "--max_retries", args.max_retries)
        if args.dryrun and "dryrun" in fields_set:
            parts.append("-d")
        add_field("num_gpus", "-ng", args.num_gpus)
        add_field("gpus_per_node", "-gn", self.system.gpus_per_node)
        if mounts:
            add("-cm", ",".join(mounts))

        # Pass extra env variables as `-cb export KEY=value` commands to avoid Megatron-Bridge's
        # --custom_env_vars parser limitation for comma-containing values.
        if self.final_env_vars:
            parts.extend(self._build_custom_bash_env_exports())

        # Model flags (Megatron-Bridge main-branch API)
        add_field("domain", "--domain", args.domain)
        if args.use_recipes and "use_recipes" in fields_set:
            parts.append("--use_recipes")
        if "enable_vboost" in fields_set:
            add_field("enable_vboost", "-vb", bool(args.enable_vboost))
        if not args.model_family_name:
            raise RuntimeError("Missing required cmd_args.model_family_name (maps to -m/--model_family_name).")
        if not args.model_recipe_name:
            raise RuntimeError("Missing required cmd_args.model_recipe_name (maps to -mr/--model_recipe_name).")
        add_field("model_family_name", "-m", args.model_family_name)
        add_field("model_recipe_name", "-mr", args.model_recipe_name)
        add_field("hidden_size", "--hidden_size", args.hidden_size)
        add_field("num_layers", "--num_layers", args.num_layers)
        add_field(
            "pipeline_model_parallel_layout", "--pipeline_model_parallel_layout", args.pipeline_model_parallel_layout
        )
        add_field("first_k_dense_replace", "--first_k_dense_replace", args.first_k_dense_replace)
        if args.enable_nsys and "enable_nsys" in fields_set:
            parts.append("-en")
        if "use_tokendrop" in fields_set and args.use_tokendrop is not None:
            add_field("use_tokendrop", "--use_tokendrop", bool(args.use_tokendrop))
        if "use_megatron_fsdp" in fields_set and args.use_megatron_fsdp is not None:
            add_field("use_megatron_fsdp", "--use_megatron_fsdp", bool(args.use_megatron_fsdp))
        if "nccl_ub" in fields_set and args.nccl_ub is not None:
            add_field("nccl_ub", "--nccl_ub", bool(args.nccl_ub))
        add_field("cuda_graph_impl", "--cuda_graph_impl", args.cuda_graph_impl)
        if args.cuda_graph_scope and "cuda_graph_scope" in fields_set:
            add_field(
                "cuda_graph_scope", "--cuda_graph_scope", self._normalize_cuda_graph_scope_arg(args.cuda_graph_scope)
            )

        # Parallelism
        add_field("tp", "-tp", args.tp)
        add_field("pp", "-pp", args.pp)
        add_field("cp", "-cp", args.cp)
        # When vp is 1 (or [1]), pass None so -vp is not emitted in sbatch
        vp_for_launcher = args.vp
        if vp_for_launcher == 1 or (
            isinstance(vp_for_launcher, (list, tuple)) and len(vp_for_launcher) == 1 and vp_for_launcher[0] == 1
        ):
            vp_for_launcher = "None"
        add_field("vp", "-vp", vp_for_launcher)
        add_field("ep", "-ep", args.ep)
        add_field("et", "-et", args.et)

        # Batch
        add_field("mb", "-mb", args.mb)
        add_field("gb", "-gb", args.gb)
        add_field("seq_length", "-sl", args.seq_length)

        # Misc
        if "moe_a2a_overlap" in fields_set:
            add_field("moe_a2a_overlap", "--moe_a2a_overlap", bool(args.moe_a2a_overlap))
        add_field("max_steps", "-ms", args.max_steps)
        add_field("recompute_num_layers", "-rl", args.recompute_num_layers)
        add_field("activation_offload_layers", "-ol", args.activation_offload_layers)
        if args.recompute_modules and "recompute_modules" in fields_set:
            parts.extend(["--recompute_modules", self._normalize_recompute_modules(args.recompute_modules)])

        # The workload is implemented to work only with non-detached MBridge run to obtain perf metrics
        parts.extend(["--detach", "false"])

        # Optimizer
        add_field("lr", "--lr", args.lr)
        add_field("min_lr", "--min_lr", args.min_lr)
        add_field("warmup_iters", "--warmup_iters", args.warmup_iters)

        # Checkpointing
        add_field("pretrained_checkpoint", "--pretrained_checkpoint", args.pretrained_checkpoint)
        add_field("save_dir", "--save_dir", args.save_dir)
        add_field("load_dir", "--load_dir", args.load_dir)
        add_field("save_interval", "--save_interval", args.save_interval)
        add_field("most_recent_k", "--most_recent_k", args.most_recent_k)
        add_field("save_config_filepath", "--save_config_filepath", args.save_config_filepath)

        # Data / Tokenizer
        add_field("data", "--data", args.data)
        add_field("dataset_paths", "--dataset_paths", args.dataset_paths)
        add_field("dataset_root", "--dataset_root", args.dataset_root)
        add_field("index_mapping_dir", "--index_mapping_dir", args.index_mapping_dir)
        add_field("dataset_name", "--dataset_name", args.dataset_name)
        if args.packed_sequence and "packed_sequence" in fields_set:
            parts.append("--packed_sequence")
        if args.head_only and "head_only" in fields_set:
            parts.append("--head_only")
        add_field("tokenizer_type", "--tokenizer_type", args.tokenizer_type)
        add_field("tokenizer_model", "--tokenizer_model", args.tokenizer_model)
        add_field("vocab_size", "--vocab_size", args.vocab_size)

        # Profiling (performance group)
        add_field("pytorch_profiler", "-pyp", args.pytorch_profiler)
        add_field("profiling_start_step", "--profiling_start_step", args.profiling_start_step)
        add_field("profiling_stop_step", "--profiling_stop_step", args.profiling_stop_step)
        add_field("record_memory_history", "-mh", args.record_memory_history)
        if args.profiling_gpu_metrics and "profiling_gpu_metrics" in fields_set:
            parts.append("--profiling_gpu_metrics")
        add_field("profiling_ranks", "--profiling_ranks", args.profiling_ranks)
        add_field("nsys_trace", "--nsys_trace", self._list_or_comma_str(args.nsys_trace))
        add_field("nsys_extra_args", "--nsys_extra_args", self._list_or_comma_str(args.nsys_extra_args))

        if include_slurm_params:
            additional_slurm_params = self._collect_additional_slurm_params()
            if additional_slurm_params:
                parts.extend(["--additional_slurm_params", shlex.quote(";".join(additional_slurm_params))])

        # Config variant
        add_field("config_variant", "-cv", args.config_variant)
        if args.list_config_variants and "list_config_variants" in fields_set:
            parts.append("--list_config_variants")

        # Extra args (dict -> Hydra overrides): defaults first, then user values which take precedence.
        parts.extend(self._add_extra_cmd_args(tdef.extra_cmd_args))

        return parts
