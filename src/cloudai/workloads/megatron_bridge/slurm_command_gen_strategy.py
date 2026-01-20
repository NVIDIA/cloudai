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
import stat
from pathlib import Path
from typing import Any, cast

import toml

from cloudai.models.scenario import TestRunDetails
from cloudai.systems.slurm import SlurmCommandGenStrategy

from .megatron_bridge import MegatronBridgeCmdArgs, MegatronBridgeTestDefinition


class MegatronBridgeSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """
    Slurm strategy (like `nemo_launcher`): execute the Megatron-Bridge launcher on the submit node.

    The launcher submits the actual training sbatch job; CloudAI tracks that job ID via SlurmRunner parsing.
    """

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

        parts = self._build_launcher_parts(args, tdef, mbridge_repo_path, launcher_py)
        full_cmd = self._wrap_launcher_for_job_id_and_quiet_output(" ".join(parts))

        self._write_command_to_file(full_cmd, self.test_run.output_path)
        return full_cmd

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

    def _normalize_cuda_graph_scope_arg(self, val: Any) -> str:
        s = str(val).strip().strip("\"'")
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        parts = [p.strip().strip("\"'") for p in s.split(",") if p.strip()]
        return ",".join(parts)

    def _wrap_launcher_for_job_id_and_quiet_output(self, launcher_cmd: str) -> str:
        """
        Run the Megatron-Bridge launcher quietly and ensure CloudAI can parse a job ID.

        CloudAI's SlurmRunner expects stdout to include "Submitted batch job <id>".
        This writes a readable wrapper script (with section breaks) into the test output directory, then runs it.
        """
        output_dir = self.test_run.output_path.absolute()
        output_dir.mkdir(parents=True, exist_ok=True)

        wrapper_path = output_dir / "cloudai_megatron_bridge_submit_and_parse_jobid.sh"
        log_path = output_dir / "cloudai_megatron_bridge_launcher.log"

        script_lines = [
            "#!/usr/bin/env bash",
            "set -uo pipefail",
            "",
            f'export NEMORUN_HOME="{output_dir}"',
            'mkdir -p "$NEMORUN_HOME"',
            f'LOG="{log_path}"',
            f'WRAPPER_STDOUT="{output_dir / "cloudai_megatron_bridge_wrapper.stdout"}"',
            f'WRAPPER_STDERR="{output_dir / "cloudai_megatron_bridge_wrapper.stderr"}"',
            # Mirror wrapper stdout/stderr to files for debugging while still emitting to the parent process.
            'exec > >(tee -a "$WRAPPER_STDOUT") 2> >(tee -a "$WRAPPER_STDERR" >&2)',
            "",
            # Launch Megatron-Bridge (log stdout/stderr to file)
            "",
            ': >"$LOG"',
            "LAUNCH_RC=0",
            f'{launcher_cmd} >>"$LOG" 2>&1 || LAUNCH_RC=$?',
            "",
            # Parse job id from Megatron-Bridge output (multiple possible formats)
            "",
            'JOB_ID=""',
            'JOB_ID=$(grep -Eio "Job[[:space:]]+id[: ]+[0-9]+" "$LOG" | '
            'tail -n1 | grep -Eo "[0-9]+" | tail -n1 || true)',
            "",
            # Emit a canonical line for CloudAI to parse
            "",
            'if [ -n "${JOB_ID}" ]; then',
            '  if [ "${LAUNCH_RC}" -ne 0 ]; then',
            '    echo "Megatron-Bridge launcher exited non-zero (${LAUNCH_RC}) after submitting job ${JOB_ID}." >&2',
            '    tail -n 200 "$LOG" >&2 || true',
            "  fi",
            '  echo "Submitted batch job ${JOB_ID}"',
            "else",
            '  echo "Failed to retrieve job ID." >&2',
            '  if [ "${LAUNCH_RC}" -ne 0 ]; then',
            '    echo "Launcher exit code: ${LAUNCH_RC}" >&2',
            "  fi",
            '  tail -n 200 "$LOG" >&2 || true',
            "  exit 1",
            "fi",
            "",
        ]

        wrapper_path.write_text("\n".join(script_lines))
        wrapper_path.chmod(wrapper_path.stat().st_mode | stat.S_IXUSR)

        return f"bash {wrapper_path}"

    def _build_launcher_parts(  # noqa: C901
        self, args: MegatronBridgeCmdArgs, tdef: MegatronBridgeTestDefinition, repo_path: Path, launcher_py: Path
    ) -> list[str]:
        fields_set = args.model_fields_set
        force_fields = {
            "model_name",
            "model_size",
            "num_gpus",
            "gpus_per_node",
            "hf_token",
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

        mounts: list[str] = []
        mounts.append(f"{repo_path.absolute()}:/opt/Megatron-Bridge")
        mounts.extend(tdef.extra_container_mounts or [])

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
        add_field("time_limit", "-t", args.time_limit)
        if container_path:
            add_field("container_image", "-i", container_path)
        add_field("compute_dtype", "-c", args.compute_dtype)
        add_field("task", "--task", args.task)
        add_field("hf_token", "-hf", args.hf_token)
        add_field("nemo_home", "-nh", args.nemo_home)
        add_field("wandb_key", "-wdk", args.wandb_key)
        add_field("wandb_prj_name", "-wdp", args.wandb_prj_name)
        add_field("wandb_exp_name", "-wdj", args.wandb_exp_name)
        if args.dryrun and "dryrun" in fields_set:
            parts.append("-d")
        add_field("num_gpus", "-ng", args.num_gpus)
        add_field("gpus_per_node", "-gn", args.gpus_per_node)
        if mounts:
            add("-cm", ",".join(mounts))

        # Model flags (Megatron-Bridge r0.2.0 API)
        if "enable_vboost" in fields_set:
            add_field("enable_vboost", "-vb", bool(args.enable_vboost))
        if not args.model_name:
            raise RuntimeError("Missing required cmd_args.model_name (maps to -m/--model_name).")
        if not args.model_size:
            raise RuntimeError("Missing required cmd_args.model_size (maps to -s/--model_size).")
        add_field("model_name", "-m", args.model_name)
        add_field("model_size", "-s", args.model_size)
        if args.enable_nsys and "enable_nsys" in fields_set:
            parts.append("-en")
        add_field("domain", "--domain", args.domain)
        if "use_tokendrop" in fields_set and args.use_tokendrop is not None:
            add_field("use_tokendrop", "--use_tokendrop", bool(args.use_tokendrop))
        if "use_megatron_fsdp" in fields_set and args.use_megatron_fsdp is not None:
            add_field("use_megatron_fsdp", "--use_megatron_fsdp", bool(args.use_megatron_fsdp))
        add_field("cuda_graph_impl", "--cuda_graph_impl", args.cuda_graph_impl)
        if args.cuda_graph_scope and "cuda_graph_scope" in fields_set:
            add_field(
                "cuda_graph_scope", "--cuda_graph_scope", self._normalize_cuda_graph_scope_arg(args.cuda_graph_scope)
            )

        # Parallelism
        add_field("tp", "-tp", args.tp)
        add_field("pp", "-pp", args.pp)
        add_field("cp", "-cp", args.cp)
        add_field("vp", "-vp", args.vp)
        add_field("ep", "-ep", args.ep)
        add_field("et", "-et", args.et)

        # Batch
        add_field("mb", "-mb", args.mb)
        add_field("gb", "-gb", args.gb)

        # Misc
        if "moe_a2a_overlap" in fields_set:
            add_field("moe_a2a_overlap", "--moe_a2a_overlap", bool(args.moe_a2a_overlap))
        add_field("max_steps", "-ms", args.max_steps)
        add_field("recompute_num_layers", "-rl", args.recompute_num_layers)
        add_field("activation_offload_layers", "-ol", args.activation_offload_layers)
        if args.recompute_modules and "recompute_modules" in fields_set:
            parts.extend(["--recompute_modules", self._normalize_recompute_modules(args.recompute_modules)])
        # r0.2.0 supports `--detach` / `--no-detach` flags (no boolean value)
        if args.detach is True and "detach" in fields_set:
            parts.append("--detach")
        elif args.detach is False and "detach" in fields_set:
            parts.append("--no-detach")

        # Extra user args (dict -> string)
        if tdef.extra_cmd_args:
            parts.append(tdef.extra_args_str)

        return parts
