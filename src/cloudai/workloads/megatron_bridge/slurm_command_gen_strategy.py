# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
        full_cmd = " ".join(parts)

        self._last_exec_cmd = full_cmd
        return full_cmd

    def store_test_run(self) -> None:
        test_cmd = getattr(self, "_last_exec_cmd", None) or ""
        trd = TestRunDetails.from_test_run(self.test_run, test_cmd=test_cmd, full_cmd=test_cmd)
        with (self.test_run.output_path / self.TEST_RUN_DUMP_FILE_NAME).open("w") as f:
            toml.dump(trd.model_dump(), f)
        if test_cmd:
            self._write_command_to_file(test_cmd, self.test_run.output_path)

    def _write_command_to_file(self, command: str, output_path: Path) -> None:
        log_file = output_path / "generated_command.sh"
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
        """
        Normalize `--cuda_graph_scope` to Megatron-Bridge's expected format.

        Megatron-Bridge expects either:
        - a string scope (single scope) -> it will be wrapped to a list internally
        - or a list of scopes

        Our configs often use a single TOML string like "[attn]" to represent a list.
        For a single item, emit "attn" (no brackets). For multi-item, keep the bracketed form.
        """
        s = str(val).strip().strip("\"'")
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            parts = [p.strip().strip("\"'") for p in inner.split(",") if p.strip()]
            if len(parts) == 1:
                return parts[0]
            return f'"[{", ".join(parts)}]"'
        return s

    def _build_launcher_parts(
        self, args: MegatronBridgeCmdArgs, tdef: MegatronBridgeTestDefinition, repo_path: Path, launcher_py: Path
    ) -> list[str]:  # noqa: C901
        if not args.hf_token:
            raise RuntimeError("HuggingFace token is required. Please set a literal 'hf_token' in your test TOML.")

        container_path = str(tdef.docker_image.installed_path) if args.container_image else ""

        mounts: list[str] = []
        # Always mount the installed Megatron-Bridge repo into the container at /opt/Megatron-Bridge
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

        # Base launcher flags
        if self.system.account:
            add("-a", self.system.account)
        add("-p", self.system.default_partition)
        add("-g", args.gpu_type)
        add("-l", args.log_dir)
        add("-t", args.time_limit)
        if container_path:
            add("-i", container_path)
        add("-c", args.compute_dtype)
        add("--task", args.task)
        add("-hf", args.hf_token)
        add("-nh", args.nemo_home)
        add("-wdk", args.wandb_key)
        add("-wdp", args.wandb_prj_name)
        add("-wdj", args.wandb_exp_name)
        if args.dryrun:
            parts.append("-d")
        add("-ng", args.num_gpus)
        add("-gn", args.gpus_per_node)
        if mounts:
            add("-cm", ",".join(mounts))

        # Model flags
        add("-vb", "true" if bool(args.enable_vboost) else "false" if args.enable_vboost is not None else None)
        add("-m", args.model_name)
        add("-s", args.model_size)
        if args.enable_nsys:
            parts.append("-en")
        add("--domain", args.domain)
        if args.use_tokendrop is not None:
            add("--use_tokendrop", bool(args.use_tokendrop))
        if args.use_megatron_fsdp is not None:
            add("--use_megatron_fsdp", bool(args.use_megatron_fsdp))
        add("--cuda_graph_impl", args.cuda_graph_impl)
        if args.cuda_graph_scope:
            add("--cuda_graph_scope", self._normalize_cuda_graph_scope_arg(args.cuda_graph_scope))

        # Parallelism
        add("-tp", args.tp)
        add("-pp", args.pp)
        add("-cp", args.cp)
        add("-vp", args.vp)
        add("-ep", args.ep)
        add("-et", args.et)

        # Batch
        add("-mb", args.mb)
        add("-gb", args.gb)

        # Misc
        if args.moe_a2a_overlap is not None:
            add("--moe_a2a_overlap", bool(args.moe_a2a_overlap))
        add("-ms", args.max_steps)
        add("-rl", args.recompute_num_layers)
        add("-ol", args.activation_offload_layers)
        if args.recompute_modules:
            parts.extend(["--recompute_modules", self._normalize_recompute_modules(args.recompute_modules)])
        if args.no_detach:
            parts.append("--no-detach")

        # Extra user args (dict -> string)
        if tdef.extra_cmd_args:
            parts.append(tdef.extra_args_str)

        return parts
