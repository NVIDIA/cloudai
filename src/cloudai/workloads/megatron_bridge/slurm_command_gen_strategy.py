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
            "set -euo pipefail",
            "",
            f'export NEMORUN_HOME="{output_dir}"',
            'mkdir -p "$NEMORUN_HOME"',
            f'LOG="{log_path}"',
            "",
            # Launch Megatron-Bridge (log stdout/stderr to file)
            "",
            ': >"$LOG"',
            f'{launcher_cmd} >>"$LOG" 2>&1',
            "",
            # Parse job id from Megatron-Bridge output (format: 'Job id: <num>')
            "",
            'JOB_ID=""',
            'JOB_ID=$(grep -Eio "Job id[: ]+[0-9]+" "$LOG" | tail -n1 | grep -Eo "[0-9]+" | tail -n1 || true)',
            "",
            # Emit a canonical line for CloudAI to parse
            "",
            'if [ -n "${JOB_ID}" ]; then',
            '  echo "Submitted batch job ${JOB_ID}"',
            "else",
            '  echo "Failed to retrieve job ID." >&2',
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
            "model_family_name",
            "model_recipe_name",
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
        add_field("time_limit", "-t", args.time_limit)
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
        add_field("gpus_per_node", "-gn", args.gpus_per_node)
        if mounts:
            add("-cm", ",".join(mounts))

        # Model flags (Megatron-Bridge main-branch API)
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
        if args.enable_nsys and "enable_nsys" in fields_set:
            parts.append("-en")
        add_field("domain", "--domain", args.domain)
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
        add_field("vp", "-vp", args.vp)
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
        if "detach" in fields_set and args.detach is not None:
            parts.extend(["--detach", "true" if args.detach else "false"])

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

        # Extra user args (dict -> string)
        if tdef.extra_cmd_args:
            parts.append(tdef.extra_args_str)

        return parts
