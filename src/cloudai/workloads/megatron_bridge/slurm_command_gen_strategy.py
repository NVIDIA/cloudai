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
import re
import shlex
import shutil
import subprocess
import textwrap
from functools import cached_property
from pathlib import Path
from typing import Any, Optional, cast

import toml

from cloudai.systems.slurm import SlurmCommandGenStrategy

from .megatron_bridge import HF_TOKEN_REDACTION, MegatronBridgeCmdArgs, MegatronBridgeTestDefinition


class MegatronBridgeSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Compose a NeMo Run generated job with CloudAI's standard Slurm features."""

    HF_TOKEN_ARTIFACT_SUFFIXES: frozenset[str] = frozenset({".log", ".out", ".sh", ".stderr", ".toml", ".yaml"})

    @property
    def final_env_vars(self) -> dict[str, str | list[str]]:
        tdef = cast(MegatronBridgeTestDefinition, self.test_run.test_definition)
        env_vars = tdef.cmd_args.executor_env_vars | super().final_env_vars
        if "enable_vboost" in tdef.cmd_args.model_fields_set:
            env_vars["ENABLE_VBOOST"] = "1" if tdef.cmd_args.enable_vboost else "0"
        return env_vars

    @final_env_vars.setter
    def final_env_vars(self, value: dict[str, str | list[str]]) -> None:
        self._final_env_vars = value

    def image_path(self) -> str | None:
        tdef = cast(MegatronBridgeTestDefinition, self.test_run.test_definition)
        return str(tdef.docker_image.installed_path) if tdef.docker_image.installed_path else None

    def _container_mounts(self) -> list[str]:
        return []

    @cached_property
    def _mbridge_sbatch(self) -> tuple[list[str], str]:
        script = self._generate_mbridge_sbatch()
        if script is None:
            return [], (
                "# MBridge script generation skipped: launcher dependencies are missing.\n"
                'echo "Install MBridge and NeMo Run, then regenerate this script." >&2\n'
                "exit 1"
            )
        directives: list[str] = []
        body: list[str] = []
        in_header = True
        for line in script.read_text().splitlines():
            if in_header and line.startswith("#SBATCH"):
                directives.append(line)
            elif in_header and (not line.strip() or line.startswith("#")):
                continue
            else:
                in_header = False
                body.append(line)
        return directives, "\n".join(body)

    @staticmethod
    def _directive_name(line: str) -> str:
        option = shlex.split(line.removeprefix("#SBATCH").strip())[0].split("=", 1)[0]
        aliases = {
            "-A": "--account",
            "-p": "--partition",
            "-J": "--job-name",
            "-N": "--nodes",
            "-n": "--ntasks",
            "-o": "--output",
            "-e": "--error",
            "-t": "--time",
            "-w": "--nodelist",
            "-x": "--exclude",
            "-D": "--chdir",
        }
        return aliases.get(option[:2], option) if not option.startswith("--") else option

    def _append_sbatch_directives(self, batch_script_content: list[str]) -> None:
        cloudai_directives: list[str] = []
        super()._append_sbatch_directives(cloudai_directives)
        owned = {"--job-name", "--output", "--error", "--nodes", "--nodelist", "--exclude"}
        owned.update(self._directive_name(line) for line in cloudai_directives if line.startswith("#SBATCH"))
        batch_script_content.extend(line for line in self._mbridge_sbatch[0] if self._directive_name(line) not in owned)
        batch_script_content.extend(cloudai_directives)

    def generate_test_command(self) -> list[str]:
        return [self._mbridge_sbatch[1]]

    def _gen_srun_command(self) -> str:
        self.test_run.output_path.mkdir(parents=True, exist_ok=True)
        env_vars_path = self._write_env_vars_script()
        body: list[str] = [f"source {shlex.quote(str(env_vars_path))}"]
        for line in self._mbridge_sbatch[1].splitlines():
            # Rendezvous accepts a hostname; avoid capturing srun output redirected by SingleSbatchRunner.
            line = line.replace(
                'head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)',
                'head_node_ip="$head_node"',
            )
            export = re.match(r"\s*export ([A-Za-z_][A-Za-z0-9_]*)=", line)
            if export and export[1] in self.final_env_vars:
                continue
            body.append(line)
        # NeMo's shell options and exit must not escape into CloudAI's post-hook block.
        return "(\n" + textwrap.indent("\n".join(body), "    ") + "\n)"

    def _generate_mbridge_sbatch(self) -> Path | None:
        existing_script = next(self.test_run.output_path.glob("experiments/**/*_sbatch.sh"), None)
        if existing_script:
            return existing_script

        tdef = cast(MegatronBridgeTestDefinition, self.test_run.test_definition)
        repo_path = tdef.megatron_bridge_repo.installed_path or (
            self.system.install_path / tdef.megatron_bridge_repo.repo_name
        )
        launcher_py = (repo_path / "scripts" / "performance" / "setup_experiment.py").absolute()
        venv_path = tdef.python_executable.venv_path
        if venv_path is None or not launcher_py.is_file() or not (venv_path / "bin" / "python").is_file():
            logging.warning(
                "MBridge launcher dependencies are missing; generating a placeholder script. "
                "Install MBridge and NeMo Run, then regenerate this script."
            )
            return None

        self.test_run.output_path.mkdir(parents=True, exist_ok=True)
        patcher_path = self._write_dryrun_patcher(launcher_py)
        parts = self._build_launcher_parts(tdef.cmd_args, tdef, repo_path, patcher_path)
        if "-d" not in parts:
            parts.append("-d")
        launcher_cmd = " ".join(parts)
        (self.test_run.output_path / "cloudai_megatron_bridge_launcher.sh").write_text(launcher_cmd + "\n")
        log_path = self.test_run.output_path / "cloudai_megatron_bridge_launcher.log"
        with log_path.open("w") as log:
            result = subprocess.run(
                ["/bin/bash", "-c", launcher_cmd],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(f"Megatron-Bridge dry-run failed. See {log_path}.")

        script = next(self.test_run.output_path.glob("experiments/**/*_sbatch.sh"), None)
        if script is None:
            raise RuntimeError(f"Megatron-Bridge dry-run did not generate an sbatch script. See {log_path}.")
        return script

    def _write_dryrun_patcher(self, launcher_py: Path) -> Path:
        patcher_path = self.test_run.output_path / "cloudai_megatron_bridge_dryrun.py"
        patcher_path.write_text(
            "\n".join(
                [
                    "import runpy",
                    "import sys",
                    "from pathlib import Path",
                    "from nemo_run.run.experiment import Experiment",
                    "",
                    "original_dryrun = Experiment.dryrun",
                    "",
                    "def keep_dryrun_artifacts(self, log=True, exist_ok=False, delete_exp_dir=True):",
                    "    return original_dryrun(self, log=log, exist_ok=exist_ok, delete_exp_dir=False)",
                    "",
                    "Experiment.dryrun = keep_dryrun_artifacts",
                    f"launcher = Path({str(launcher_py)!r})",
                    "sys.path.insert(0, str(launcher.parent))",
                    'runpy.run_path(str(launcher), run_name="__main__")',
                    "",
                ]
            )
        )
        return patcher_path

    def store_test_run(self) -> None:
        self.test_run.output_path.mkdir(parents=True, exist_ok=True)
        details = self.test_run.model_dump(mode="json")
        details["test_definition"]["cmd_args"]["hf_token"] = HF_TOKEN_REDACTION
        token = cast(MegatronBridgeTestDefinition, self.test_run.test_definition).cmd_args.hf_token
        serialized = toml.dumps(details)
        if token:
            serialized = serialized.replace(token, HF_TOKEN_REDACTION)
        (self.test_run.output_path / self.TEST_RUN_DUMP_FILE_NAME).write_text(serialized)

    def cleanup_job_artifacts(self) -> None:  # noqa: C901
        tdef = cast(MegatronBridgeTestDefinition, self.test_run.test_definition)
        output_path = self.test_run.output_path
        if not output_path.exists():
            return

        cleanup_failures: list[Path] = []
        experiments_path = output_path / "experiments"
        if experiments_path.is_dir():
            for code_path in list(experiments_path.rglob("code")):
                if code_path.is_symlink() or not code_path.is_dir():
                    continue
                try:
                    shutil.rmtree(code_path)
                    logging.debug("Removed packaged code directory from job artifacts: %s", code_path)
                except OSError:
                    cleanup_failures.append(code_path)
                    logging.warning("Failed to remove packaged code directory: %s", code_path, exc_info=True)

        token = tdef.cmd_args.hf_token
        if token and token != HF_TOKEN_REDACTION:
            token_bytes = token.encode()
            redaction_bytes = b"X" * len(token_bytes)
            for path in output_path.rglob("*"):
                if path.is_symlink() or not path.is_file():
                    continue
                if path.suffix.lower() not in self.HF_TOKEN_ARTIFACT_SUFFIXES:
                    continue

                try:
                    redacted = False
                    with path.open("r+b") as f:
                        tail = b""
                        while chunk := f.read(1024 * 1024):
                            contents = tail + chunk
                            read_end = f.tell()
                            start = 0
                            while (match := contents.find(token_bytes, start)) != -1:
                                f.seek(read_end - len(contents) + match)
                                f.write(redaction_bytes)
                                start = match + len(token_bytes)
                                redacted = True
                            f.seek(read_end)
                            # Retain unprocessed bytes that could start a token across chunks.
                            tail = contents[max(start, len(contents) - len(token_bytes) + 1) :]
                    if redacted:
                        logging.debug("Redacted Hugging Face token from job artifact: %s", path)
                except OSError:
                    cleanup_failures.append(path)
                    logging.warning("Failed to redact Hugging Face token from job artifact: %s", path, exc_info=True)

        if cleanup_failures:
            raise RuntimeError(f"Failed to clean {len(cleanup_failures)} job artifact(s).")

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
            "logger.tensorboard_dir": "/nemo_run/tb_logs",
            "logger.log_timers_to_tensorboard": "true",
            "logger.log_throughput_to_tensorboard": "true",
            "logger.log_memory_to_tensorboard": "true",
        }
        overrides.update(extra_cmd_args)
        return [shlex.quote(f"{key}={value}" if value else key) for key, value in overrides.items()]

    def _build_custom_srun_args(self, extra_cmd_args: dict[str, str]) -> list[str]:
        """Consume native srun options and combine them with CloudAI arguments."""
        srun_args = [extra_cmd_args.pop(flag) for flag in ("-cs", "--custom_srun_args") if flag in extra_cmd_args]
        srun_args.extend(arg for arg in (self.system.extra_srun_args, self.test_run.extra_srun_args) if arg)
        if self.test_run.pre_test:
            for tr in self.test_run.pre_test.test_runs:
                srun_args.extend(self._get_cmd_gen_strategy(tr).pre_test_srun_extra_args(self.test_run))
        if custom_srun_args := ",".join(arg for arg in srun_args if arg):
            return [shlex.quote(f"-cs={custom_srun_args}")]
        return []

    @staticmethod
    def _with_default_packager(repo_path: Path, extra_cmd_args: dict[str, str]) -> dict[str, str]:
        """Disable source packaging when the installed Megatron-Bridge version supports the option."""
        overrides = extra_cmd_args.copy()
        argument_parser_path = repo_path / "scripts" / "performance" / "argument_parser.py"
        try:
            argument_parser_source = argument_parser_path.read_text()
        except OSError:
            logging.debug("Unable to inspect Megatron-Bridge argument parser at %s", argument_parser_path)
        else:
            # r0.3.0 forwards unknown setup_experiment.py arguments to run_script.py, where Hydra rejects them.
            # Only inject this default when the installed Megatron-Bridge parser advertises native support.
            if '"--packager"' in argument_parser_source or "'--packager'" in argument_parser_source:
                overrides.setdefault("--packager", "none")
        return overrides

    def _build_launcher_parts(  # noqa: C901
        self,
        args: MegatronBridgeCmdArgs,
        tdef: MegatronBridgeTestDefinition,
        repo_path: Path,
        launcher_py: Path,
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

        mounts = self.container_mounts()

        venv_path = tdef.python_executable.venv_path
        if venv_path is None:
            raise RuntimeError("NeMo Run Python environment path is not set.")
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
        if args.container_image:
            add_field("container_image", "-i", self.image_path())
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
        parts.extend(["-cb", shlex.quote(f"source {self.CONTAINER_MOUNT_OUTPUT}/env_vars.sh")])

        # Model flags (Megatron-Bridge main-branch API)
        add_field("domain", "--domain", args.domain)
        if args.use_recipes and "use_recipes" in fields_set:
            parts.append("--use_recipes")
        add("-vb", False)
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
        add_field(
            "moe_flex_dispatcher_backend",
            "--moe_flex_dispatcher_backend",
            args.moe_flex_dispatcher_backend,
        )
        add_field("max_steps", "-ms", args.max_steps)
        add_field("recompute_num_layers", "-rl", args.recompute_num_layers)
        add_field("activation_offload_layers", "-ol", args.activation_offload_layers)
        if args.recompute_modules and "recompute_modules" in fields_set:
            parts.extend(["--recompute_modules", self._normalize_recompute_modules(args.recompute_modules)])

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

        # Config variant
        add_field("config_variant", "-cv", args.config_variant)
        if args.list_config_variants and "list_config_variants" in fields_set:
            parts.append("--list_config_variants")

        # Extra args (dict -> Hydra overrides): defaults first, then user values which take precedence.
        extra_cmd_args = self._with_default_packager(repo_path, tdef.extra_cmd_args)
        parts.extend(self._build_custom_srun_args(extra_cmd_args))
        parts.extend(self._add_extra_cmd_args(extra_cmd_args))

        return parts
