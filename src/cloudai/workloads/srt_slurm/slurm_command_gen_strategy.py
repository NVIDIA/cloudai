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
from pathlib import Path
from typing import Any, cast

import yaml

from cloudai.systems.slurm import SlurmCommandGenStrategy

from .srt_slurm import SRTSlurmTestDefinition


class SRTSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Submit the recipe job directly and let CloudAI monitor its job ID."""

    @staticmethod
    def _load_recipe(path: Path) -> dict[str, Any]:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise ValueError(f"Unable to read srt-slurm recipe {path}: {error}") from error
        except yaml.YAMLError as error:
            raise ValueError(f"Unable to parse srt-slurm recipe {path}: {error}") from error
        if not isinstance(data, dict):
            raise ValueError(f"srt-slurm recipe {path} must contain a YAML mapping.")
        return data

    @staticmethod
    def _override_args(overrides: dict[str, Any], prefix: str = "") -> list[str]:
        """Flatten DSE-selected values into typed srtctl --set arguments."""
        args: list[str] = []
        for key, value in overrides.items():
            segment = f'"{key}"' if any(char in key for char in ".[]") else key
            path = f"{prefix}.{segment}" if prefix else segment
            if isinstance(value, dict):
                args.extend(SRTSlurmCommandGenStrategy._override_args(value, path))
            else:
                # JSON is valid YAML and quotes strings such as "true" and "720".
                args.extend(["--set", f"{path}={json.dumps(value)}"])
        return args

    @property
    def tdef(self) -> SRTSlurmTestDefinition:
        return cast(SRTSlurmTestDefinition, self.test_run.test)

    def _paths(self) -> dict[str, Path]:
        output = self.test_run.output_path.absolute()
        return {
            "recipe": self.tdef.cmd_args.recipe_path.expanduser().absolute(),
            "rendered_recipe": output / "srt_slurm_recipe.yaml",
            "nested_outputs": output / "srt_slurm_outputs",
            "run_log": output / "run_srtctl.log",
            "run_script": output / "run_srtctl.sh",
            "job_id": self.tdef.job_id_path(self.test_run).absolute(),
        }

    def _prepare_recipe(self, paths: dict[str, Path]) -> tuple[Path, list[str]]:
        """Select the original recipe with CLI overrides or write a rendered copy."""
        recipe = paths["recipe"]
        overrides = self._override_args(self.tdef.cmd_args.recipe_overrides)
        # "render" mode is for older CLIs that don't support passing `--set` to srtctl.
        # Apply the overrides to a generated recipe instead of passing them on the CLI.
        if self.tdef.cmd_args.recipe_override_mode == "render":
            data = self._load_recipe(recipe)
            self._overlay_recipe(data, self.tdef.cmd_args.recipe_overrides)
            recipe = paths["rendered_recipe"]
            recipe.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
            overrides = []
        return recipe, overrides

    def write_run_script(self) -> None:
        paths = self._paths()
        repo = self.tdef.srtctl.git_repo
        repo_path = (repo.installed_path or self.system.install_path / repo.repo_name).absolute()
        cluster_config = self.tdef.cmd_args.cluster_config.expanduser().absolute()
        recipe, overrides = self._prepare_recipe(paths)
        apply_command = shlex.join(
            [
                str(self.tdef.srtctl_path),
                "apply",
                "-f",
                str(recipe),
                "-o",
                str(paths["nested_outputs"]),
                "-y",
                *overrides,
                *self._extra_cmd_arg_parts(),
            ]
        )
        script = "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "",
                "# Configure submission environment",
                *[f"export {key}={shlex.quote(str(value))}" for key, value in self.final_env_vars.items()],
                f"export SRTSLURM_CONFIG={shlex.quote(str(cluster_config))}",
                *(["export AGENTIC_OUTPUT_DIR=/logs/agentic"] if self.tdef.cmd_args.result_format == "agentx" else []),
                "",
                "# Prevent login-node Python environments from leaking into compute jobs.",
                "# A: uv searches PATH after VIRTUAL_ENV. Remove every virtualenv-backed",
                "# PATH entry so it cannot select an incompatible login-node python/python3.",
                'IFS=: read -r -a SRT_PATH_PARTS <<< "$PATH"',
                "SRT_CLEAN_PATH=()",
                'for SRT_PATH_PART in "${SRT_PATH_PARTS[@]}"; do',
                '    if [[ ! -f "$SRT_PATH_PART/../pyvenv.cfg" ]]; then',
                '        SRT_CLEAN_PATH+=("$SRT_PATH_PART")',
                "    fi",
                "done",
                'export PATH="$(IFS=:; printf \'%s\' "${SRT_CLEAN_PATH[*]}")"',
                "# B: uv prioritizes the active environment. Unset VIRTUAL_ENV so it cannot",
                "# inspect or execute CloudAI's login-node interpreter on a compute node.",
                "unset VIRTUAL_ENV",
                "",
                "# Submit the recipe",
                f"mkdir -p {shlex.quote(str(paths['nested_outputs']))}",
                "# Clear the old result lookup ID in case this wrapper is manually rerun and submission fails.",
                f"rm -f {shlex.quote(str(paths['job_id']))}",
                "# C: Slurm inherits this working directory. Leave CloudAI's checkout so",
                "# compute-side uv does not discover its login-node .venv, even with --no-project.",
                f"cd {shlex.quote(str(repo_path))}",
                f"if ! {apply_command} > {shlex.quote(str(paths['run_log']))} 2>&1; then",
                f"    cat {shlex.quote(str(paths['run_log']))} >&2",
                "    exit 1",
                "fi",
                f"cat {shlex.quote(str(paths['run_log']))} >&2",
                "",
                "# Record the submitted job ID for CloudAI",
                "if ! SRT_JOB_ID=\"$(sed -nE 's/.*Job ([0-9]+) submitted.*/\\1/p; "
                "s/.*Submitted batch job ([0-9]+).*/\\1/p' "
                f'{shlex.quote(str(paths["run_log"]))} | tail -1)" || [[ ! "$SRT_JOB_ID" =~ ^[0-9]+$ ]]; then',
                '    echo "Unable to find submitted srt-slurm job ID" >&2',
                "    exit 1",
                "fi",
                "# Persist the result lookup ID now; slurm-job.toml is written at completion",
                "# only when accounting is available. Stdout separately supplies the runner ID.",
                f'printf "%s\\n" "$SRT_JOB_ID" > {shlex.quote(str(paths["job_id"]))}',
                'echo "Submitted batch job $SRT_JOB_ID"',
                "",
            ]
        )
        paths["run_script"].write_text(script, encoding="utf-8")

    @staticmethod
    def _overlay_recipe(data: dict[str, Any], overrides: dict[str, Any], prefix: str = "") -> None:
        """Overlay existing keys of a concrete recipe for older CLIs without --set."""
        for key, value in overrides.items():
            path = f"{prefix}.{key}" if prefix else key
            if key not in data:
                raise ValueError(f"Rendered recipe override targets an absent key: {path}")
            if isinstance(value, dict):
                if not isinstance(data[key], dict):
                    raise ValueError(f"Rendered recipe override targets a non-mapping: {path}")
                SRTSlurmCommandGenStrategy._overlay_recipe(data[key], value, path)
            else:
                data[key] = value

    def _extra_cmd_arg_parts(self) -> list[str]:
        args = []
        for name, value in self.tdef.extra_cmd_args.items():
            args.append(f"--{name.replace('_', '-')}")
            if value:
                args.append(value)
        return args

    def _container_mounts(self) -> list[str]:
        return []

    def store_test_run(self) -> None:
        # SlurmRunner records command metadata before submitting the test.
        # Prepare the wrapper first, without executing srtctl.
        self.gen_exec_command()
        super().store_test_run()

    def gen_exec_command(self) -> str:
        self.test_run.output_path.mkdir(parents=True, exist_ok=True)
        self._load_recipe(self._paths()["recipe"])
        if not self.tdef.cmd_args.cluster_config.expanduser().is_file():
            raise ValueError("The supplied srt-slurm cluster_config does not exist.")
        if self.system.uses_slurm_api:
            raise ValueError("SRTSlurm requires Slurm CLI submission.")
        if self.test_run.pre_test or self.test_run.post_test:
            raise ValueError("SRTSlurm hooks must be configured in the recipe.")
        self.write_run_script()
        return self.gen_srun_command()

    def gen_srun_command(self) -> str:
        if not self._paths()["run_script"].is_file():
            raise ValueError("SRTSlurm submits its own job; single-sbatch execution is unsupported.")
        return shlex.join(["bash", str(self._paths()["run_script"])])
