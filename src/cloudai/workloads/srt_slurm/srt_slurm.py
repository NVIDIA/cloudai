# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, PrivateAttr, model_validator

from cloudai.core import GitRepo, Installable, JobStatusResult, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition

from .installable import SRTSlurmExecutable


class SRTSlurmCmdArgs(CmdArgs):
    """An existing srt-slurm recipe and the values CloudAI may tune."""

    recipe_path: Path
    cluster_config: Path
    result_format: Literal["sa-bench", "agentx"] = "sa-bench"
    # native passes typed --set arguments to srtctl; render writes a per-trial
    # recipe copy for older CLIs without --set, replacing existing keys only.
    recipe_override_mode: Literal["native", "render"] = "native"
    recipe_overrides: dict[str, Any] = Field(default_factory=dict)

    def excluded_cmd_args(self) -> list[str]:
        return [
            "cmd_args.recipe_path",
            "cmd_args.cluster_config",
            "cmd_args.result_format",
            "cmd_args.recipe_override_mode",
        ]


class SRTSlurmTestDefinition(TestDefinition):
    """Run one declarative srt-slurm recipe as one CloudAI trial."""

    cmd_args: SRTSlurmCmdArgs
    git_repos: list[GitRepo] = Field(
        default_factory=list, min_length=1, validate_default=True, description="The first repository supplies srtctl."
    )
    _srtctl: SRTSlurmExecutable | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def exclude_recipe_structure_from_dse(self) -> "SRTSlurmTestDefinition":
        self.dse_excluded_args = list(dict.fromkeys([*self.dse_excluded_args, *self.cmd_args.excluded_cmd_args()]))
        return self

    @property
    def srtctl(self) -> SRTSlurmExecutable:
        if self._srtctl is None:
            self._srtctl = SRTSlurmExecutable(git_repo=self.git_repos[0])
        return self._srtctl

    @property
    def srtctl_path(self) -> Path:
        if self.srtctl.venv_path is None:
            raise RuntimeError("srt-slurm must be installed before generating its command.")
        return self.srtctl.venv_path.absolute() / "bin" / "srtctl"

    @property
    def installables(self) -> list[Installable]:
        return [self.srtctl, *self.git_repos[1:]]

    @staticmethod
    def job_id_path(tr: TestRun) -> Path:
        """Locate the saved job ID used for result lookup and later report regeneration."""
        return tr.output_path / "srt_job_id.txt"

    def job_output_path(self, tr: TestRun) -> Path | None:
        """Return the submitted job's output directory, without guessing its identity."""
        # TestRun has no runtime SlurmJob ID; avoid guessing among old submission directories.
        try:
            job_id = self.job_id_path(tr).read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not job_id.isdigit():
            return None
        return tr.output_path / "srt_slurm_outputs" / job_id

    def metadata_path(self, tr: TestRun) -> Path | None:
        output = self.job_output_path(tr)
        return output / f"{output.name}.json" if output else None

    def rollup_path(self, tr: TestRun) -> Path | None:
        logs = self.logs_path(tr)
        return logs / "benchmark-rollup.json" if logs else None

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        from .results import load_results

        try:
            results = load_results(self, tr)
        except ValueError as error:
            return JobStatusResult(is_successful=False, error_message=str(error))
        if not results:
            return JobStatusResult(is_successful=False, error_message="No usable job-local srt-slurm results found.")
        return JobStatusResult(is_successful=True)

    def logs_path(self, tr: TestRun) -> Path | None:
        output = self.job_output_path(tr)
        return output / "logs" if output else None
