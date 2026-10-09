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
    recipe_overrides: dict[str, Any] = Field(default_factory=dict)

    def excluded_cmd_args(self) -> list[str]:
        return ["cmd_args.recipe_path", "cmd_args.cluster_config", "cmd_args.result_format"]


class SRTSlurmTestDefinition(TestDefinition):
    """Run one declarative srt-slurm recipe as one CloudAI trial."""

    cmd_args: SRTSlurmCmdArgs
    serving_gpus: int | None = Field(default=None, gt=0)
    prefill_gpus: int | None = Field(default=None, gt=0)
    decode_gpus: int | None = Field(default=None, gt=0)
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
        return tr.output_path / "srt_job_id.txt"

    def rollup_path(self, tr: TestRun) -> Path | None:
        try:
            job_id = self.job_id_path(tr).read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not job_id.isdigit():
            return None
        return tr.output_path / "srt_slurm_outputs" / job_id / "logs" / "benchmark-rollup.json"

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        from .results import load_results

        if not load_results(self, tr):
            return JobStatusResult(is_successful=False, error_message="No usable job-local srt-slurm results found.")
        return JobStatusResult(is_successful=True)

    def logs_path(self, tr: TestRun) -> Path | None:
        rollup = self.rollup_path(tr)
        return rollup.parent if rollup else None
