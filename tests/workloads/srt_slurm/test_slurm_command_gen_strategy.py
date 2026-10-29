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

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from cloudai.core import TestRun, TestScenario
from cloudai.systems.slurm import SlurmRunner, SlurmSystem
from cloudai.workloads.srt_slurm import SRTSlurmCommandGenStrategy


@pytest.mark.parametrize("schema", ["", "schema: 2\n"])
def test_runner_tracks_the_recipe_job(slurm_system: SlurmSystem, tmp_path: Path, schema: str, monkeypatch, make_tdef):
    tdef = make_tdef(tmp_path)
    tdef.cmd_args.recipe_path.write_text(schema + "name: original\n")
    tdef.srtctl.venv_path = tmp_path / "venv"
    tdef.srtctl_path.parent.mkdir(parents=True)
    repo = slurm_system.install_path / tdef.git_repos[0].repo_name
    repo.mkdir(parents=True)
    tdef.git_repos[0].installed_path = repo
    tdef.srtctl_path.write_text(f"#!{sys.executable}\nprint('Submitted batch job 111')\nprint('Job 222 submitted')\n")
    tdef.srtctl_path.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    tr = TestRun(name="srt", test=tdef, num_nodes=1, nodes=[], output_path=Path("relative output"))
    runner = SlurmRunner("run", slurm_system, TestScenario(name="srt", test_runs=[tr]), tmp_path)
    runner.submit_test(tr)
    job = runner.jobs[0]
    assert job.id == 222
    assert tdef.job_id_path(tr).read_text().strip() == "222"
    assert (tr.output_path / "test-run.toml").is_file()


def test_failed_submission_does_not_publish_a_job_id(slurm_system: SlurmSystem, tmp_path: Path, make_tdef):
    tdef = make_tdef(tmp_path)
    tdef.srtctl.venv_path = tmp_path / "venv"
    tdef.srtctl_path.parent.mkdir(parents=True)
    repo = slurm_system.install_path / tdef.git_repos[0].repo_name
    repo.mkdir(parents=True)
    tdef.srtctl_path.write_text(f"#!{sys.executable}\nprint('Submitted batch job 111')\nraise SystemExit(2)\n")
    tdef.srtctl_path.chmod(0o755)
    tr = TestRun(name="srt", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path / "output")
    strategy = SRTSlurmCommandGenStrategy(slurm_system, tr)
    strategy.gen_exec_command()
    completed = subprocess.run(["bash", strategy._paths()["run_script"]], capture_output=True, text=True)
    assert completed.returncode != 0
    assert "Submitted batch job" not in completed.stdout
    assert not tdef.job_id_path(tr).exists()


def test_single_sbatch_rejected(slurm_system: SlurmSystem, tmp_path: Path, make_tdef):
    tdef = make_tdef(tmp_path)
    tr = TestRun(name="srt", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path / "output")
    with pytest.raises(ValueError, match="single-sbatch"):
        SRTSlurmCommandGenStrategy(slurm_system, tr).gen_srun_command()


@pytest.mark.parametrize("extraction_fails", [False, True])
def test_job_id_errors_are_reported_to_stderr(
    slurm_system: SlurmSystem, tmp_path: Path, monkeypatch, extraction_fails, make_tdef
):
    tdef = make_tdef(tmp_path)
    tdef.srtctl.venv_path = tmp_path / "venv"
    tdef.srtctl_path.parent.mkdir(parents=True)
    repo = slurm_system.install_path / tdef.git_repos[0].repo_name
    repo.mkdir(parents=True)
    response = "Submitted batch job 111" if extraction_fails else "Submission completed without a job ID"
    tdef.srtctl_path.write_text(f"#!{sys.executable}\nprint({response!r})\n")
    tdef.srtctl_path.chmod(0o755)
    if extraction_fails:
        tools = tmp_path / "tools"
        tools.mkdir()
        sed = tools / "sed"
        sed.write_text("#!/usr/bin/env bash\nexit 1\n")
        sed.chmod(0o755)
        monkeypatch.setenv("PATH", f"{tools}:{os.environ['PATH']}")
    tr = TestRun(name="srt", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path / "output")
    strategy = SRTSlurmCommandGenStrategy(slurm_system, tr)
    strategy.gen_exec_command()
    tdef.job_id_path(tr).write_text("999\n")
    completed = subprocess.run(["bash", strategy._paths()["run_script"]], capture_output=True, text=True)
    assert completed.returncode != 0
    assert completed.stdout == ""
    assert response in completed.stderr
    assert "Unable to find submitted srt-slurm job ID" in completed.stderr
    assert not tdef.job_id_path(tr).exists()


def test_rendered_overrides_support_old_cli(slurm_system: SlurmSystem, tmp_path: Path, make_tdef):
    tdef = make_tdef(tmp_path)
    original = tdef.cmd_args.recipe_path.read_text()
    tdef.cmd_args.recipe_override_mode = "render"
    tdef.srtctl.venv_path = tmp_path / "venv"
    tr = TestRun(name="srt", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path / "results")
    strategy = SRTSlurmCommandGenStrategy(slurm_system, tr)
    strategy.gen_exec_command()
    rendered = tr.output_path / "srt_slurm_recipe.yaml"
    assert (tr.output_path / "run_srtctl.sh").is_file()
    assert not (tr.output_path / "cloudai_generated").exists()
    assert yaml.safe_load(rendered.read_text())["roles"]["agg"]["gpus"] == 4
    assert tdef.cmd_args.recipe_path.read_text() == original
    assert str(rendered) in strategy._paths()["run_script"].read_text()
    assert "--set" not in strategy._paths()["run_script"].read_text()
    tdef.cmd_args.recipe_overrides = {"roles": {"agg": {"typo": 1}}}
    with pytest.raises(ValueError, match=r"absent key: roles\.agg\.typo"):
        strategy.gen_exec_command()
