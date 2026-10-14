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

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from cloudai.core import GitRepo, TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.srt_slurm import (
    SRTSlurmCommandGenStrategy,
    SRTSlurmTestDefinition,
)


def make_tdef(tmp_path: Path) -> SRTSlurmTestDefinition:
    recipe_path = tmp_path / "source_recipe.yaml"
    recipe_path.write_text(
        """\
schema: 2
name: demo
model:
  path: model
  container: image
resources:
  gpu_type: h100
  gpus_per_node: 8
frontend:
  type: sglang
engine: sglang
roles:
  agg:
    nodes: 1
    workers: 1
    gpus: 1
benchmark:
  type: sa-bench
  isl: 128
  osl: 128
  concurrencies: "4"
""",
        encoding="utf-8",
    )
    (tmp_path / "cluster.yaml").write_text("default_partition: main\n")
    return SRTSlurmTestDefinition.model_validate(
        {
            "name": "srt",
            "description": "srt-slurm test",
            "test_template_name": "SRTSlurm",
            "git_repos": [{"url": "git@github.com:NVIDIA/srt-slurm.git", "commit": "test-revision"}],
            "cmd_args": {
                "recipe_path": recipe_path,
                "cluster_config": tmp_path / "cluster.yaml",
                "recipe_overrides": {"roles": {"agg": {"gpus": 4}}},
            },
            "extra_cmd_args": {"no-preflight": ""},
        }
    )


def write_job_metadata(tdef: SRTSlurmTestDefinition, tr: TestRun, **resources) -> Path:
    path = tdef.metadata_path(tr)
    assert path is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    defaults = {
        "gpus_per_node": 8,
        "agg_nodes": 1,
        "agg_workers": 1,
        "gpus_per_agg": 4,
        "prefill_workers": 0,
        "decode_workers": 0,
    }
    path.write_text(json.dumps({"job_id": path.stem, "resources": defaults | resources}))
    return path


def test_command_generation_returns_srt_slurm_job(slurm_system: SlurmSystem, tmp_path: Path):
    tdef = make_tdef(tmp_path)
    tdef.srtctl.venv_path = tmp_path / "venv"
    test_run = TestRun(name="srt", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path / "output")
    strategy = SRTSlurmCommandGenStrategy(slurm_system, test_run)

    command = strategy.gen_exec_command()
    run_script = test_run.output_path / "run_srtctl.sh"
    script = run_script.read_text(encoding="utf-8")

    subprocess.run(["bash", "-n", run_script], check=True)
    assert command == f"bash {run_script}"
    assert str(tdef.cmd_args.recipe_path) in script
    assert "--set roles.agg.gpus=4" in script
    assert not (run_script.parent / "recipe.yaml").exists()
    assert "--no-preflight" in run_script.read_text(encoding="utf-8")
    assert "srt_job_id.txt" in run_script.read_text(encoding="utf-8")
    assert 'echo "Submitted batch job $SRT_JOB_ID"' in run_script.read_text(encoding="utf-8")
    assert "--dependency" not in run_script.read_text(encoding="utf-8")


def test_run_script_exports_merged_environment(slurm_system: SlurmSystem, tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SRTSLURM_CONFIG", raising=False)
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "login-node-venv"))
    slurm_system.global_env_vars = {"SRTSLURM_CONFIG": "system.yaml", "SRT_GLOBAL_ONLY": "global value"}
    tdef = make_tdef(tmp_path)
    repo_path = tmp_path / "installed srt-slurm"
    repo_path.mkdir()
    tdef.git_repos[0].installed_path = repo_path
    caller_path = tmp_path / "cloudaix"
    (caller_path / ".venv" / "bin").mkdir(parents=True)
    (caller_path / ".venv" / "bin" / "python").write_bytes(b"invalid login-node interpreter")
    (caller_path / ".venv" / "pyvenv.cfg").write_text("home = /login-node/python\n")
    active_bin = tmp_path / "login-node-venv" / "bin"
    active_bin.mkdir(parents=True)
    (active_bin.parent / "pyvenv.cfg").write_text("home = /login-node/python\n")
    retained_bin = tmp_path / "cluster tools" / "bin"
    retained_bin.mkdir(parents=True)
    monkeypatch.setenv("PATH", f"{caller_path / '.venv' / 'bin'}:{active_bin}:{retained_bin}:{os.environ['PATH']}")
    tdef.extra_env_vars = {
        "SRTSLURM_CONFIG": str(tmp_path / "cluster's config" / "srtslurm.yaml"),
        "SRT_LITERAL": "$HOME; $(printf injected)",
        "SRT_EMPTY": "",
    }
    expected_config = str(tdef.cmd_args.cluster_config.absolute())
    tdef.srtctl.venv_path = tmp_path / "venv"
    tdef.srtctl_path.parent.mkdir(parents=True)
    tdef.srtctl_path.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({key: os.environ.get(key) for key in "
        "['SRTSLURM_CONFIG', 'SRT_GLOBAL_ONLY', 'SRT_LITERAL', 'SRT_EMPTY', 'VIRTUAL_ENV']}))\n"
        "print(os.getcwd())\n"
        "print(os.environ['PATH'])\n"
        "print(json.dumps(sys.argv[1:]))\n"
        "print('Job 12345 submitted')\n",
        encoding="utf-8",
    )
    tdef.srtctl_path.chmod(0o755)
    test_run = TestRun(name="srt", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path / "output")
    strategy = SRTSlurmCommandGenStrategy(slurm_system, test_run)
    strategy.gen_exec_command()

    subprocess.run(
        ["bash", strategy._paths()["run_script"]], cwd=caller_path, check=True, capture_output=True, text=True
    )
    received = json.loads(strategy._paths()["run_log"].read_text(encoding="utf-8").splitlines()[0])

    assert received == {
        "SRT_GLOBAL_ONLY": "global value",
        **tdef.extra_env_vars,
        "SRTSLURM_CONFIG": expected_config,
        "VIRTUAL_ENV": None,
    }
    assert Path(strategy._paths()["run_log"].read_text().splitlines()[1]) == repo_path
    submitted_path = strategy._paths()["run_log"].read_text().splitlines()[2].split(os.pathsep)
    assert str(caller_path / ".venv" / "bin") not in submitted_path
    assert str(active_bin) not in submitted_path
    assert str(retained_bin) in submitted_path
    submitted_args = json.loads(strategy._paths()["run_log"].read_text().splitlines()[3])
    assert submitted_args[submitted_args.index("-f") + 1] == str(tdef.cmd_args.recipe_path)
    assert submitted_args[submitted_args.index("--set") + 1] == "roles.agg.gpus=4"
    assert tdef.job_id_path(test_run).read_text(encoding="utf-8").strip() == "12345"


def test_existing_recipe_is_loaded_and_dse_values_are_overlaid(tmp_path: Path):
    source = tmp_path / "existing.yaml"
    original = """\
schema: 2
name: existing
roles:
  agg:
    gpus: 1
    args:
      tensor-parallel-size: 1
benchmark:
  concurrencies: [4, 8]
"""
    source.write_text(original, encoding="utf-8")
    tdef = SRTSlurmTestDefinition.model_validate(
        {
            "name": "external",
            "description": "existing recipe",
            "test_template_name": "SRTSlurm",
            "git_repos": [{"url": "git@github.com:NVIDIA/srt-slurm.git", "commit": "test-revision"}],
            "cmd_args": {
                "recipe_path": source,
                "cluster_config": tmp_path / "cluster.yaml",
                "recipe_overrides": {
                    "roles": {"agg": {"gpus": [1, 2, 4]}},
                    "benchmark": {"concurrencies": [[32, 64], [64, 128]]},
                },
            },
        }
    )

    test_run = TestRun(name="dse", test=tdef, num_nodes=1, nodes=[])
    assert test_run.param_space == {
        "recipe_overrides.roles.agg.gpus": [1, 2, 4],
        "recipe_overrides.benchmark.concurrencies": [[32, 64], [64, 128]],
    }
    selected = test_run.apply_params_set(
        {
            "recipe_overrides.roles.agg.gpus": 4,
            "recipe_overrides.benchmark.concurrencies": [32, 64],
        }
    )
    assert isinstance(selected.test, SRTSlurmTestDefinition)
    assert SRTSlurmCommandGenStrategy._override_args(selected.test.cmd_args.recipe_overrides) == [
        "--set",
        "roles.agg.gpus=4",
        "--set",
        "benchmark.concurrencies=[32, 64]",
    ]
    assert source.read_text(encoding="utf-8") == original
    assert tdef.dse_excluded_args == [
        "cmd_args.recipe_path",
        "cmd_args.cluster_config",
        "cmd_args.result_format",
        "cmd_args.recipe_override_mode",
    ]
    assert "cmd_args.recipe_overrides" not in tdef.dse_excluded_args


def test_standard_git_repos_controls_srtctl_revision(tmp_path: Path):
    config = make_tdef(tmp_path).model_dump()
    config["extra_env_vars"] = {"SRTSLURM_CONFIG": str(tmp_path / "cluster.yaml")}
    config["git_repos"][0]["commit"] = "updated-revision"
    config["git_repos"].append({"url": "https://github.com/example/recipes.git", "commit": "main"})
    tdef = SRTSlurmTestDefinition.model_validate(config)

    assert tdef.srtctl.git_repo is tdef.git_repos[0]
    assert tdef.srtctl.git_repo.commit == "updated-revision"
    assert "updated-revision" in tdef.srtctl.venv_name
    assert tdef.installables == [tdef.srtctl, tdef.git_repos[1]]
    assert isinstance(tdef.installables[1], GitRepo)


@pytest.mark.parametrize("git_repos", [None, []])
def test_srtctl_requires_a_configured_git_repo(tmp_path: Path, git_repos):
    config = make_tdef(tmp_path).model_dump()
    if git_repos is None:
        config.pop("git_repos")
    else:
        config["git_repos"] = git_repos

    with pytest.raises(ValidationError, match="git_repos"):
        SRTSlurmTestDefinition.model_validate(config)


@pytest.mark.parametrize("value", [4, 0.75, True, False, None, [32, 64], "true", "720", "", "$HOME; $(id)"])
def test_set_arguments_preserve_yaml_value_types(value):
    args = SRTSlurmCommandGenStrategy._override_args({"roles": {"agg": {"args": {"flag": value}}}})
    assert args[0] == "--set"
    key, raw = args[1].split("=", 1)
    assert key == "roles.agg.args.flag"
    parsed = yaml.safe_load(raw)
    assert parsed == value
    assert type(parsed) is type(value)


def test_set_arguments_quote_literal_dotted_keys():
    assert SRTSlurmCommandGenStrategy._override_args({"container_mounts": {"/models/model.v1": "/model"}}) == [
        "--set",
        'container_mounts."/models/model.v1"="/model"',
    ]
