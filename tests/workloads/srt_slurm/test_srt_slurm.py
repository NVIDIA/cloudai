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

from pathlib import Path

import pytest
import yaml

from cloudai.core import GitRepo, HFModel, TestRun
from cloudai.workloads.srt_slurm import (
    SRTSlurmCommandGenStrategy,
    SRTSlurmTestDefinition,
)


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


@pytest.mark.parametrize("srt_first", [True, False])
@pytest.mark.parametrize("url", ["https://github.com/example/srt-slurm.git", "git@github.com:example/srt-slurm"])
def test_standard_git_repos_controls_srtctl_revision(tmp_path: Path, make_tdef, srt_first: bool, url: str):
    config = make_tdef(tmp_path).model_dump()
    config["extra_env_vars"] = {"SRTSLURM_CONFIG": str(tmp_path / "cluster.yaml")}
    config["git_repos"][0]["commit"] = "updated-revision"
    config["git_repos"][0]["url"] = url
    config["git_repos"].append({"url": "https://github.com/example/recipes.git", "commit": "main"})
    if not srt_first:
        config["git_repos"].reverse()
    tdef = SRTSlurmTestDefinition.model_validate(config)
    srt_repo = tdef.git_repos[0 if srt_first else 1]
    other_repo = tdef.git_repos[1 if srt_first else 0]

    assert tdef.srtctl.git_repo is srt_repo
    assert tdef.srtctl.git_repo.commit == "updated-revision"
    assert "updated-revision" in tdef.srtctl.venv_name
    assert tdef.installables == [tdef.srtctl, other_repo]
    assert isinstance(tdef.installables[1], GitRepo)


@pytest.mark.parametrize("git_repos", [None, [], [{"url": "https://github.com/example/recipes.git", "commit": "main"}]])
def test_srtctl_requires_a_configured_git_repo(tmp_path: Path, git_repos, make_tdef):
    config = make_tdef(tmp_path).model_dump()
    if git_repos is None:
        config.pop("git_repos")
    else:
        config["git_repos"] = git_repos

    tdef = SRTSlurmTestDefinition.model_validate(config)
    with pytest.raises(ValueError, match="requires a repository named 'srt-slurm' in git_repos"):
        _ = tdef.installables


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


@pytest.mark.parametrize("models", [[], ["Qwen/Qwen3-0.6B"], ["Qwen/Qwen3-0.6B", "example/another-model"]])
def test_optional_hf_model_installables(tmp_path: Path, make_tdef, models: list[str]):
    config = make_tdef(tmp_path).model_dump()
    config["hf_models"] = models
    tdef = SRTSlurmTestDefinition.model_validate(config)

    assert tdef.installables == [tdef.srtctl, *(HFModel(model_name=name) for name in models)]
    assert tdef.hf_models == models
    tr = TestRun(name="srt", test=tdef, num_nodes=1, nodes=[])
    assert not any("hf_models" in name for name in tr.param_space)


def test_pinned_hf_model_installable(tmp_path: Path, make_tdef):
    config = make_tdef(tmp_path).model_dump()
    config["hf_models"] = [{"model_name": "Qwen/Qwen3-0.6B", "revision": "a" * 40}, "example/unpinned"]
    tdef = SRTSlurmTestDefinition.model_validate(config)
    assert tdef.installables == [
        tdef.srtctl,
        HFModel("Qwen/Qwen3-0.6B", revision="a" * 40),
        HFModel("example/unpinned"),
    ]
    assert SRTSlurmTestDefinition.model_validate(tdef.model_dump()).installables == tdef.installables
