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
from collections.abc import Callable
from pathlib import Path

import pytest

from cloudai.core import TestRun
from cloudai.workloads.srt_slurm import SRTSlurmTestDefinition


def _make_tdef(tmp_path: Path) -> SRTSlurmTestDefinition:
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


def _write_job_metadata(tdef: SRTSlurmTestDefinition, tr: TestRun, **resources) -> Path:
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


@pytest.fixture(name="make_tdef")
def fixture_make_tdef() -> Callable[[Path], SRTSlurmTestDefinition]:
    return _make_tdef


@pytest.fixture(name="write_job_metadata")
def fixture_write_job_metadata() -> Callable[..., Path]:
    return _write_job_metadata
