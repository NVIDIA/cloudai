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

from cloudai.core import TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.srt_slurm import SRTSlurmCommandGenStrategy


def test_single_sbatch_rejected(slurm_system: SlurmSystem, tmp_path: Path, make_tdef):
    tdef = make_tdef(tmp_path)
    tr = TestRun(name="srt", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path / "output")
    with pytest.raises(ValueError, match="single-sbatch"):
        SRTSlurmCommandGenStrategy(slurm_system, tr).gen_srun_command()


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
