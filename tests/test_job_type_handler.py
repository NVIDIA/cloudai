# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from cloudai.core import TestRun
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.workloads.nccl_test import NCCLCmdArgs, NCCLTestDefinition


@pytest.fixture
def tr(slurm_system: SlurmSystem) -> TestRun:
    tdef = NCCLTestDefinition(
        name="nccl",
        description="NCCL Test",
        test_template_name="NcclTest",
        cmd_args=NCCLCmdArgs(docker_image_url="fake://url/nccl"),
    )
    return TestRun(name="test_run", test=tdef, num_nodes=1, nodes=[], output_path=slurm_system.output_path)


def test_is_dse_job_non_dse(tr: TestRun):
    assert tr.is_dse_job is False


def test_is_dse_job_dse_args(tr: TestRun):
    tr.test.cmd_args.nthreads = [1, 2]
    tr.test.extra_env_vars = {"VAR1": "singular"}
    assert tr.is_dse_job is True


def test_is_dse_job_dse_env_vars(tr: TestRun):
    tr.test.extra_env_vars = {"VAR1": ["list-item1", "list-item2"], "VAR2": "singular3"}
    assert tr.is_dse_job is True


def test_is_dse_job_num_nodes(tr: TestRun):
    tr.num_nodes = [1, 2]
    assert tr.is_dse_job is True
