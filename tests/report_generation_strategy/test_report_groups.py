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

import copy
from pathlib import Path

import pandas as pd

from cloudai.core import Test, TestRun, TestTemplate
from cloudai.report_generator.groups import TestRunsGrouper
from cloudai.report_generator.util import diff_test_runs
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.workloads.nemo_run import Data, NeMoRunCmdArgs, NeMoRunTestDefinition


class TestGrouping:
    def _mock_extract_data(self, tr: TestRun) -> pd.DataFrame:
        return pd.DataFrame()

    def test_single_tr(self, nccl_tr: TestRun) -> None:
        groups = TestRunsGrouper(items=[nccl_tr], group_by=[]).groups()
        assert len(groups) == 1
        assert groups[0].name == "all-in-one"
        assert groups[0].items[0].name == "0.0"

    def test_multiple_trs_no_group_by_fields_same_trs(self, nccl_tr: TestRun) -> None:
        groups = TestRunsGrouper(items=[nccl_tr, nccl_tr], group_by=[]).groups()
        assert len(groups) == 1
        assert groups[0].name == "all-in-one"
        assert groups[0].items[0].name == "0.0"
        assert groups[0].items[1].name == "0.1"

    def test_multiple_trs_no_group_by_fields(self, nccl_tr: TestRun) -> None:
        nccl1 = copy.deepcopy(nccl_tr)
        nccl2 = copy.deepcopy(nccl_tr)
        nccl1.test.test_definition.cmd_args.subtest_name = "all_gather_perf"
        nccl2.test.test_definition.cmd_args.subtest_name = "all_reduce_perf"
        groups = TestRunsGrouper(items=[nccl1, nccl2], group_by=[]).groups()
        assert len(groups) == 1
        assert groups[0].name == "all-in-one"
        assert groups[0].items[0].name == "subtest_name=all_gather_perf"
        assert groups[0].items[1].name == "subtest_name=all_reduce_perf"

    def test_group_by_one_field(self, nccl_tr: TestRun) -> None:
        nccl1 = copy.deepcopy(nccl_tr)
        nccl2 = copy.deepcopy(nccl_tr)
        nccl1.test.test_definition.cmd_args.subtest_name = "all_gather_perf"
        nccl2.test.test_definition.cmd_args.subtest_name = "all_reduce_perf"

        groups = TestRunsGrouper(items=[nccl1, nccl2], group_by=["subtest_name"]).groups()

        assert len(groups) == 2
        assert groups[0].name == "subtest_name=all_gather_perf"
        assert groups[1].name == "subtest_name=all_reduce_perf"
        assert groups[0].items[0].name == "0.0"
        assert groups[1].items[0].name == "1.0"

    def test_group_by_two_fields(self, nccl_tr: TestRun) -> None:
        nccl_tr.test.test_definition.cmd_args.subtest_name = ["all_gather_perf", "all_reduce_perf"]
        nccl_tr.test.test_definition.extra_env_vars["NCCL_IB_SPLIT_DATA_ON_QPS"] = ["0", "1"]
        trs: list[TestRun] = [nccl_tr.apply_params_set(combination) for combination in nccl_tr.all_combinations]

        groups = TestRunsGrouper(
            items=trs,
            group_by=["subtest_name", "extra_env_vars.NCCL_IB_SPLIT_DATA_ON_QPS"],
        ).groups()

        assert len(groups) == 4
        assert all(len(group.items) == 1 for group in groups)
        assert groups[0].name == "subtest_name=all_gather_perf NCCL_IB_SPLIT_DATA_ON_QPS=0"
        assert groups[1].name == "subtest_name=all_gather_perf NCCL_IB_SPLIT_DATA_ON_QPS=1"
        assert groups[2].name == "subtest_name=all_reduce_perf NCCL_IB_SPLIT_DATA_ON_QPS=0"
        assert groups[3].name == "subtest_name=all_reduce_perf NCCL_IB_SPLIT_DATA_ON_QPS=1"

    def test_multiple_trs_in_a_group(self, nccl_tr: TestRun) -> None:
        nccl_tr.test.test_definition.cmd_args.subtest_name = ["all_gather_perf", "all_reduce_perf"]
        nccl_tr.test.test_definition.extra_env_vars["NCCL_IB_SPLIT_DATA_ON_QPS"] = ["0", "1"]
        trs: list[TestRun] = [nccl_tr.apply_params_set(combination) for combination in nccl_tr.all_combinations]

        groups = TestRunsGrouper(items=trs, group_by=["subtest_name"]).groups()

        assert len(groups) == 2

        assert groups[0].name == "subtest_name=all_gather_perf"
        assert len(groups[0].items) == 2
        assert groups[0].items[0].name == "NCCL_IB_SPLIT_DATA_ON_QPS=0"
        assert groups[0].items[1].name == "NCCL_IB_SPLIT_DATA_ON_QPS=1"

        assert groups[1].name == "subtest_name=all_reduce_perf"
        assert len(groups[1].items) == 2
        assert groups[1].items[0].name == "NCCL_IB_SPLIT_DATA_ON_QPS=0"
        assert groups[1].items[1].name == "NCCL_IB_SPLIT_DATA_ON_QPS=1"

    def test_missing_group_by_field(self, nccl_tr: TestRun) -> None:
        nccl_tr1 = copy.deepcopy(nccl_tr)
        nccl_tr2 = copy.deepcopy(nccl_tr)
        nccl_tr1.test.test_definition.cmd_args.unknown_field = "value"

        groups = TestRunsGrouper(items=[nccl_tr1, nccl_tr2], group_by=["unknown_field"]).groups()
        assert len(groups) == 2
        assert groups[0].name == "unknown_field=value"
        assert groups[1].name == "unknown_field="


class TestDiffTrs:
    def test_diff_cmd_args_field(self, nccl_tr: TestRun) -> None:
        nccl1 = copy.deepcopy(nccl_tr)
        nccl2 = copy.deepcopy(nccl_tr)
        nccl1.test.test_definition.cmd_args.subtest_name = "all_gather_perf"
        nccl2.test.test_definition.cmd_args.subtest_name = "all_reduce_perf"

        diff = diff_test_runs([nccl1, nccl2])

        assert diff == {"subtest_name": ["all_gather_perf", "all_reduce_perf"]}

    def test_diff_num_nodes(self, nccl_tr: TestRun) -> None:
        nccl1 = copy.deepcopy(nccl_tr)
        nccl2 = copy.deepcopy(nccl_tr)
        nccl1.num_nodes = 1
        nccl2.num_nodes = 2

        diff = diff_test_runs([nccl1, nccl2])
        assert diff == {"NUM_NODES": [1, 2]}

    def test_diff_extra_env_vars(self, nccl_tr: TestRun) -> None:
        nccl1 = copy.deepcopy(nccl_tr)
        nccl2 = copy.deepcopy(nccl_tr)
        nccl1.test.test_definition.extra_env_vars["NCCL_IB_SPLIT_DATA_ON_QPS"] = "0"
        nccl2.test.test_definition.extra_env_vars["NCCL_IB_SPLIT_DATA_ON_QPS"] = "1"

        diff = diff_test_runs([nccl1, nccl2])
        assert diff == {"extra_env_vars.NCCL_IB_SPLIT_DATA_ON_QPS": ["0", "1"]}

    def test_diff_nested(self, slurm_system: SlurmSystem) -> None:
        data1 = Data(micro_batch_size=1, global_batch_size=1)
        data2 = Data(micro_batch_size=2, global_batch_size=2)
        nemo1 = TestRun(
            name="nemo1",
            num_nodes=1,
            nodes=[],
            output_path=Path("/"),
            test=Test(
                test_definition=NeMoRunTestDefinition(
                    name="nemo1",
                    description="desc",
                    test_template_name="t",
                    cmd_args=NeMoRunCmdArgs(
                        docker_image_url="fake://url/nemo1", task="task1", recipe_name="recipe1", data=data1
                    ),
                ),
                test_template=TestTemplate(system=slurm_system),
            ),
        )
        nemo2 = TestRun(
            name="nemo1",
            num_nodes=1,
            nodes=[],
            output_path=Path("/"),
            test=Test(
                test_definition=NeMoRunTestDefinition(
                    name="nemo2",
                    description="desc",
                    test_template_name="t",
                    cmd_args=NeMoRunCmdArgs(
                        docker_image_url="fake://url/nemo1", task="task1", recipe_name="recipe1", data=data2
                    ),
                ),
                test_template=TestTemplate(system=slurm_system),
            ),
        )
        diff = diff_test_runs([nemo1, nemo2])
        assert diff == {"data.micro_batch_size": [1, 2], "data.global_batch_size": [1, 2]}
