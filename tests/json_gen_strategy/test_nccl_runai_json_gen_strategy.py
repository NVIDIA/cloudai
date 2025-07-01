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

from unittest.mock import Mock

import pytest

from cloudai.core import Test, TestRun
from cloudai.systems.runai.runai_system import RunAISystem
from cloudai.workloads.nccl_test import NCCLCmdArgs, NCCLTestDefinition, NcclTestRunAIJsonGenStrategy


class TestNcclTestRunAIJsonGenStrategy:
    @pytest.fixture
    def json_gen_strategy(self, runai_system: RunAISystem) -> NcclTestRunAIJsonGenStrategy:
        return NcclTestRunAIJsonGenStrategy(runai_system)

    def test_gen_json(self, json_gen_strategy: NcclTestRunAIJsonGenStrategy) -> None:
        cmd_args = NCCLCmdArgs.model_validate({"subtest_name": "all_reduce_perf", "docker_image_url": "fake_image_url"})
        nccl = NCCLTestDefinition(name="name", description="desc", test_template_name="tt", cmd_args=cmd_args)
        t = Test(test_definition=nccl, test_template=Mock())
        tr = TestRun(name="t1", test=t, nodes=["node1", "node2"], num_nodes=2)

        json_payload = json_gen_strategy.gen_json(tr)

        assert json_payload["projectId"] == "test_project_id"
        assert json_payload["clusterId"] == "test_cluster_id"
        assert json_payload["spec"]["command"] == "all_reduce_perf"
        assert json_payload["spec"]["image"] == "fake_image_url"
        assert json_payload["spec"]["parallelism"] == 2
        assert json_payload["spec"]["completions"] == 2

    def test_gen_json_with_cmd_args(self, json_gen_strategy: NcclTestRunAIJsonGenStrategy) -> None:
        cmd_args = NCCLCmdArgs.model_validate(
            {
                "subtest_name": "all_reduce_perf",
                "docker_image_url": "fake_image_url",
                "nthreads": "4",
                "ngpus": "2",
                "minbytes": "32M",
                "maxbytes": "64M",
                "stepbytes": "1M",
                "op": "sum",
                "datatype": "float",
                "root": "0",
                "iters": "20",
                "warmup_iters": "5",
                "agg_iters": "1",
                "average": "1",
                "parallel_init": "0",
                "check": "1",
                "blocking": "0",
                "cudagraph": "0",
            }
        )
        nccl = NCCLTestDefinition(name="name", description="desc", test_template_name="tt", cmd_args=cmd_args)
        t = Test(test_definition=nccl, test_template=Mock())
        tr = TestRun(name="t1", test=t, nodes=["node1", "node2"], num_nodes=2)

        json_payload = json_gen_strategy.gen_json(tr)

        assert json_payload["projectId"] == "test_project_id"
        assert json_payload["clusterId"] == "test_cluster_id"
        assert json_payload["spec"]["command"] == "all_reduce_perf"

        expected_args = [
            "--nthreads 4",
            "--ngpus 2",
            "--minbytes 32M",
            "--maxbytes 64M",
            "--stepbytes 1M",
            "--op sum",
            "--datatype float",
            "--root 0",
            "--iters 20",
            "--warmup_iters 5",
            "--agg_iters 1",
            "--average 1",
            "--parallel_init 0",
            "--check 1",
            "--blocking 0",
            "--cudagraph 0",
        ]

        for arg in expected_args:
            assert arg in json_payload["spec"]["args"]

        assert json_payload["spec"]["image"] == "fake_image_url"
        assert json_payload["spec"]["parallelism"] == 2
        assert json_payload["spec"]["completions"] == 2

    def test_gen_json_with_extra_cmd_args(self, json_gen_strategy: NcclTestRunAIJsonGenStrategy) -> None:
        cmd_args = NCCLCmdArgs.model_validate({"subtest_name": "all_reduce_perf", "docker_image_url": "fake_image_url"})
        nccl = NCCLTestDefinition(
            name="name",
            description="desc",
            test_template_name="tt",
            cmd_args=cmd_args,
            extra_cmd_args={"--extra-arg": "value"},
        )
        t = Test(test_definition=nccl, test_template=Mock())
        tr = TestRun(name="t1", test=t, nodes=["node1", "node2"], num_nodes=2)

        json_payload = json_gen_strategy.gen_json(tr)

        assert json_payload["projectId"] == "test_project_id"
        assert json_payload["clusterId"] == "test_cluster_id"
        assert json_payload["spec"]["command"] == "all_reduce_perf"
        assert "--extra-arg value" in json_payload["spec"]["args"]
        assert json_payload["spec"]["image"] == "fake_image_url"
        assert json_payload["spec"]["parallelism"] == 2
        assert json_payload["spec"]["completions"] == 2
