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

from textwrap import dedent

import pytest

from cloudai.core import TestRun
from cloudai.workloads.nixl_ep import NixlEPCmdArgs, NixlEPTestDefinition

EXPANSION_CONTRACTION_PLAN = (
    "[[0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7], "
    "[0, 1, 2, 3, 4, -6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]"
)


@pytest.fixture
def nixl_ep_tr(tmp_path) -> TestRun:
    tdef = NixlEPTestDefinition(
        name="nixl_ep",
        description="NIXL Elastic EP benchmark",
        test_template_name="NixlEP",
        cmd_args=NixlEPCmdArgs(
            docker_image_url="docker.io/nvidia/nixl-ep:latest",
            elastic_script="/workspace/nixl/examples/device/ep/tests/elastic/elastic.py",
            plan=EXPANSION_CONTRACTION_PLAN,
            num_processes_per_node=4,
        ),
    )
    return TestRun(name="nixl-ep", test=tdef, num_nodes=3, nodes=[], output_path=tmp_path / "output")


class TestNixlEPStatusCheck:
    def test_successful_job(self, nixl_ep_tr: TestRun) -> None:
        nixl_ep_tr.output_path.mkdir(parents=True, exist_ok=True)
        for node_idx in range(nixl_ep_tr.num_nodes):
            (nixl_ep_tr.output_path / f"nixl-ep-node-{node_idx}.log").write_text("run completed\n", encoding="utf-8")
        (nixl_ep_tr.output_path / "slurm-job.toml").write_text(
            'state = "COMPLETED"\nexit_code = "0:0"\n',
            encoding="utf-8",
        )

        result = nixl_ep_tr.test.was_run_successful(nixl_ep_tr)

        assert result.is_successful
        assert result.error_message == ""

    def test_launcher_path_error_is_reported(self, nixl_ep_tr: TestRun) -> None:
        nixl_ep_tr.output_path.mkdir(parents=True, exist_ok=True)
        (nixl_ep_tr.output_path / "nixl-ep-node-0.log").write_text(
            (
                "python3: can't open file '/workspace/nixl/tests/elastic/elastic.py': "
                "[Errno 2] No such file or directory\n"
            ),
            encoding="utf-8",
        )

        result = nixl_ep_tr.test.was_run_successful(nixl_ep_tr)

        assert not result.is_successful
        assert "entrypoint could not be opened" in result.error_message
        assert "nixl-ep-node-0.log" in result.error_message

    def test_missing_node_logs_is_reported(self, nixl_ep_tr: TestRun) -> None:
        nixl_ep_tr.output_path.mkdir(parents=True, exist_ok=True)
        (nixl_ep_tr.output_path / "nixl-ep-node-0.log").write_text("run completed\n", encoding="utf-8")

        result = nixl_ep_tr.test.was_run_successful(nixl_ep_tr)

        assert not result.is_successful
        assert "nixl-ep-node-1.log, nixl-ep-node-2.log" in result.error_message

    def test_failed_slurm_job_status_is_reported(self, nixl_ep_tr: TestRun) -> None:
        nixl_ep_tr.output_path.mkdir(parents=True, exist_ok=True)
        for node_idx in range(nixl_ep_tr.num_nodes):
            (nixl_ep_tr.output_path / f"nixl-ep-node-{node_idx}.log").write_text("run completed\n", encoding="utf-8")
        (nixl_ep_tr.output_path / "slurm-job.toml").write_text(
            dedent(
                """
                state = "FAILED"
                exit_code = "1:0"

                [[job_steps]]
                step_id = "3"
                name = "bash"
                state = "FAILED"
                exit_code = "2:0"
                submit_line = "srun bash -c python3 /workspace/nixl/examples/device/ep/tests/elastic/elastic.py"
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        result = nixl_ep_tr.test.was_run_successful(nixl_ep_tr)

        assert not result.is_successful
        assert "state=FAILED" in result.error_message
        assert "Last failing step: 3 (bash), exit_code=2:0." in result.error_message
