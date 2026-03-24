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

from typing import cast

import pytest

from cloudai.core import TestRun
from cloudai.workloads.nixl_ep import NixlEPCmdArgs, NixlEPTestDefinition

EXPANSION_CONTRACTION_PLAN = (
    "[[0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, -6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]"
)
SUCCESSFUL_BANDWIDTH_LINE = (
    "[rank 0] Dispatch + combine bandwidth: 12.34 GB/s, avg_t=56.7 us, min_t=50.0 us, max_t=60.0 us\n"
)
TCPSTORE_TIMEOUT_LINE = (
    "recvValueWithTimeout failed on SocketImpl(fd=65, addr=[pool0-01876]:11088, remote=[pool0-01873.cm.cluster]:9999)\n"
)


def num_nodes(test_run: TestRun) -> int:
    return cast(int, test_run.num_nodes)


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
        for node_idx in range(num_nodes(nixl_ep_tr)):
            (nixl_ep_tr.output_path / f"nixl-ep-node-{node_idx}.log").write_text(
                SUCCESSFUL_BANDWIDTH_LINE, encoding="utf-8"
            )
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
        (nixl_ep_tr.output_path / "nixl-ep-node-0.log").write_text(SUCCESSFUL_BANDWIDTH_LINE, encoding="utf-8")

        result = nixl_ep_tr.test.was_run_successful(nixl_ep_tr)

        assert not result.is_successful
        assert "nixl-ep-node-1.log, nixl-ep-node-2.log" in result.error_message

    def test_plan_mismatch_is_reported(self, nixl_ep_tr: TestRun) -> None:
        nixl_ep_tr.output_path.mkdir(parents=True, exist_ok=True)
        for node_idx in range(num_nodes(nixl_ep_tr)):
            (nixl_ep_tr.output_path / f"nixl-ep-node-{node_idx}.log").write_text(
                SUCCESSFUL_BANDWIDTH_LINE, encoding="utf-8"
            )
        (nixl_ep_tr.output_path / "nixl-ep-node-1.log").write_text(
            "Process 0 -> no plan phases were found for rank 9 after phase None, exiting\n",
            encoding="utf-8",
        )
        (nixl_ep_tr.output_path / "slurm-job.toml").write_text(
            'state = "COMPLETED"\nexit_code = "0:0"\n',
            encoding="utf-8",
        )

        result = nixl_ep_tr.test.was_run_successful(nixl_ep_tr)

        assert not result.is_successful
        assert "never appears in the plan" in result.error_message

    def test_tcpstore_timeout_is_reported(self, nixl_ep_tr: TestRun) -> None:
        nixl_ep_tr.output_path.mkdir(parents=True, exist_ok=True)
        for node_idx in range(num_nodes(nixl_ep_tr)):
            (nixl_ep_tr.output_path / f"nixl-ep-node-{node_idx}.log").write_text(
                SUCCESSFUL_BANDWIDTH_LINE, encoding="utf-8"
            )
        (nixl_ep_tr.output_path / "nixl-ep-node-2.log").write_text(
            TCPSTORE_TIMEOUT_LINE,
            encoding="utf-8",
        )
        (nixl_ep_tr.output_path / "slurm-job.toml").write_text(
            'state = "COMPLETED"\nexit_code = "0:0"\n',
            encoding="utf-8",
        )

        result = nixl_ep_tr.test.was_run_successful(nixl_ep_tr)

        assert not result.is_successful
        assert "lost its TCPStore connection" in result.error_message

    def test_primary_launch_exit_before_phase_completion_is_reported(self, nixl_ep_tr: TestRun) -> None:
        nixl_ep_tr.output_path.mkdir(parents=True, exist_ok=True)
        for node_idx in range(num_nodes(nixl_ep_tr)):
            (nixl_ep_tr.output_path / f"nixl-ep-node-{node_idx}.log").write_text(
                SUCCESSFUL_BANDWIDTH_LINE, encoding="utf-8"
            )
        (nixl_ep_tr.output_path / "nixl-ep-node-1.log").write_text(
            "Primary NIXL EP launch exited before phase 1 completed\n",
            encoding="utf-8",
        )
        (nixl_ep_tr.output_path / "slurm-job.toml").write_text(
            'state = "COMPLETED"\nexit_code = "0:0"\n',
            encoding="utf-8",
        )

        result = nixl_ep_tr.test.was_run_successful(nixl_ep_tr)

        assert not result.is_successful
        assert "primary NIXL EP launch exited before phase 1 completed" in result.error_message

    def test_initial_primary_launch_exit_explains_missing_later_node_logs(self, nixl_ep_tr: TestRun) -> None:
        nixl_ep_tr.output_path.mkdir(parents=True, exist_ok=True)
        (nixl_ep_tr.output_path / "nixl-ep-node-0.log").write_text(
            "global_rank=0, local_rank=0 -> start phase 0\n",
            encoding="utf-8",
        )
        (nixl_ep_tr.output_path / "stdout.txt").write_text(
            "Primary NIXL EP launch exited before phase 0 completed\n",
            encoding="utf-8",
        )

        result = nixl_ep_tr.test.was_run_successful(nixl_ep_tr)

        assert not result.is_successful
        assert "initial NIXL EP launch exited before phase 0 completed" in result.error_message
        assert "later stage launches never started" in result.error_message
        assert "some node logs may be absent" in result.error_message

    def test_ucx_remote_memory_view_failure_is_reported(self, nixl_ep_tr: TestRun) -> None:
        nixl_ep_tr.output_path.mkdir(parents=True, exist_ok=True)
        for node_idx in range(num_nodes(nixl_ep_tr)):
            (nixl_ep_tr.output_path / f"nixl-ep-node-{node_idx}.log").write_text(
                SUCCESSFUL_BANDWIDTH_LINE, encoding="utf-8"
            )
        (nixl_ep_tr.output_path / "nixl-ep-node-0.log").write_text(
            "E0319 04:13:25.442619  950677 ucx_backend.cpp:1486] "
            "Failed to prepare remote memory view: Failed to create device memory list(remote): No such device\n",
            encoding="utf-8",
        )
        (nixl_ep_tr.output_path / "slurm-job.toml").write_text(
            'state = "COMPLETED"\nexit_code = "0:0"\n',
            encoding="utf-8",
        )

        result = nixl_ep_tr.test.was_run_successful(nixl_ep_tr)

        assert not result.is_successful
        assert "failed to initialize its UCX remote memory view" in result.error_message

    @pytest.mark.parametrize(
        "log_content, expected_fragment",
        [
            (
                "Traceback (most recent call last):\n  File elastic.py, line 42\nRuntimeError: boom\n",
                "Python traceback",
            ),
            (
                "Timed out waiting for NIXL EP master services on 192.168.1.1\n",
                "master services never became ready",
            ),
            (
                "recvValueWithTimeout timed out after 300000ms\n",
                "worker timed out",
            ),
            (
                "srun: error: Unable to allocate resources: Invalid node name\n",
                "srun failure",
            ),
            (
                "node001: task 0: Exited with exit code 1\n",
                "non-zero status",
            ),
        ],
    )
    def test_launcher_failure_patterns_are_reported(
        self, nixl_ep_tr: TestRun, log_content: str, expected_fragment: str
    ) -> None:
        nixl_ep_tr.output_path.mkdir(parents=True, exist_ok=True)
        (nixl_ep_tr.output_path / "nixl-ep-node-0.log").write_text(log_content, encoding="utf-8")

        result = nixl_ep_tr.test.was_run_successful(nixl_ep_tr)

        assert not result.is_successful
        assert expected_fragment in result.error_message

    def test_completed_job_without_benchmark_output_is_reported(self, nixl_ep_tr: TestRun) -> None:
        nixl_ep_tr.output_path.mkdir(parents=True, exist_ok=True)
        for node_idx in range(num_nodes(nixl_ep_tr)):
            (nixl_ep_tr.output_path / f"nixl-ep-node-{node_idx}.log").write_text("run completed\n", encoding="utf-8")
        (nixl_ep_tr.output_path / "slurm-job.toml").write_text(
            'state = "COMPLETED"\nexit_code = "0:0"\n',
            encoding="utf-8",
        )

        result = nixl_ep_tr.test.was_run_successful(nixl_ep_tr)

        assert not result.is_successful
        assert "no benchmark summary lines were found" in result.error_message
