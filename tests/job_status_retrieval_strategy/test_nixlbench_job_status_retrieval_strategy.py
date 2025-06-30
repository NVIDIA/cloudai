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


from cloudai.core import TestRun
from cloudai.workloads.nixl_bench import NIXLBenchCmdArgs, NIXLBenchTestDefinition

LOG_EXTRACT = """
Num threads (--num_threads=N)                               : 1
--------------------------------------------------------------------------------

Block Size (B)      Batch Size     Avg Lat. (us)  B/W (MiB/Sec)  B/W (GiB/Sec)  B/W (GB/Sec)
--------------------------------------------------------------------------------
4096                1              6.36607        613.604        0.599223       0.643411
8192                1              6.36806        1226.83        1.19807
"""


class TestNIXLBenchSuccessCheck:
    def setup_method(self) -> None:
        self.nixl_tdef = NIXLBenchTestDefinition(
            name="n",
            description="d",
            test_template_name="tt",
            cmd_args=NIXLBenchCmdArgs(docker_image_url="", etcd_endpoint="", path_to_benchmark=""),
            etcd_image_url="e",
        )

    def test_no_file(self, base_tr: TestRun) -> None:
        result = self.nixl_tdef.was_run_successful(base_tr)
        assert not result.is_successful
        assert (
            result.error_message
            == f"stdout.txt file not found in the specified output directory {base_tr.output_path}."
        )

    def test_no_header(self, base_tr: TestRun) -> None:
        base_tr.output_path.mkdir(parents=True, exist_ok=True)
        (base_tr.output_path / "stdout.txt").write_text(LOG_EXTRACT.splitlines()[-1])
        result = self.nixl_tdef.was_run_successful(base_tr)
        assert not result.is_successful
        assert result.error_message == f"NIXLBench results table not found in {base_tr.output_path / 'stdout.txt'}."

    def test_no_data(self, base_tr: TestRun) -> None:
        base_tr.output_path.mkdir(parents=True, exist_ok=True)
        (base_tr.output_path / "stdout.txt").write_text("\n".join(LOG_EXTRACT.splitlines()[:-2]))
        result = self.nixl_tdef.was_run_successful(base_tr)
        assert not result.is_successful
        assert result.error_message == f"NIXLBench data not found in {base_tr.output_path / 'stdout.txt'}."

    def test_successfull_job(self, base_tr: TestRun) -> None:
        base_tr.output_path.mkdir(parents=True, exist_ok=True)
        (base_tr.output_path / "stdout.txt").write_text(LOG_EXTRACT)
        result = self.nixl_tdef.was_run_successful(base_tr)
        assert result.is_successful
        assert result.error_message == ""
