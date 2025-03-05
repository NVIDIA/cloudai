# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from cloudai.workloads.jax_toolbox import JaxToolboxJobStatusRetrievalStrategy


class TestJaxToolboxJobStatusRetrievalStrategy:
    def setup_method(self) -> None:
        self.js = JaxToolboxJobStatusRetrievalStrategy()

    def test_no_profile_stderr_file(self, tmp_path: Path) -> None:
        result = self.js.get_job_status(tmp_path)
        assert not result.is_successful
        assert result.error_message == (
            f"No profile_stderr_*.txt files found in the specified output directory, {tmp_path}. "
            "These files are expected to be created during the profiling stage. "
            "Please ensure the profiling stage completed successfully. "
            "Run the generated sbatch script manually to debug."
        )

    def test_missing_pax_status_keyword(self, tmp_path: Path) -> None:
        profile_stderr_file = tmp_path / "profile_stderr_1.txt"
        profile_stderr_content = "Some initialization output\nMore output\nFinal output without the expected keyword"
        profile_stderr_file.write_text(profile_stderr_content)
        result = self.js.get_job_status(tmp_path)
        assert not result.is_successful
        assert result.error_message == (
            "The profiling stage completed but did not generate the expected '[PAX STATUS]: E2E time: "
            "Elapsed time for ' keyword in any of the profile_stderr_*.txt files. "
            "There are two stages in the Grok run, and an error occurred in the profiling stage. "
            "While profile_stderr_*.txt files were created, the expected keyword is missing. "
            "You need to run the sbatch script manually to see what happens."
        )

    def test_no_error_files(self, tmp_path: Path) -> None:
        profile_stderr_file = tmp_path / "profile_stderr.txt"
        profile_stderr_content = "[PAX STATUS]: E2E time: Elapsed time for profiling"
        profile_stderr_file.write_text(profile_stderr_content)
        result = self.js.get_job_status(tmp_path)
        assert not result.is_successful
        assert result.error_message == (
            f"No profile_stderr_*.txt files found in the specified output directory, {tmp_path}. "
            "These files are expected to be created during the profiling stage. "
            "Please ensure the profiling stage completed successfully. "
            "Run the generated sbatch script manually to debug."
        )

    def test_cuda_no_device_error_in_profile_stderr(self, tmp_path: Path) -> None:
        profile_stderr_file = tmp_path / "profile_stderr_1.txt"
        profile_stderr_content = "[PAX STATUS]: E2E time: Elapsed time for profiling\n"
        profile_stderr_content += "CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected"
        profile_stderr_file.write_text(profile_stderr_content)

        result = self.js.get_job_status(tmp_path)
        assert not result.is_successful
        assert result.error_message == (
            "CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected. This may be due to missing "
            "environment variables, specifically but not limited to CUDA_VISIBLE_DEVICES. "
            "First, check if GPUs are available on the server. "
            "Second, if running the job with Slurm, ensure proper resource-related options are set, "
            "including GPU resource requirements. Lastly, check environment variables. "
            "If the problem persists, verify commands and environment variables by running a simple GPU-only "
            "example command."
        )

    def test_missing_e2e_time_keyword(self, tmp_path: Path) -> None:
        profile_stderr_file = tmp_path / "profile_stderr_1.txt"
        profile_stderr_content = "[PAX STATUS]: E2E time: Elapsed time for profiling"
        profile_stderr_file.write_text(profile_stderr_content)

        error_file = tmp_path / "error-1.txt"
        error_content = "Some initialization output\nMore output\nFinal output without the expected keyword"
        error_file.write_text(error_content)

        result = self.js.get_job_status(tmp_path)
        assert not result.is_successful
        assert result.error_message == (
            f"The file {error_file} does not contain the expected 'E2E time: Elapsed time for' keyword at the "
            "end. This indicates the actual run did not complete successfully. "
            "Please debug this manually to ensure the actual run stage completes as expected."
        )

    def test_cuda_no_device_error_in_error_file(self, tmp_path: Path) -> None:
        profile_stderr_file = tmp_path / "profile_stderr_1.txt"
        profile_stderr_content = "[PAX STATUS]: E2E time: Elapsed time for profiling"
        profile_stderr_file.write_text(profile_stderr_content)

        error_file = tmp_path / "error-1.txt"
        error_content = "CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected"
        error_file.write_text(error_content)

        result = self.js.get_job_status(tmp_path)
        assert not result.is_successful
        assert result.error_message == (
            "CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected. This may be due to missing "
            "environment variables, specifically but not limited to CUDA_VISIBLE_DEVICES. "
            "First, check if GPUs are available on the server. "
            "Second, if running the job with Slurm, ensure proper resource-related options are set, "
            "including GPU resource requirements. Lastly, check environment variables. "
            "If the problem persists, verify commands and environment variables by running a simple GPU-only "
            "example command."
        )

    def test_successful_job(self, tmp_path: Path) -> None:
        profile_stderr_file = tmp_path / "profile_stderr_1.txt"
        profile_stderr_content = "[PAX STATUS]: E2E time: Elapsed time for profiling"
        profile_stderr_file.write_text(profile_stderr_content)

        error_file = tmp_path / "error-1.txt"
        error_content = "Some initialization output\nMore output\nE2E time: Elapsed time for actual run\nFinal output"
        error_file.write_text(error_content)

        result = self.js.get_job_status(tmp_path)
        assert result.is_successful
        assert result.error_message == ""

    def test_nccl_group_end_error_in_profile_stderr(self, tmp_path: Path) -> None:
        profile_stderr_file = tmp_path / "profile_stderr_1.txt"
        profile_stderr_content = "[PAX STATUS]: E2E time: Elapsed time for profiling\n"
        profile_stderr_content += "NCCL operation ncclGroupEnd() failed: unhandled system error"
        profile_stderr_file.write_text(profile_stderr_content)

        result = self.js.get_job_status(tmp_path)
        assert not result.is_successful
        assert result.error_message == (
            "NCCL operation ncclGroupEnd() failed: unhandled system error. Please check if the NCCL-test "
            "passes. Run with NCCL_DEBUG=INFO for more details."
        )

    def test_nccl_group_end_error_in_error_file(self, tmp_path: Path) -> None:
        profile_stderr_file = tmp_path / "profile_stderr_1.txt"
        profile_stderr_content = "[PAX STATUS]: E2E time: Elapsed time for profiling"
        profile_stderr_file.write_text(profile_stderr_content)

        error_file = tmp_path / "error-1.txt"
        error_content = "NCCL operation ncclGroupEnd() failed: unhandled system error"
        error_file.write_text(error_content)

        result = self.js.get_job_status(tmp_path)
        assert not result.is_successful
        assert result.error_message == (
            "NCCL operation ncclGroupEnd() failed: unhandled system error. Please check if the NCCL-test "
            "passes. Run with NCCL_DEBUG=INFO for more details."
        )

    def test_heartbeat_error_in_profile_stderr(self, tmp_path: Path) -> None:
        profile_stderr_file = tmp_path / "profile_stderr_1.txt"
        profile_stderr_content = "[PAX STATUS]: E2E time: Elapsed time for profiling\n"
        profile_stderr_content += "Terminating process because the coordinator detected missing heartbeats"
        profile_stderr_file.write_text(profile_stderr_content)

        result = self.js.get_job_status(tmp_path)
        assert not result.is_successful
        assert result.error_message == (
            "Terminating process because the coordinator detected missing heartbeats. This most likely "
            f"indicates that another task died. Please review the file at {profile_stderr_file} and any relevant "
            f"logs in {tmp_path}. Ensure the servers allocated for this task can reach each other with their "
            "hostnames, and they can open any ports and reach others' ports."
        )

    def test_heartbeat_error_in_error_file(self, tmp_path: Path) -> None:
        profile_stderr_file = tmp_path / "profile_stderr_1.txt"
        profile_stderr_content = "[PAX STATUS]: E2E time: Elapsed time for profiling"
        profile_stderr_file.write_text(profile_stderr_content)

        error_file = tmp_path / "error-1.txt"
        error_content = "Terminating process because the coordinator detected missing heartbeats"
        error_file.write_text(error_content)

        result = self.js.get_job_status(tmp_path)
        assert not result.is_successful
        assert result.error_message == (
            "Terminating process because the coordinator detected missing heartbeats. This most likely "
            f"indicates that another task died. Please review the file at {error_file} and any relevant logs in"
            f" {tmp_path}. Ensure the servers allocated for this task can reach each other with their "
            "hostnames, and they can open any ports and reach others' ports."
        )

    def test_pyxis_mktemp_error_in_profile_stderr(self, tmp_path: Path) -> None:
        profile_stderr_file = tmp_path / "profile_stderr_1.txt"
        profile_stderr_content = "[PAX STATUS]: E2E time: Elapsed time for profiling\n"
        profile_stderr_content += (
            "pyxis:     mktemp: failed to create directory via template '/tmp/enroot.XXXXXXXXXX': "
            "No space left on device"
        )
        profile_stderr_file.write_text(profile_stderr_content)

        result = self.js.get_job_status(tmp_path)
        assert not result.is_successful
        assert result.error_message == (
            "pyxis: mktemp: failed to create directory via template. This is due to insufficient disk cache "
            "capacity. This is not a CloudAI issue. When you run JaxToolbox, CloudAI executes srun, which "
            "includes the container image option. When the container image argument is a remote URL, "
            "Slurm downloads and caches the Docker image locally. It fails with this error when the system "
            "does not have enough disk capacity to cache the Docker image."
        )

    def test_pyxis_mktemp_error_in_error_file(self, tmp_path: Path) -> None:
        profile_stderr_file = tmp_path / "profile_stderr_1.txt"
        profile_stderr_content = "[PAX STATUS]: E2E time: Elapsed time for profiling"
        profile_stderr_file.write_text(profile_stderr_content)

        error_file = tmp_path / "error-1.txt"
        error_content = (
            "pyxis:     mktemp: failed to create directory via template '/tmp/enroot.XXXXXXXXXX': "
            "No space left on device"
        )
        error_file.write_text(error_content)

        result = self.js.get_job_status(tmp_path)
        assert not result.is_successful
        assert result.error_message == (
            "pyxis: mktemp: failed to create directory via template. This is due to insufficient disk cache "
            "capacity. This is not a CloudAI issue. When you run JaxToolbox, CloudAI executes srun, which "
            "includes the container image option. When the container image argument is a remote URL, "
            "Slurm downloads and caches the Docker image locally. It fails with this error when the system "
            "does not have enough disk capacity to cache the Docker image."
        )
