# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from cloudai.schema.test_template.jax_toolbox.job_status_retrieval_strategy import JaxToolboxJobStatusRetrievalStrategy
from cloudai.schema.test_template.nccl_test.job_status_retrieval_strategy import NcclTestJobStatusRetrievalStrategy


class TestNcclTestJobStatusRetrievalStrategy:
    """Tests for the NcclTestJobStatusRetrievalStrategy class."""

    def setup_method(self) -> None:
        """Setup method for initializing NcclTestJobStatusRetrievalStrategy."""
        self.js = NcclTestJobStatusRetrievalStrategy()

    def test_no_stdout_file(self, tmp_path: Path) -> None:
        """Test that job status is False when no stdout.txt file is present."""
        result = self.js.get_job_status(str(tmp_path))
        assert not result.is_successful
        assert result.error_message == (
            f"stdout.txt file not found in the specified output directory {tmp_path}. "
            "This file is expected to be created as a result of the NCCL test run. "
            "Please ensure the NCCL test was executed properly and that stdout.txt is generated. "
            f"You can run the generated NCCL test command manually and verify the creation of "
            f"{tmp_path / 'stdout.txt'}."
        )

    def test_successful_job(self, tmp_path: Path) -> None:
        """Test that job status is True when stdout.txt contains success indicators."""
        stdout_file = tmp_path / "stdout.txt"
        stdout_content = """
        # Some initialization output
        # More output
        # Out of bounds values : 0 OK
        # Avg bus bandwidth    : 100.00
        # Some final output
        """
        stdout_file.write_text(stdout_content)
        result = self.js.get_job_status(str(tmp_path))
        assert result.is_successful
        assert result.error_message == ""

    def test_failed_job(self, tmp_path: Path) -> None:
        """Test that job status is False when stdout.txt does not contain success indicators."""
        stdout_file = tmp_path / "stdout.txt"
        stdout_content = """
        # Some initialization output
        # More output
        # Some final output without success indicators
        """
        stdout_file.write_text(stdout_content)
        result = self.js.get_job_status(str(tmp_path))
        assert not result.is_successful
        assert result.error_message == (
            f"Missing success indicators in {stdout_file}: '# Out of bounds values', '# Avg bus bandwidth'. "
            "These keywords are expected to be present in stdout.txt, usually towards the end of the file. "
            f"Please ensure the NCCL test ran to completion. You can run the generated sbatch script manually "
            f"and check if {stdout_file} is created and contains the expected keywords."
        )


class TestJaxToolboxJobStatusRetrievalStrategy:
    """Tests for the JaxToolboxJobStatusRetrievalStrategy class."""

    def setup_method(self) -> None:
        """Setup method for initializing JaxToolboxJobStatusRetrievalStrategy."""
        self.js = JaxToolboxJobStatusRetrievalStrategy()

    def test_no_profile_stderr_file(self, tmp_path: Path) -> None:
        """Test that job status is False when no profile_stderr.txt file is present."""
        result = self.js.get_job_status(str(tmp_path))
        assert not result.is_successful
        assert result.error_message == (
            f"profile_stderr.txt file not found in the specified output directory, {str(tmp_path)}. "
            "This file is expected to be created during the profiling stage. "
            "Please ensure the profiling stage completed successfully. "
            "Run the generated sbatch script manually to debug."
        )

    def test_missing_pax_status_keyword(self, tmp_path: Path) -> None:
        """Test that job status is False when profile_stderr.txt does not contain the PAX STATUS keyword."""
        profile_stderr_file = tmp_path / "profile_stderr.txt"
        profile_stderr_content = "Some initialization output\nMore output\nFinal output without the expected keyword"
        profile_stderr_file.write_text(profile_stderr_content)
        result = self.js.get_job_status(str(tmp_path))
        assert not result.is_successful
        assert result.error_message == (
            "The profiling stage completed but did not generate the expected '[PAX STATUS]: E2E time: "
            "Elapsed time for ' keyword. There are two stages in the Grok run, and an error occurred in "
            "the profiling stage. While profile_stderr.txt was created, the expected keyword is missing. "
            "You need to run the sbatch script manually to see what happens."
        )

    def test_no_error_files(self, tmp_path: Path) -> None:
        """Test that job status is False when no error-*.txt files are present."""
        profile_stderr_file = tmp_path / "profile_stderr.txt"
        profile_stderr_content = "[PAX STATUS]: E2E time: Elapsed time for profiling"
        profile_stderr_file.write_text(profile_stderr_content)
        result = self.js.get_job_status(str(tmp_path))
        assert not result.is_successful
        assert result.error_message == (
            f"No 'error-*.txt' files found in the output directory, {str(tmp_path)}. There are two stages in the Grok "
            "run. The profiling stage passed successfully, but something went wrong in the actual run stage. "
            "Please ensure the actual run stage completed successfully. "
            "Run the generated sbatch script manually to debug."
        )

    def test_cuda_no_device_error_in_profile_stderr(self, tmp_path: Path) -> None:
        """Test that job status is False when profile_stderr.txt contains CUDA_ERROR_NO_DEVICE."""
        profile_stderr_file = tmp_path / "profile_stderr.txt"
        profile_stderr_content = "[PAX STATUS]: E2E time: Elapsed time for profiling\n"
        profile_stderr_content += "CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected"
        profile_stderr_file.write_text(profile_stderr_content)

        result = self.js.get_job_status(str(tmp_path))
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
        """Test that job status is False when error-*.txt files do not contain the E2E time keyword."""
        profile_stderr_file = tmp_path / "profile_stderr.txt"
        profile_stderr_content = "[PAX STATUS]: E2E time: Elapsed time for profiling"
        profile_stderr_file.write_text(profile_stderr_content)

        error_file = tmp_path / "error-1.txt"
        error_content = "Some initialization output\nMore output\nFinal output without the expected keyword"
        error_file.write_text(error_content)

        result = self.js.get_job_status(str(tmp_path))
        assert not result.is_successful
        assert result.error_message == (
            f"The file {str(error_file)} does not contain the expected 'E2E time: Elapsed time for' keyword at the "
            "end. This indicates the actual run did not complete successfully. "
            "Please debug this manually to ensure the actual run stage completes as expected."
        )

    def test_cuda_no_device_error_in_error_file(self, tmp_path: Path) -> None:
        """Test that job status is False when error-*.txt contains CUDA_ERROR_NO_DEVICE."""
        profile_stderr_file = tmp_path / "profile_stderr.txt"
        profile_stderr_content = "[PAX STATUS]: E2E time: Elapsed time for profiling"
        profile_stderr_file.write_text(profile_stderr_content)

        error_file = tmp_path / "error-1.txt"
        error_content = "CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected"
        error_file.write_text(error_content)

        result = self.js.get_job_status(str(tmp_path))
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
        """Test that job status is True when profile_stderr.txt and error-*.txt files contain success indicators."""
        profile_stderr_file = tmp_path / "profile_stderr.txt"
        profile_stderr_content = "[PAX STATUS]: E2E time: Elapsed time for profiling"
        profile_stderr_file.write_text(profile_stderr_content)

        error_file = tmp_path / "error-1.txt"
        error_content = "Some initialization output\nMore output\nE2E time: Elapsed time for actual run\nFinal output"
        error_file.write_text(error_content)

        result = self.js.get_job_status(str(tmp_path))
        assert result.is_successful
        assert result.error_message == ""

    def test_nccl_group_end_error_in_profile_stderr(self, tmp_path: Path) -> None:
        """Test that job status is False when profile_stderr.txt contains NCCL operation ncclGroupEnd() failed."""
        profile_stderr_file = tmp_path / "profile_stderr.txt"
        profile_stderr_content = "[PAX STATUS]: E2E time: Elapsed time for profiling\n"
        profile_stderr_content += "NCCL operation ncclGroupEnd() failed: unhandled system error"
        profile_stderr_file.write_text(profile_stderr_content)

        result = self.js.get_job_status(str(tmp_path))
        assert not result.is_successful
        assert result.error_message == (
            "NCCL operation ncclGroupEnd() failed: unhandled system error. Please check if the NCCL-test "
            "passes. Run with NCCL_DEBUG=INFO for more details."
        )

    def test_nccl_group_end_error_in_error_file(self, tmp_path: Path) -> None:
        """Test that job status is False when error-*.txt contains NCCL operation ncclGroupEnd() failed."""
        profile_stderr_file = tmp_path / "profile_stderr.txt"
        profile_stderr_content = "[PAX STATUS]: E2E time: Elapsed time for profiling"
        profile_stderr_file.write_text(profile_stderr_content)

        error_file = tmp_path / "error-1.txt"
        error_content = "NCCL operation ncclGroupEnd() failed: unhandled system error"
        error_file.write_text(error_content)

        result = self.js.get_job_status(str(tmp_path))
        assert not result.is_successful
        assert result.error_message == (
            "NCCL operation ncclGroupEnd() failed: unhandled system error. Please check if the NCCL-test "
            "passes. Run with NCCL_DEBUG=INFO for more details."
        )

    def test_heartbeat_error_in_profile_stderr(self, tmp_path: Path) -> None:
        """Test that job status is False when profile_stderr.txt contains coordinator detected missing heartbeats."""
        profile_stderr_file = tmp_path / "profile_stderr.txt"
        profile_stderr_content = "[PAX STATUS]: E2E time: Elapsed time for profiling\n"
        profile_stderr_content += "Terminating process because the coordinator detected missing heartbeats"
        profile_stderr_file.write_text(profile_stderr_content)

        result = self.js.get_job_status(str(tmp_path))
        assert not result.is_successful
        assert result.error_message == (
            "Terminating process because the coordinator detected missing heartbeats. This most likely "
            f"indicates that another task died. Please review the file at {str(profile_stderr_file)} and any relevant "
            f"logs in {str(tmp_path)}. Ensure the servers allocated for this task can reach each other with their "
            "hostnames, and they can open any ports and reach others' ports."
        )

    def test_heartbeat_error_in_error_file(self, tmp_path: Path) -> None:
        """Test that job status is False when error-*.txt contains coordinator detected missing heartbeats."""
        profile_stderr_file = tmp_path / "profile_stderr.txt"
        profile_stderr_content = "[PAX STATUS]: E2E time: Elapsed time for profiling"
        profile_stderr_file.write_text(profile_stderr_content)

        error_file = tmp_path / "error-1.txt"
        error_content = "Terminating process because the coordinator detected missing heartbeats"
        error_file.write_text(error_content)

        result = self.js.get_job_status(str(tmp_path))
        assert not result.is_successful
        assert result.error_message == (
            "Terminating process because the coordinator detected missing heartbeats. This most likely "
            f"indicates that another task died. Please review the file at {str(error_file)} and any relevant logs in"
            f" {str(tmp_path)}. Ensure the servers allocated for this task can reach each other with their "
            "hostnames, and they can open any ports and reach others' ports."
        )
