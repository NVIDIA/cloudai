from pathlib import Path

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
            "stdout.txt file not found in the specified output directory. "
            "This file is expected to be created as a result of the NCCL test run. "
            "Please ensure the NCCL test was executed properly and that stdout.txt is generated. "
            "You can run the generated NCCL test command manually and verify the creation of stdout.txt."
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
            "Missing success indicators in stdout.txt: '# Out of bounds values', '# Avg bus bandwidth'. "
            "These keywords are expected to be present in stdout.txt, usually towards the end of the file. "
            "Please ensure the NCCL test ran to completion. You can run the generated sbatch script manually "
            "and check if stdout.txt is created and contains the expected keywords."
        )
