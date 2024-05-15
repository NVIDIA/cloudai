from pathlib import Path

from cloudai.schema.test_template.jax_toolbox.report_generation_strategy import (
    JaxToolboxReportGenerationStrategy,
)


class TestJaxExtractTime:
    """Tests for the JaxToolboxReportGenerationStrategy class."""

    def setup_method(self) -> None:
        """Setup method for initializing JaxToolboxReportGenerationStrategy."""
        self.js = JaxToolboxReportGenerationStrategy()

    def test_no_files(self, tmp_path: Path) -> None:
        """Test that no times are extracted when no files are present."""
        assert self.js._extract_times(str(tmp_path)) == []

    def test_no_matches(self, tmp_path: Path) -> None:
        """Test that no times are extracted when no matching lines are present."""
        (tmp_path / "error-1.txt").write_text("fake line")
        assert self.js._extract_times(str(tmp_path)) == []

    def test_one_match(self, tmp_path: Path) -> None:
        """Test that the correct time is extracted when one matching line is present."""
        err_file = tmp_path / "error-1.txt"
        sample_line = (
            "I0508 15:25:28.482553 140737334253888 programs.py:379] "
            "[PAX STATUS]: train_step() took 38.727223 seconds.\n"
        )
        with err_file.open("w") as f:
            for _ in range(11):
                f.write(sample_line)
        assert self.js._extract_times(str(err_file.parent)) == [38.727223]
