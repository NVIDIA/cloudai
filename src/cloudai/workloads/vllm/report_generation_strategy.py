import json
from functools import cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.table import Table

from cloudai.core import ReportGenerationStrategy

from .vllm import VLLM_BENCH_JSON_FILE


class VLLMBenchReport(BaseModel):
    """Report for vLLM benchmark results."""

    model_config = ConfigDict(extra="ignore")

    num_prompts: int
    completed: int
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p99_tpot_ms: float


@cache
def parse_vllm_bench_output(res_file: Path) -> VLLMBenchReport | None:
    """Parse the vLLM benchmark output file and return a VLLMBenchReport object."""
    if not res_file.is_file():
        return None

    with res_file.open("r") as f:
        data = json.load(f)

    return VLLMBenchReport.model_validate(data)


class VLLMBenchReportGenerationStrategy(ReportGenerationStrategy):
    """Generate a report for vLLM benchmark results."""

    def can_handle_directory(self) -> bool:
        return parse_vllm_bench_output(self.test_run.output_path / VLLM_BENCH_JSON_FILE) is not None

    def generate_report(self) -> None:
        results = parse_vllm_bench_output(self.test_run.output_path / VLLM_BENCH_JSON_FILE)
        if results is None:
            return

        console = Console()
        table = Table(title=f"vLLM Benchmark Results ({self.test_run.output_path})", title_justify="left")
        table.add_column("Successful prompts", justify="right")
        table.add_column("TTFT Mean, ms", justify="right")
        table.add_column("TTFT Median, ms", justify="right")
        table.add_column("TTFT P99, ms", justify="right")
        table.add_column("TPOT Mean, ms", justify="right")
        table.add_column("TPOT Median, ms", justify="right")
        table.add_column("TPOT P99, ms", justify="right")
        table.add_row(
            f"{results.completed / results.num_prompts * 100:.2f}% ({results.completed} of {results.num_prompts})",
            f"{results.mean_ttft_ms:.4f}",
            f"{results.median_ttft_ms:.4f}",
            f"{results.p99_ttft_ms:.4f}",
            f"{results.mean_tpot_ms:.4f}",
            f"{results.median_tpot_ms:.4f}",
            f"{results.p99_tpot_ms:.4f}",
        )

        console.print(table)
