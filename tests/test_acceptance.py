from pathlib import Path
from typing import Dict

import pytest
from cloudai.__main__ import handle_dry_run_and_run

SLURM_TEST_SCENARIOS = [
    {
        "path": Path("conf/v0.6/general/test_scenario/sleep.toml"),
        "expected_dirs_number": 3,
    },
    {
        "path": Path("conf/v0.6/general/test_scenario/ucc_test.toml"),
        "expected_dirs_number": 1,
    },
]


@pytest.mark.parametrize("scenario", SLURM_TEST_SCENARIOS, ids=lambda x: str(x))
def test_slurm(tmp_path: Path, scenario: Dict):
    test_scenario_path = scenario["path"]
    expected_dirs_number = scenario.get("expected_dirs_number")

    handle_dry_run_and_run(
        "dry-run",
        Path("conf/v0.6/general/system/example_slurm_cluster.toml"),
        Path("conf/v0.6/general/test_template"),
        Path("conf/v0.6/general/test"),
        test_scenario_path,
        tmp_path,
    )

    results_output = list(tmp_path.glob("*"))[0]
    test_dirs = list(results_output.iterdir())

    if expected_dirs_number is not None:
        assert len(test_dirs) == expected_dirs_number, "Dirs number in output is not as expected"

    for td in test_dirs:
        assert td.is_dir(), "Invalid test directory"
        assert "Tests." in td.name, "Invalid test directory name"
