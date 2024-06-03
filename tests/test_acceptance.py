import argparse
from pathlib import Path
from typing import List, Dict

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
    }
]


@pytest.mark.parametrize("scenario", SLURM_TEST_SCENARIOS, ids=lambda x: str(x))
def test_slurm(tmp_path: Path, scenario: Dict):
    test_scenario_path = scenario["path"]
    expected_dirs_number = scenario.get("expected_dirs_number")

    args = argparse.Namespace(
        log_file=None,
        log_level=None,
        mode="dry-run",
        output_path=str(tmp_path),
        system_config_path="conf/v0.6/general/system/example_slurm_cluster.toml",
        test_scenario_path=str(test_scenario_path),
        test_path="conf/v0.6/general/test",
        test_template_path="conf/v0.6/general/test_template",
    )
    handle_dry_run_and_run(args)

    results_output = list(tmp_path.glob("*"))[0]
    test_dirs = list(results_output.iterdir())

    if expected_dirs_number is not None:
        assert len(test_dirs) == expected_dirs_number, "Dirs number in output is not as expected"

    for td in test_dirs:
        assert td.is_dir(), "Invalid test directory"
        assert "Tests." in td.name, "Invalid test directory name"
