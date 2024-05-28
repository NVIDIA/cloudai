import argparse
from pathlib import Path

import pytest
from cloudai.__main__ import handle_dry_run_and_run

SLURM_TEST_SCENARIOS = [
    Path("conf/v0.6/general/test_scenario/sleep.toml"),
    Path("conf/v0.6/general/test_scenario/ucc_test.toml"),
]


@pytest.mark.parametrize("test_scenario_path", SLURM_TEST_SCENARIOS, ids=lambda x: str(x))
def test_slurm(tmp_path: Path, test_scenario_path: Path):
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

    test_dir = list(tmp_path.glob("*"))[0]
    for td in test_dir.iterdir():
        assert td.is_dir(), "Invalid test directory"
        assert "Tests." in td.name, "Invalid test directory name"
