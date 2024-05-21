"""
Execute a scenario in dry-run mode and compare the results folder with an
expected output folder, if any difference is found in one of the files,
the test fails. Note that the empty lines and commented lines
(the ones that start with #) are not taken into account when computing the
difference.
Expected output can be None, in this case the output won't be compared and basic tests will be performed instead

# How to add a new test:

- Create the desired scenario
- Create a new folder under tests/expected_outputs/test_acceptance/ named after the name
of this test and fill it with the expected output of the scenario
(you can copy the output of a valid dry run)
- Add the path to the scenario and to this expected output folder to SLURM_TEST_SCENARIOS below
"""

import argparse
from pathlib import Path
from typing import Optional

import pytest
from cloudai.__main__ import handle_dry_run_and_run

SLURM_TEST_SCENARIOS = [
    {
        "scenario": Path("conf/v0.6/general/test_scenario/nccl_test/test_scenario.toml"),
        "expected_output": Path("tests/expected_outputs/test_acceptance/nccl_test"),
    },
    {
        "scenario": Path("conf/v0.6/general/test_scenario/sleep/test_scenario.toml"),
        "expected_output": None,
    },
    {
        "scenario": Path("conf/v0.6/general/test_scenario/ucc_test/test_scenario.toml"),
        "expected_output": Path("tests/expected_outputs/test_acceptance/ucc_test"),
    },
]


def read_wout_comments(f):
    with open(f) as fp:
        return "\n".join(line.strip() for line in fp.readlines() if line.strip() and not line.strip().startswith("#"))


def diff_dirs(dir1: Path, dir2: Path) -> bool:
    """Recursively compare the two directories and check if the files are all the same
    This function ignore empty lines and commented lines
    """
    ok = True
    assert dir1.is_dir() and dir2.is_dir(), f"One of the directory is invalid ({dir1=}, {dir2=}"

    assert len(list(dir1.iterdir())) == len(list(dir2.iterdir())), f"Dirs {dir1} and {dir2} have different files count"

    for p in dir1.iterdir():
        sib_p = dir2.joinpath(p.name)
        if p.is_dir() and not diff_dirs(p, sib_p):
            ok = False
        elif p.is_file():
            assert sib_p.is_file(), f"File {sib_p} doesn't exist but its sibling {p} does"
            assert read_wout_comments(p) == read_wout_comments(sib_p), f"Different content in files {p} and {sib_p}"

    return ok


@pytest.mark.parametrize(
    "test_scenario_path, expected_output",
    [(e["scenario"], e["expected_output"]) for e in SLURM_TEST_SCENARIOS],
    ids=lambda x: str(x),
)
def test_slurm(tmp_path: Path, test_scenario_path: Path, expected_output: Optional[Path]):
    args = argparse.Namespace(
        log_file=None,
        log_level=None,
        mode="dry-run",
        output_path=str(tmp_path),
        system_config_path="conf/v0.6/general/system/ci.toml",
        test_scenario_path=str(test_scenario_path),
        test_path="conf/v0.6/general/test",
        test_template_path="conf/v0.6/general/test_template",
    )
    handle_dry_run_and_run(args)

    test_dir = list(tmp_path.glob("*"))[0]

    if expected_output is not None:
        diff_dirs(test_dir, expected_output), "Output is not as expected"
    else:
        for td in test_dir.iterdir():
            assert td.is_dir(), "Invalid test directory"
            assert "Tests." in td.name, "Invalid test directory name"
