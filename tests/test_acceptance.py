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
        "expected_dirs_number": 4,
    },
    {
        "scenario": Path("conf/v0.6/general/test_scenario/ucc_test/test_scenario.toml"),
        "expected_output": Path("tests/expected_outputs/test_acceptance/ucc_test"),
    },
]


def read_wout_comments(f):
    with open(f) as fp:
        return "\n".join(line.strip() for line in fp.readlines() if line.strip() and not line.strip().startswith("#"))


@pytest.mark.parametrize(
    "test_scenario_path, expected_output",
    [(e["scenario"], e["expected_output"]) for e in SLURM_TEST_SCENARIOS if e.get("expected_output")],
    ids=lambda x: str(x),
)
def test_dry_run_compare_output(tmp_path: Path, test_scenario_path: Path, expected_output: Path):
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

    results_output = list(tmp_path.glob("*"))[0]

    assert results_output.is_dir(), "Output is not a directory"
    assert expected_output.is_dir(), "Expected output is not a directory"

    results_files = list(results_output.glob("**/*"))
    expected_files = list(expected_output.glob("**/*"))

    assert len(results_files) == len(expected_files), "Results don't have the expected number of files"

    for p in expected_files:
        relpath = p.relative_to(expected_output)
        sibling_p = results_output.joinpath(relpath)
        if p.is_dir():
            assert sibling_p.is_dir(), f"Dir {sibling_p} is expected as a sibling of {p} but doesn't exist"
        elif p.is_file():
            assert sibling_p.is_file(), f"File {sibling_p} is expected as a sibling of {p} d but doesn't exist"
            assert read_wout_comments(p) == read_wout_comments(
                sibling_p
            ), f"Different content in files {p} and {sibling_p}"


@pytest.mark.parametrize(
    "test_scenario_path, expected_dirs_number",
    [(e["scenario"], e.get("expected_dirs_number")) for e in SLURM_TEST_SCENARIOS if not e.get("expected_output")],
    ids=lambda x: str(x),
)
def test_dry_run_structure(tmp_path: Path, test_scenario_path: Path, expected_dirs_number: Optional[int]):
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

    results_output = list(tmp_path.glob("*"))[0]

    test_dirs = list(results_output.iterdir())

    if expected_dirs_number is not None:
        assert len(test_dirs) == expected_dirs_number, "Dirs number in output is not as expected"

    for td in test_dirs:
        assert td.is_dir(), "Invalid test directory"
        assert "Tests." in td.name, "Invalid test directory name"
