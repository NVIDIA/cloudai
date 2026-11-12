# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

import cloudai.handlers
from cloudai.cli import main
from cloudai.cli.cli import handle_dry_run_and_run
from cloudai.core import JobSubmissionError


@pytest.mark.parametrize("cli", ["-h", "--help"])
def test_help(cli: str):
    runner = CliRunner()
    result = runner.invoke(main, [cli])
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "CloudAI, version" in result.output


def test_list_reward_functions_verbose():
    runner = CliRunner()
    result = runner.invoke(main, ["list", "reward-functions", "--verbose"])

    assert result.exit_code == 0
    assert "Available reward functions:" in result.output


def test_tests_dir_is_optional(tmp_path: Path):
    system_cfg, scenario_cfg = tmp_path / "system.toml", tmp_path / "scenario.toml"
    system_cfg.touch()
    scenario_cfg.touch()
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--system-config", str(system_cfg), "--test-scenario", str(scenario_cfg)])
    assert "Missing option '--tests-dir'" not in result.output


def test_system_config_can_be_set_via_env_var(tmp_path: Path):
    system_cfg, scenario_cfg = tmp_path / "system.toml", tmp_path / "scenario.toml"
    system_cfg.touch()
    scenario_cfg.touch()
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "--test-scenario", str(scenario_cfg)],
        env={"CLOUDAI_SYSTEM_CONFIG": str(system_cfg)},
    )
    assert "Missing option '--system-config'" not in result.output


def test_system_config_flag_takes_precedence_over_env_var(tmp_path: Path):
    system_cfg, other_cfg, scenario_cfg = (
        tmp_path / "system.toml",
        tmp_path / "other.toml",
        tmp_path / "scenario.toml",
    )
    system_cfg.touch()
    other_cfg.touch()
    scenario_cfg.touch()
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "--system-config", str(system_cfg), "--test-scenario", str(scenario_cfg)],
        env={"CLOUDAI_SYSTEM_CONFIG": str(other_cfg)},
    )
    assert "Missing 'scheduler' key" in result.output
    assert str(system_cfg) in result.output
    assert str(other_cfg) not in result.output


@pytest.mark.parametrize("subcommand", ["install", "uninstall", "dry-run", "run", "generate-report"])
def test_system_config_env_var_documented_in_help(subcommand: str):
    runner = CliRunner()
    result = runner.invoke(main, [subcommand, "--help"])
    assert result.exit_code == 0
    assert "CLOUDAI_SYSTEM_CONFIG" in result.output


@pytest.mark.parametrize("subcommand", ["install", "uninstall", "dry-run", "run", "generate-report"])
def test_hook_dir_is_available_on_run_commands(subcommand: str):
    runner = CliRunner()

    result = runner.invoke(main, [subcommand, "--help"])

    assert result.exit_code == 0
    assert "--hook-dir DIRECTORY" in result.output


@pytest.mark.parametrize(
    "subcommand", ["dry-run", "generate-report", "install", "list", "run", "uninstall", "verify-configs"]
)
def test_help_subcommands(subcommand: str):
    runner = CliRunner()
    result = runner.invoke(main, [subcommand, "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_result_dir_exists_for_generate_report(tmp_path: Path):
    system_cfg, scenario_cfg = tmp_path / "system.toml", tmp_path / "scenario.toml"
    system_cfg.touch()
    scenario_cfg.touch()

    result_dir = tmp_path / "results"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "generate-report",
            f"--system-config={system_cfg}",
            f"--tests-dir={tmp_path}",
            f"--test-scenario={scenario_cfg}",
            f"--result-dir={result_dir}",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--result-dir'" in result.output and "does not exist" in result.output


@pytest.mark.parametrize("opt", ["--system-config", "--test-scenario", "--tests-dir"])
@pytest.mark.parametrize("subcommand", ["run", "dry-run", "install", "uninstall", "generate-report"])
def test_mandatory_path_args(subcommand: str, opt: str, tmp_path: Path):
    system_cfg, scenario_cfg, tests_dir = tmp_path / "system.toml", tmp_path / "scenario.toml", tmp_path / "tests"
    opt2path = {"--system-config": system_cfg, "--tests-dir": tests_dir, "--test-scenario": scenario_cfg}
    for k, v in opt2path.items():
        if k == opt:
            continue
        if k == "--tests-dir":
            v.mkdir()
        else:
            v.touch()

    runner = CliRunner()
    result = runner.invoke(main, [subcommand, *[f"{k}={v}" for k, v in opt2path.items()]])
    assert result.exit_code == 2
    assert f"Invalid value for '{opt}'" in result.output and "does not exist" in result.output


def test_job_submission_error_fails_the_run(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    def cannot_submit(*args, **kwargs):
        raise JobSubmissionError("test", "sbatch run.sh", "", "sbatch: error", "Failed to submit")

    monkeypatch.setattr(cloudai.handlers, "load_experiment", lambda *args, **kwargs: (MagicMock(), [], MagicMock()))
    monkeypatch.setattr(cloudai.handlers, "create_experiment_runner", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(cloudai.handlers, "execute_experiment", cannot_submit)
    args = argparse.Namespace(
        test_scenario=Path("scenario.toml"),
        system_config=Path("system.toml"),
        tests_dir=None,
        hook_dir=None,
        output_dir=None,
        mode="run",
        single_sbatch=False,
        enable_cache_without_check=False,
    )

    assert handle_dry_run_and_run(args) == 1
    assert "Failed to submit" in caplog.text
