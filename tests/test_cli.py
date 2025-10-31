# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path

import pytest
from click.testing import CliRunner

from cloudai.cli import main


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


def test_tests_dir_is_optional(tmp_path: Path):
    system_cfg, scenario_cfg = tmp_path / "system.toml", tmp_path / "scenario.toml"
    system_cfg.touch()
    scenario_cfg.touch()
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--system-config", str(system_cfg), "--test-scenario", str(scenario_cfg)])
    assert "Missing option '--tests-dir'" not in result.output


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
