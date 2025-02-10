# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from unittest.mock import patch

import pytest

from cloudai.cli import CloudAICLI, handle_generate_report, handle_install_and_uninstall
from cloudai.cli.handlers import handle_verify_all_configs


def test_help_message(capsys: pytest.CaptureFixture[str]) -> None:
    cli = CloudAICLI()
    with patch("sys.argv", ["cloudai", "--help"]):
        with pytest.raises(SystemExit) as e:
            cli.run()
        assert e.value.code == 0

    captured = capsys.readouterr()
    assert "CloudAI" in captured.out


def test_command_is_mandatory(capsys: pytest.CaptureFixture[str]) -> None:
    cli = CloudAICLI()
    with patch("sys.argv", ["cloudai"]):
        with pytest.raises(SystemExit) as e:
            cli.run()
        assert e.value.code == 2

    captured = capsys.readouterr()
    assert "the following arguments are required: mode" in captured.err


def test_can_add_and_use_command() -> None:
    cli = CloudAICLI()

    called = False

    def handler(args):
        nonlocal called
        called = True
        return 0

    cli.add_command("test", "Test command", handler)
    assert "test" in cli.handlers

    with patch("sys.argv", ["cloudai", "test"]):
        assert cli.run() == 0
        assert called


def test_no_default_args() -> None:
    cli = CloudAICLI()

    cli.add_command("test", "Test command", lambda _: 0)
    args = cli.parser.parse_args(["test"])
    assert args == argparse.Namespace(log_file="debug.log", log_level="INFO", mode="test")


def test_default_args() -> None:
    cli = CloudAICLI()

    with patch.object(cli, "add_command") as add_command:
        cli.init_default_args()

    assert add_command.call_count == len(cli.DEFAULT_MODES)

    enabled_modes = set()
    for idx in range(len(cli.DEFAULT_MODES)):
        enabled_modes.add(add_command.call_args_list[idx][0][0])

    assert enabled_modes == cli.DEFAULT_MODES


def test_disable_default_modes() -> None:
    cli = CloudAICLI()

    cli.DEFAULT_MODES.clear()

    with patch.object(cli, "add_command") as add_command:
        cli.init_default_args()

    assert add_command.call_count == 0
    assert cli.handlers == {}


def test_add_command_all_optional():
    cli = CloudAICLI()

    cli.add_command(
        "test",
        "Test command",
        lambda _: 0,
        system_config=False,
        tests_dir=False,
        test_scenario=False,
        output_dir=False,
    )
    args = cli.parser.parse_args(["test"])
    assert args == argparse.Namespace(
        log_file="debug.log",
        log_level="INFO",
        mode="test",
        system_config=None,
        tests_dir=None,
        test_scenario=None,
        output_dir=None,
    )


def test_add_command_all_required():
    cli = CloudAICLI()

    cli.add_command(
        "test",
        "Test command",
        lambda _: 0,
        system_config=True,
        tests_dir=True,
        test_scenario=True,
        output_dir=True,
    )
    args = cli.parser.parse_args(
        [
            "test",
            "--system-config",
            "system_config",
            "--tests-dir",
            "tests_dir",
            "--test-scenario",
            "test_scenario",
            "--output-dir",
            "output_dir",
        ]
    )
    assert args == argparse.Namespace(
        log_file="debug.log",
        log_level="INFO",
        mode="test",
        system_config=Path("system_config"),
        tests_dir=Path("tests_dir"),
        test_scenario=Path("test_scenario"),
        output_dir=Path("output_dir"),
    )


def test_real_uninstall():
    cli = CloudAICLI()
    cli.init_default_args()

    args = cli.parser.parse_args(
        [
            "uninstall",
            "--system-config",
            "conf/common/system/example_slurm_cluster.toml",
            "--tests-dir",
            "conf/common/test",
        ]
    )
    cli.handlers["uninstall"](args)


class TestCLIDefaultModes:
    @pytest.fixture()
    def cli(self) -> CloudAICLI:
        cli = CloudAICLI()
        cli.init_default_args()
        return cli

    def test_install_uninstall_modes(self, cli: CloudAICLI):
        assert "install" in cli.handlers
        assert "uninstall" in cli.handlers

        assert cli.handlers["install"] is handle_install_and_uninstall
        assert cli.handlers["uninstall"] is handle_install_and_uninstall

        for mode in {"install", "uninstall"}:
            args = cli.parser.parse_args(
                [
                    mode,
                    "--system-config",
                    "system_config",
                    "--tests-dir",
                    "tests_dir",
                ]
            )

            assert args == argparse.Namespace(
                log_file="debug.log",
                log_level="INFO",
                mode=mode,
                system_config=Path("system_config"),
                tests_dir=Path("tests_dir"),
                test_scenario=None,
                output_dir=None,
            )

    def test_verify_all_configs_mode(self, cli: CloudAICLI):
        assert "verify-configs" in cli.handlers
        assert cli.handlers["verify-configs"] is handle_verify_all_configs

        args = cli.parser.parse_args(
            ["verify-configs", "--system-config", "system_config", "--tests-dir", "tests_dir", "configs_dir"]
        )
        assert args == argparse.Namespace(
            log_file="debug.log",
            log_level="INFO",
            mode="verify-configs",
            system_config=Path("system_config"),
            tests_dir=Path("tests_dir"),
            **{"configs_dir": Path("configs_dir")},
        )

        args = cli.parser.parse_args(["verify-configs", "configs_dir"])
        assert args == argparse.Namespace(
            log_file="debug.log",
            log_level="INFO",
            mode="verify-configs",
            system_config=None,
            tests_dir=None,
            **{"configs_dir": Path("configs_dir")},
        )

    def test_report_generation_mode(self, cli: CloudAICLI):
        assert "generate-report" in cli.handlers
        assert cli.handlers["generate-report"] is handle_generate_report

        args = cli.parser.parse_args(
            [
                "generate-report",
                "--system-config",
                "system_config",
                "--tests-dir",
                "tests_dir",
                "--test-scenario",
                "test_scenario",
                "--result-dir",
                "result_dir",
            ]
        )
        assert args == argparse.Namespace(
            log_file="debug.log",
            log_level="INFO",
            mode="generate-report",
            test_scenario=Path("test_scenario"),
            result_dir=Path("result_dir"),
            system_config=Path("system_config"),
            tests_dir=Path("tests_dir"),
        )

    def test_run_dry_run_modes(self, cli: CloudAICLI):
        assert "dry-run" in cli.handlers
        assert "run" in cli.handlers

        for mode in {"dry-run", "run"}:
            args = cli.parser.parse_args(
                [
                    mode,
                    "--system-config",
                    "system_config",
                    "--tests-dir",
                    "tests_dir",
                    "--test-scenario",
                    "test_scenario",
                ]
            )

            assert args == argparse.Namespace(
                log_file="debug.log",
                log_level="INFO",
                mode=mode,
                system_config=Path("system_config"),
                tests_dir=Path("tests_dir"),
                test_scenario=Path("test_scenario"),
                output_dir=None,
            )

    @pytest.mark.parametrize(
        "mode_and_missing_options",
        [
            (
                "generate-report",
                ["--system-config", "--tests-dir", "--test-scenario", "--output-dir"],
            ),
            ("install", ["--system-config", "--tests-dir"]),
            ("uninstall", ["--system-config", "--tests-dir"]),
            ("dry-run", ["--system-config", "--tests-dir", "--test-scenario"]),
            ("run", ["--system-config", "--tests-dir", "--test-scenario"]),
        ],
    )
    def test_required_args(
        self, mode_and_missing_options: tuple[str, list[str]], cli: CloudAICLI, capsys: pytest.CaptureFixture[str]
    ):
        opts = {
            "--system-config": "system_config",
            "--tests-dir": "tests_dir",
            "--test-scenario": "test_scenario",
            "--output-dir": "output_dir",
        }
        mode, missing_options = mode_and_missing_options

        for missing_option in missing_options:
            opts.pop(missing_option)

            with pytest.raises(SystemExit) as e:
                cli.parser.parse_args([mode, *[f"{k} {v}" for k, v in opts.items()]])

            assert e.value.code == 2
            assert f"{mode}: error: the following arguments are required" in capsys.readouterr().err
