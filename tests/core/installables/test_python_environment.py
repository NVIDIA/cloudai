# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from cloudai.core import BaseInstaller, PythonEnvironment


@pytest.fixture
def installer(slurm_system) -> BaseInstaller:
    installer = BaseInstaller(slurm_system)
    installer.system.install_path.mkdir(parents=True)
    installer._check_low_thread_environment = lambda threshold=None: False
    return installer


def test_python_environment_identity_uses_stable_configuration() -> None:
    env = PythonEnvironment(name="aiconfigurator", python_version="3.10", requirements=["aiconfigurator~=0.5.0"])
    same = PythonEnvironment(name="aiconfigurator", python_version="3.10", requirements=["aiconfigurator~=0.5.0"])
    different = PythonEnvironment(name="aiconfigurator", python_version="3.11", requirements=["aiconfigurator~=0.5.0"])

    assert env == same
    assert hash(env) == hash(same)
    assert env != different


def test_python_environment_install_uses_uv(installer: BaseInstaller) -> None:
    env = PythonEnvironment(name="aiconfigurator", python_version="3.10", requirements=["aiconfigurator~=0.5.0"])

    with (
        patch(
            "cloudai._core.installables.python_environment.resolve_uv_bin",
            return_value="/cloudai/bin/uv",
        ) as resolve_uv,
        patch("subprocess.run") as run,
    ):
        run.return_value = CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        res = env.install(installer)

    assert res.success
    resolve_uv.assert_called_once_with()
    assert env.venv_path == installer.system.install_path / env.venv_name
    assert run.call_args_list[0].args[0] == ["/cloudai/bin/uv", "python", "install", "3.10"]
    assert run.call_args_list[1].args[0] == [
        "/cloudai/bin/uv",
        "venv",
        "--python",
        "3.10",
        str(installer.system.install_path / env.venv_name),
    ]
    assert run.call_args_list[2].args[0] == [
        "/cloudai/bin/uv",
        "pip",
        "install",
        "--python",
        str(env.python_path(installer.system.install_path)),
        "aiconfigurator~=0.5.0",
    ]


def test_python_environment_reports_bundled_uv_resolution_failure(installer: BaseInstaller) -> None:
    env = PythonEnvironment(name="aiconfigurator", python_version="3.10")

    with patch(
        "cloudai._core.installables.python_environment.resolve_uv_bin",
        side_effect=RuntimeError("bundled uv is unavailable"),
    ):
        res = env.install(installer)

    assert not res.success
    assert res.message == "Cannot install Python environment: bundled uv is unavailable"


def test_packaged_uv_resolver_uses_public_uv_api() -> None:
    from cloudai._core.installables._uv import resolve_uv_bin

    with patch("cloudai._core.installables._uv.uv.find_uv_bin", return_value="/cloudai/bin/uv") as find_uv_bin:
        assert resolve_uv_bin() == "/cloudai/bin/uv"

    find_uv_bin.assert_called_once_with()


def test_python_environment_is_installed_checks_python_executable(installer: BaseInstaller) -> None:
    env = PythonEnvironment(name="aiconfigurator", python_version="3.10")
    venv_path = installer.system.install_path / env.venv_name
    python_path = env.python_path(installer.system.install_path)
    python_path.parent.mkdir(parents=True)
    python_path.touch()

    res = env.is_installed(installer)

    assert res.success
    assert env.venv_path == venv_path


def test_python_environment_uninstall_removes_venv(installer: BaseInstaller) -> None:
    env = PythonEnvironment(name="aiconfigurator", python_version="3.10")
    venv_path = installer.system.install_path / env.venv_name
    (venv_path / "bin").mkdir(parents=True)
    env.venv_path = venv_path

    res = env.uninstall(installer)

    assert res.success
    assert not venv_path.exists()
    assert env.venv_path is None
