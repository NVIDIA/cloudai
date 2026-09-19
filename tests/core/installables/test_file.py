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

from pathlib import Path

import pytest

from cloudai.core import BaseInstaller, File


@pytest.fixture
def installer(slurm_system):
    installer = BaseInstaller(slurm_system)
    installer.system.install_path.mkdir(parents=True)
    return installer


@pytest.fixture
def f(tmp_path: Path) -> File:
    f = tmp_path / "file"
    f.write_text("content")
    return File(f)


def test_no_dst(installer: BaseInstaller, f: File):
    res = f.install(installer)
    assert res.success
    assert f.installed_path == installer.system.install_path / f.src.name
    assert f.installed_path.exists()
    assert f.installed_path.read_bytes() == f.src.read_bytes()


def test_file_exists_but_overriden(installer: BaseInstaller, f: File):
    f.installed_path = installer.system.install_path / f.src.name
    f.installed_path.touch()
    res = f.install(installer)
    assert res.success
    assert f.src.read_bytes() == f.installed_path.read_bytes()


def test_is_installed_checks_content(installer: BaseInstaller, f: File):
    f.installed_path = installer.system.install_path / f.src.name
    f.installed_path.touch()
    f.src.write_text("new content")

    res = f.is_installed(installer)
    assert not res.success
    assert res.message == f"File {f.installed_path} differs from {f.src}"


def test_is_installed_missing_file(installer: BaseInstaller, f: File):
    res = f.is_installed(installer)
    assert not res.success
    assert res.message == f"File {installer.system.install_path / f.src.name} does not exist"


def test_is_installed_binary_file(installer: BaseInstaller, tmp_path: Path):
    src = tmp_path / "file.bin"
    src.write_bytes(b"\xff\xfe\x00 not valid UTF-8")
    f = File(src)
    f.install(installer)

    res = f.is_installed(installer)
    assert res.success
    assert f.installed_path == installer.system.install_path / src.name
