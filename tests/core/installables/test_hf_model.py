# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from unittest.mock import patch

import pytest

from cloudai.core import BaseInstaller, HFModel


@pytest.fixture
def hf_model() -> HFModel:
    return HFModel(model_name="some_model_name")


@pytest.fixture
def installer(slurm_system, tmp_path: Path) -> BaseInstaller:
    slurm_system.hf_home_path = tmp_path
    return BaseInstaller(slurm_system)


def test_download(hf_model: HFModel, installer: BaseInstaller) -> None:
    assert hf_model._installed_path is None

    with patch("huggingface_hub.snapshot_download", return_value=str("/real/path")):
        hf_model.install(installer)

    assert hf_model.installed_path == Path("/real/path")


def test_download_raises(hf_model: HFModel, installer: BaseInstaller) -> None:
    with patch("huggingface_hub.snapshot_download", side_effect=Exception("some error message")):
        result = hf_model.install(installer)

    assert not result.success
    assert "some error message" in result.message
    assert hf_model._installed_path is None


def test_is_downloaded(hf_model: HFModel, installer: BaseInstaller) -> None:
    with patch("huggingface_hub.snapshot_download", return_value=str("/real/path")):
        result = hf_model.is_installed(installer)

    assert result.success
    assert result.message == "/real/path"
    assert hf_model.installed_path == Path("/real/path")


def test_is_downloaded_raises(hf_model: HFModel, installer: BaseInstaller) -> None:
    with patch("huggingface_hub.snapshot_download", side_effect=Exception("some error message")):
        result = hf_model.is_installed(installer)

    assert not result.success
    assert "some error message" in result.message
    assert hf_model._installed_path is None
