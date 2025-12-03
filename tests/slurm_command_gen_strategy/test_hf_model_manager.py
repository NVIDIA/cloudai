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
from unittest.mock import patch

import pytest

from cloudai.core import HFModel
from cloudai.util.hf_model_manager import HFModelManager


@pytest.fixture
def hf_model() -> HFModel:
    return HFModel(model_name="some_model_name")


def test_download(hf_model: HFModel, tmp_path: Path) -> None:
    assert hf_model._installed_path is None

    with patch("cloudai.util.hf_model_manager.snapshot_download", return_value=str("/real/path")):
        HFModelManager(root_path=tmp_path).download_model(hf_model)

    assert hf_model.installed_path == Path("/real/path")


def test_download_raises(hf_model: HFModel, tmp_path: Path) -> None:
    with patch(
        "cloudai.util.hf_model_manager.snapshot_download",
        side_effect=Exception("some error message"),
    ):
        result = HFModelManager(root_path=tmp_path).download_model(hf_model)

    assert not result.success
    assert "some error message" in result.message
    assert hf_model._installed_path is None
