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

import os
from pathlib import Path
from unittest import mock

import pytest

from cloudai.workloads.ai_dynamo.ai_dynamo import AIDynamoCmdArgs


class TestHFHostPath:
    @pytest.fixture
    def args(self) -> dict:
        return {
            "docker_image_url": "docker://image",
            "dynamo": {"prefill_worker": {"num-nodes": 1}, "decode_worker": {"num-nodes": 1}},
            "genai_perf": {},
        }

    @pytest.fixture
    def existing_dir(self, tmp_path: Path) -> Path:
        custom_path = tmp_path / "custom/hf/cache"
        custom_path.mkdir(parents=True, exist_ok=True)
        return custom_path

    def test_default(self, args: dict, tmp_path: Path):
        existing_hf_home = tmp_path / "default_hf_home" / ".cache/huggingface"
        existing_hf_home.mkdir(parents=True, exist_ok=True)
        with (
            mock.patch("pathlib.Path.home", return_value=existing_hf_home.parent.parent),
            mock.patch.dict(os.environ, clear=True),
        ):
            dynamo = AIDynamoCmdArgs(**args)
            assert dynamo.huggingface_home_host_path == existing_hf_home

    def test_model_override(self, args: dict, existing_dir: Path):
        test_args = {**args, "huggingface_home_host_path": existing_dir}
        with mock.patch.dict(os.environ, clear=True):
            dynamo = AIDynamoCmdArgs.model_validate(test_args)
            assert dynamo.huggingface_home_host_path == existing_dir

    def test_model_override_no_dir(self, args: dict, tmp_path: Path):
        test_args = {**args, "huggingface_home_host_path": tmp_path / "nonexistent"}
        with mock.patch.dict(os.environ, clear=True):
            with pytest.raises(ValueError) as exc_info:
                AIDynamoCmdArgs.model_validate(test_args)
            assert "Path is invalid" in str(exc_info.value)
            assert "overridden by HF_HOME environment variable" not in str(exc_info.value)

    def test_env_override_no_dir(self, args: dict, tmp_path: Path):
        non_existing_dir = tmp_path / "nonexistent_env"
        with mock.patch.dict(os.environ, {"HF_HOME": str(non_existing_dir)}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                AIDynamoCmdArgs(**args)
            assert "Path is invalid" in str(exc_info.value)
            assert "overridden by HF_HOME environment variable" in str(exc_info.value)

    def test_env_override_no_model(self, args: dict, existing_dir: Path):
        with mock.patch.dict(os.environ, {"HF_HOME": str(existing_dir)}):
            dynamo = AIDynamoCmdArgs(**args)
            assert dynamo.huggingface_home_host_path == existing_dir

    def test_env_override_with_model(self, args: dict, existing_dir: Path):
        test_args = {**args, "huggingface_home_host_path": existing_dir}
        env_path = existing_dir / "env_override"
        env_path.mkdir(parents=True, exist_ok=True)
        with mock.patch.dict(os.environ, {"HF_HOME": str(env_path)}):
            dynamo = AIDynamoCmdArgs.model_validate(test_args)
            assert dynamo.huggingface_home_host_path == env_path
