# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from ..nemo_launcher.nemo_launcher import NeMoLauncherCmdArgs, NeMoLauncherTestDefinition


class DataPreparationRun(BaseModel):
    """DataPreparationRun."""

    model_config = ConfigDict(extra="allow")
    results_dir: str = "${base_results_dir}/${name}"


class DataPreparationConfig(BaseModel):
    """DataPreparationConfig."""

    model_config = ConfigDict(extra="allow")
    file_numbers: str = ""
    download_tokenizer_url: Optional[str] = None
    tokenizer_library: str = "huggingface"
    tokenizer_save_dir: Optional[str] = None
    tokenizer_model: Optional[str] = None
    rm_downloaded: bool = True
    rm_extracted: bool = True
    tokenizer_path: str = "${stage_path}/llama3.1-dataset/llama"


class DatasetPreparationCmdArgs(NeMoLauncherCmdArgs):
    """DatasetPreparationCmdArgs."""

    data_dir: str = ""
    stages: str = '["data_preparation"]'
    run: DataPreparationRun = Field(default_factory=DataPreparationRun)
    data_preparation: DataPreparationConfig = Field(default_factory=DataPreparationConfig)


class DatasetPreparationTestDefinition(NeMoLauncherTestDefinition[DatasetPreparationCmdArgs]):
    """DatasetPreparationTestDefinition."""

    cmd_args: DatasetPreparationCmdArgs
