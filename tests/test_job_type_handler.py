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

from cloudai.cli.handlers import is_dse_job

mock_toml_dse = {
    "test": {
        "cmd_args": {
            "FakeConfig": {
                "policy": ["option1", "option2"],
                "dtype": "fake_type",
            },
        },
        "extra_env_vars": {"ENV_VAR_1": "value1", "ENV_VAR_2": "value2"},
    }
}

mock_toml_non_dse = {
    "test": {
        "cmd_args": {
            "FakeConfig": {
                "policy": "option1",
                "dtype": "fake_type",
            },
        },
        "extra_env_vars": {"ENV_VAR_1": "value1", "ENV_VAR_2": "value2"},
    }
}

mock_toml_dse_list_env_vars = {
    "test": {
        "cmd_args": {
            "FakeConfig": {
                "policy": ["option1", "option2"],
                "dtype": "fake_type",
            },
        },
        "extra_env_vars": {"ENV_VAR_1": ["value1", "value2"], "ENV_VAR_2": "value2"},
    }
}

mock_toml_non_dse_list_env_vars = {
    "test": {
        "cmd_args": {
            "FakeConfig": {
                "policy": "option1",
                "dtype": "fake_type",
            },
        },
        "extra_env_vars": {"ENV_VAR_1": "value1", "ENV_VAR_2": "value2"},
    }
}


def test_is_dse_job_dse():
    assert is_dse_job(mock_toml_dse["test"]["cmd_args"], mock_toml_dse["test"]["extra_env_vars"])


def test_is_dse_job_non_dse():
    assert not is_dse_job(mock_toml_non_dse["test"]["cmd_args"], mock_toml_non_dse["test"]["extra_env_vars"])


def test_is_dse_job_dse_list_env_vars():
    assert is_dse_job(
        mock_toml_dse_list_env_vars["test"]["cmd_args"], mock_toml_dse_list_env_vars["test"]["extra_env_vars"]
    )


def test_is_dse_job_non_dse_list_env_vars():
    assert not is_dse_job(
        mock_toml_non_dse_list_env_vars["test"]["cmd_args"], mock_toml_non_dse_list_env_vars["test"]["extra_env_vars"]
    )
