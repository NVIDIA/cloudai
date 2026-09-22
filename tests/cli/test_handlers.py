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

import inspect
from unittest.mock import Mock, call

import pytest

import cloudai.cli.handlers
import cloudai.handlers


@pytest.mark.parametrize(
    "name",
    [
        "_log_installation_dirs",
        "handle_install_and_uninstall",
        "prepare_installation",
        "_scenario_installables",
        "handle_dse_job",
        "_record_run_failure",
        "generate_reports",
        "handle_non_dse_job",
        "register_signal_handlers",
        "_setup_system_and_scenario",
        "_handle_single_sbatch",
        "_check_installation",
        "handle_dry_run_and_run",
        "handle_generate_report",
        "expand_file_list",
        "_ensure_kube_config_exists",
        "verify_system_configs",
        "verify_test_configs",
        "verify_test_scenarios",
        "handle_verify_all_configs",
        "load_tomls_by_type",
        "handle_list_registered_items",
        "validate_domain_randomization_active",
        "load_test_toml_file",
        "format_toml_decode_error",
        "prepare_output_dir",
    ],
)
def test_legacy_handler(name: str, monkeypatch: pytest.MonkeyPatch):
    original = getattr(cloudai.handlers, name)
    legacy = getattr(cloudai.cli.handlers, name)
    signature = inspect.signature(original)
    assert inspect.signature(legacy) == signature
    assert legacy.__doc__ == original.__doc__

    target = Mock()
    monkeypatch.setattr(cloudai.handlers, name, target)
    kwargs = {parameter: object() for parameter in signature.parameters}
    args = tuple(kwargs.values())
    required_args = tuple(
        value
        for parameter, value in kwargs.items()
        if signature.parameters[parameter].default is inspect.Parameter.empty
    )
    warning = rf"cloudai\.cli\.handlers\.{name} is deprecated"
    for positional, keywords in [(args, {}), ((), kwargs), (required_args, {})]:
        with pytest.warns(DeprecationWarning, match=warning) as recorded:
            assert legacy(*positional, **keywords) is target.return_value
        assert len(recorded) == 1
        assert recorded[0].filename == __file__

    assert target.call_args_list == [call(*args), call(**kwargs), call(*required_args)]

    error = RuntimeError("handler failure")
    target.side_effect = error
    with pytest.warns(DeprecationWarning, match=warning), pytest.raises(RuntimeError) as raised:
        legacy(*required_args)
    assert raised.value is error
