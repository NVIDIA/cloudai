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

import functools
import typing
import warnings

import cloudai.handlers

_P = typing.ParamSpec("_P")
_R = typing.TypeVar("_R")


def _deprecated(function: typing.Callable[_P, _R]) -> typing.Callable[_P, _R]:
    @functools.wraps(function)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        warnings.warn(
            f"cloudai.cli.handlers.{function.__name__} is deprecated; do not call CLI handlers directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(cloudai.handlers, function.__name__)(*args, **kwargs)

    return wrapper


_log_installation_dirs = _deprecated(cloudai.handlers._log_installation_dirs)
handle_install_and_uninstall = _deprecated(cloudai.handlers.handle_install_and_uninstall)
prepare_installation = _deprecated(cloudai.handlers.prepare_installation)
_scenario_installables = _deprecated(cloudai.handlers._scenario_installables)
handle_dse_job = _deprecated(cloudai.handlers.handle_dse_job)
_record_run_failure = _deprecated(cloudai.handlers._record_run_failure)
generate_reports = _deprecated(cloudai.handlers.generate_reports)
handle_non_dse_job = _deprecated(cloudai.handlers.handle_non_dse_job)
register_signal_handlers = _deprecated(cloudai.handlers.register_signal_handlers)
_setup_system_and_scenario = _deprecated(cloudai.handlers._setup_system_and_scenario)
_handle_single_sbatch = _deprecated(cloudai.handlers._handle_single_sbatch)
_check_installation = _deprecated(cloudai.handlers._check_installation)
handle_dry_run_and_run = _deprecated(cloudai.handlers.handle_dry_run_and_run)
handle_generate_report = _deprecated(cloudai.handlers.handle_generate_report)
expand_file_list = _deprecated(cloudai.handlers.expand_file_list)
_ensure_kube_config_exists = _deprecated(cloudai.handlers._ensure_kube_config_exists)
verify_system_configs = _deprecated(cloudai.handlers.verify_system_configs)
verify_test_configs = _deprecated(cloudai.handlers.verify_test_configs)
verify_test_scenarios = _deprecated(cloudai.handlers.verify_test_scenarios)
handle_verify_all_configs = _deprecated(cloudai.handlers.handle_verify_all_configs)
load_tomls_by_type = _deprecated(cloudai.handlers.load_tomls_by_type)
handle_list_registered_items = _deprecated(cloudai.handlers.handle_list_registered_items)
validate_domain_randomization_active = _deprecated(cloudai.handlers.validate_domain_randomization_active)
load_test_toml_file = _deprecated(cloudai.handlers.load_test_toml_file)
format_toml_decode_error = _deprecated(cloudai.handlers.format_toml_decode_error)
prepare_output_dir = _deprecated(cloudai.handlers.prepare_output_dir)
