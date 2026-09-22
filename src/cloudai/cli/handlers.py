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

import argparse
import functools
import typing
import warnings

import cloudai.cli.cli
import cloudai.core
import cloudai.handlers
import cloudai.test_parser
import cloudai.toml_utils

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
        return function(*args, **kwargs)

    return wrapper


handle_install_and_uninstall = _deprecated(cloudai.cli.cli.handle_install_and_uninstall)
prepare_installation = _deprecated(cloudai.handlers.prepare_installation)
generate_reports = _deprecated(cloudai.handlers.generate_reports)
register_signal_handlers = _deprecated(cloudai.cli.cli.register_signal_handlers)
handle_dry_run_and_run = _deprecated(cloudai.cli.cli.handle_dry_run_and_run)
handle_generate_report = _deprecated(cloudai.cli.cli.handle_generate_report)
expand_file_list = _deprecated(cloudai.cli.cli.expand_file_list)
verify_system_configs = _deprecated(cloudai.cli.cli.verify_system_configs)
verify_test_configs = _deprecated(cloudai.cli.cli.verify_test_configs)
verify_test_scenarios = _deprecated(cloudai.cli.cli.verify_test_scenarios)
handle_verify_all_configs = _deprecated(cloudai.cli.cli.handle_verify_all_configs)
load_tomls_by_type = _deprecated(cloudai.cli.cli.load_tomls_by_type)
handle_list_registered_items = _deprecated(cloudai.cli.cli.handle_list_registered_items)
validate_domain_randomization_active = _deprecated(cloudai.handlers.validate_domain_randomization_active)
load_test_toml_file = _deprecated(cloudai.test_parser.load_test_toml_file)
format_toml_decode_error = _deprecated(cloudai.toml_utils.format_toml_decode_error)
prepare_output_dir = _deprecated(cloudai.handlers.prepare_output_dir)


@_deprecated
def handle_dse_job(runner: cloudai.core.Runner, args: argparse.Namespace) -> int:
    return cloudai.handlers.handle_dse_job(runner, args.mode)


@_deprecated
def handle_non_dse_job(runner: cloudai.core.Runner, args: argparse.Namespace) -> bool:
    return cloudai.handlers.handle_non_dse_job(runner)
