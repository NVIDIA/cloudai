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

from __future__ import annotations

import shlex
from typing import Iterable

from .fio import FioArgValue, FioCmdArgs


def _format_scalar(value: bool | int | float | str) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _format_option(name: str, value: FioArgValue) -> list[str]:
    if value is None or value is False:
        return []

    normalized = name.strip().replace("_", "-")
    flag = normalized if normalized.startswith("-") else f"--{normalized}"
    if value is True:
        return [flag]

    if isinstance(value, list):
        value = ",".join(_format_scalar(item) for item in value)

    return [f"{flag}={shlex.quote(_format_scalar(value))}"]


def build_fio_command_parts(cmd_args: FioCmdArgs) -> list[str]:
    parts = [shlex.quote(cmd_args.fio_binary)]
    for name, value in cmd_args.args.items():
        parts.extend(_format_option(name, value))
    parts.extend(cmd_args.passthrough_args)
    if cmd_args.job_file:
        parts.append(shlex.quote(cmd_args.job_file))
    return parts


def build_fio_command(cmd_args: FioCmdArgs, prefix: Iterable[str] | None = None) -> str:
    parts = [*(prefix or []), *build_fio_command_parts(cmd_args)]
    return " ".join(parts)
