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
from collections.abc import Mapping
from typing import Any, Iterable

from .fio import FioCmdArgs


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _format_option(name: str, value: Any) -> list[str]:
    if value is None or value is False:
        return []

    if isinstance(value, Mapping):
        parts: list[str] = []
        for nested_value in value.values():
            parts.extend(_format_option(name, nested_value))
        return parts

    normalized = name.strip()
    flag = normalized if normalized.startswith("-") else f"--{normalized}"
    if value is True:
        return [flag]

    if isinstance(value, list):
        value = ",".join(_format_scalar(item) for item in value)

    return [f"{flag}={shlex.quote(_format_scalar(value))}"]


def build_fio_command_parts(cmd_args: FioCmdArgs) -> list[str]:
    parts = [shlex.quote(cmd_args.fio_binary)]
    for name, value in cmd_args.fio_args().items():
        parts.extend(_format_option(name, value))
    if cmd_args.job_file:
        parts.append(shlex.quote(cmd_args.job_file))
    return parts


def build_fio_command(cmd_args: FioCmdArgs, prefix: Iterable[str] | None = None) -> str:
    parts = [*(prefix or []), *build_fio_command_parts(cmd_args)]
    return " ".join(parts)
