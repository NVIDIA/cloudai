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

import re
from datetime import timedelta


def parse_time_limit(limit: str) -> timedelta:
    try:
        if re.match(r"^\d+[smhdw]$", limit, re.IGNORECASE):
            return parse_abbreviated_time(limit)
        if "-" in limit:
            return parse_dashed_time(limit)
        if len(limit.split(":")) == 3:
            hours, minutes, seconds = map(int, limit.split(":"))
            return timedelta(hours=hours, minutes=minutes, seconds=seconds)
        if len(limit.split(":")) == 2:
            hours, minutes = map(int, limit.split(":"))
            return timedelta(hours=hours, minutes=minutes)
    except ValueError as err:
        raise ValueError(f"Invalid time limit format: {limit}. Refer to SLURM time format documentation.") from err

    raise ValueError(f"Unsupported time limit format: {limit}. Refer to SLURM time format documentation.")


def parse_abbreviated_time(limit: str) -> timedelta:
    value, unit = int(limit[:-1]), limit[-1].lower()
    if unit == "s":
        return timedelta(seconds=value)
    if unit == "m":
        return timedelta(minutes=value)
    if unit == "h":
        return timedelta(hours=value)
    if unit == "d":
        return timedelta(days=value)
    if unit == "w":
        return timedelta(weeks=value)
    raise ValueError(f"Invalid abbreviated time format: {limit}")


def parse_dashed_time(limit: str) -> timedelta:
    days, time_part = limit.split("-", 1)
    hours, minutes, seconds = map(int, time_part.split(":"))
    return timedelta(days=int(days), hours=hours, minutes=minutes, seconds=seconds)


def format_time_limit(total_time: timedelta) -> str:
    total_seconds = int(total_time.total_seconds())
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days > 0:
        return f"{days}-{hours:02}:{minutes:02}:{seconds:02}"
    return f"{hours:02}:{minutes:02}:{seconds:02}"
