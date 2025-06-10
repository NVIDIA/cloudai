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

import logging
import os
from pathlib import Path
from typing import Optional

from .command_shell import CommandShell
from .utils import format_time_limit, parse_time_limit


def prepare_output_dir(path: Path) -> Optional[Path]:
    exists = False
    try:
        exists = path.exists()
    except PermissionError as e:
        logging.error(f"Output path '{path.absolute()}' is not accessible: {e}")
        return None

    if exists:
        if not os.access(path, os.W_OK):
            logging.error(f"Output path '{path.absolute()}' exists but is not writable.")
            return None
        if not path.is_dir():
            logging.error(f"Output path '{path.absolute()}' exists but is not a directory.")
            return None
        return path

    path.mkdir(parents=True)
    return path


__all__ = [
    "CommandShell",
    "format_time_limit",
    "parse_time_limit",
    "prepare_output_dir",
]
