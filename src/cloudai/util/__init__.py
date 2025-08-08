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
import sys
from pathlib import Path
from typing import Optional

from .command_shell import CommandShell
from .utils import format_time_limit, parse_time_limit


def _validate_path_format(path: Path) -> Optional[Path]:
    try:
        return path.resolve()
    except (RuntimeError, OSError) as e:
        logging.error(f"Invalid path format '{path}': {e}")
        return None


def _validate_parent_dir(path: Path, parent: Path) -> bool:
    try:
        if not parent.exists():
            msg = f"{path} does not exist."
            logging.error(msg)
            sys.exit(1)
        if not parent.is_dir():
            msg = f"{path} is not a directory."
            logging.error(msg)
            sys.exit(1)
        if not os.access(parent, os.W_OK):
            msg = f"{path} is not writable."
            logging.error(msg)
            sys.exit(1)
        return True
    except PermissionError:
        msg = f"Cannot access {path}: Permission denied."
        logging.error(msg)
        sys.exit(1)


def _validate_existing_path(path: Path) -> Optional[Path]:
    if not path.is_dir():
        msg = f"{path} is not a directory."
        logging.error(msg)
        sys.exit(1)
    if not os.access(path, os.W_OK):
        msg = f"{path} is not writable."
        logging.error(msg)
        sys.exit(1)
    return path


def prepare_output_dir(path: Path) -> Optional[Path]:
    resolved_path = _validate_path_format(path)
    if resolved_path is None:
        return None

    if not _validate_parent_dir(path, resolved_path.parent):
        return None

    try:
        if resolved_path.exists():
            return _validate_existing_path(resolved_path)

        # Path doesn't exist, try to create it
        try:
            resolved_path.mkdir(parents=True)
            return resolved_path
        except OSError as e:
            msg = (
                f"Failed to create directory '{resolved_path.absolute()}': {e}. "
                "Please check directory permissions and available disk space."
            )
            logging.error(msg)
            sys.exit(1)
    except PermissionError:
        logging.error(
            f"Cannot access path '{resolved_path.absolute()}': Permission denied. Please check directory permissions."
        )
        return None
    except OSError as e:
        logging.error(f"Cannot access path '{resolved_path.absolute()}': {e}")
        return None


__all__ = [
    "CommandShell",
    "format_time_limit",
    "parse_time_limit",
    "prepare_output_dir",
]
