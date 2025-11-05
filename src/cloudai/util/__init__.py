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
from typing import Any, Optional

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
            logging.error(f"Output path '{path.absolute()}' does not exist.")
            return False
        if not parent.is_dir():
            logging.error(f"Output path '{path.absolute()}' is not a directory.")
            return False
        if not os.access(parent, os.W_OK):
            logging.error(f"Output path '{path.absolute()}' is not writable.")
            return False
        return True
    except PermissionError as e:
        logging.error(f"Output path '{path.absolute()}' is not accessible: {e}")
        return False


def _validate_existing_path(path: Path) -> Optional[Path]:
    if not path.is_dir():
        logging.error(f"Output path '{path.absolute()}' exists but is not a directory.")
        return None
    if not os.access(path, os.W_OK):
        logging.error(f"Output path '{path.absolute()}' exists but is not writable.")
        return None
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
            logging.error(
                f"Failed to create directory '{resolved_path.absolute()}': {e}. "
                "Please check directory permissions and available disk space."
            )
            return None
    except PermissionError:
        logging.error(
            f"Cannot access path '{resolved_path.absolute()}': Permission denied. Please check directory permissions."
        )
        return None
    except OSError as e:
        logging.error(f"Cannot access path '{resolved_path.absolute()}': {e}")
        return None


def deep_merge(a: dict, b: dict) -> dict:
    result = a.copy()
    for key in b:
        if key in result:
            if isinstance(result[key], dict) and isinstance(b[key], dict):
                result[key] = deep_merge(result[key], b[key])
            else:
                result[key] = b[key]
        else:
            result[key] = b[key]
    return result


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """
    Flatten a nested dictionary into a single level dictionary with dot-separated keys.

    Args:
        d (Dict[str, Any]): The dictionary to flatten.
        parent_key (str): The base key for recursion (used internally).
        sep (str): Separator used between keys.

    Returns:
        Dict[str, Any]: Flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


__all__ = [
    "CommandShell",
    "deep_merge",
    "flatten_dict",
    "format_time_limit",
    "parse_time_limit",
    "prepare_output_dir",
]
