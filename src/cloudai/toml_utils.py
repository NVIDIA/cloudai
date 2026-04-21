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

from pathlib import Path

import toml


def format_toml_decode_error(file_path: Path, error: toml.TomlDecodeError, config_label: str) -> str:
    if error.msg == "Duplicate keys!":
        duplicate_key = extract_duplicate_key(error.doc, error.lineno)
        return (
            f"Failed to parse {config_label} '{file_path}': duplicate TOML key "
            f"'{duplicate_key}' at line {error.lineno}, column {error.colno}."
        )

    return (
        f"Failed to parse {config_label} '{file_path}': TOML parsing error "
        f"at line {error.lineno}, column {error.colno}: {error.msg}"
    )


def extract_duplicate_key(doc: str, lineno: int) -> str:
    key = _extract_duplicate_key(doc, lineno)
    return key or "<unknown>"


def _extract_duplicate_key(doc: str, lineno: int) -> str | None:
    lines = doc.splitlines()
    if lineno < 1 or lineno > len(lines):
        return None

    line = lines[lineno - 1].split("#", 1)[0].strip()
    if not line or "=" not in line or line.startswith("["):
        return None

    key = line.split("=", 1)[0].strip()
    return key or None
