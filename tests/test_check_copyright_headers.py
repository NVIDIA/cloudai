# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

HEADER = """# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# limitations under the License."""

HEADER_LINES = HEADER.count("\n") + 1

PY_FILES = [p for p in Path().rglob("**/*.py") if "venv" not in str(p)]
TOML_FILES = [p for p in Path().rglob("**/*.toml") if "venv" not in str(p)]


@pytest.mark.parametrize("py_file", PY_FILES, ids=[str(f) for f in PY_FILES])
def test_src_copyright_header(py_file):
    with open(py_file, "r") as file:
        first_lines = [next(file).strip() for _ in range(HEADER_LINES)]
    assert "\n".join(first_lines) == HEADER, f"Header mismatch in {py_file}"


@pytest.mark.parametrize("toml_file", TOML_FILES, ids=[str(f) for f in TOML_FILES])
def test_toml_copyright_header(toml_file):
    with open(toml_file, "r") as file:
        first_lines = [next(file).strip() for _ in range(HEADER_LINES)]
    assert "\n".join(first_lines) == HEADER, f"Header mismatch in {toml_file}"
