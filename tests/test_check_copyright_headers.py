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

import subprocess
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

PY_FILES = [p for p in Path().glob("./src/**/*.py")]
PY_FILES += [p for p in Path().glob("./tests/**/*.py")]
TOML_FILES = [p for p in Path().glob("conf/**/*.toml")] + ["pyproject.toml", ".taplo.toml"]


def prepare_copyright_with_year(file: Path, line: str) -> str:
    res = subprocess.run(
        ["git", "log", "--format=%ad", "--date=format:%Y", "--follow", "-1", file],
        capture_output=True,
        text=True,
    )
    curr_year_spec = line.split(" ")[3]
    spec_is_range = "-" in curr_year_spec

    changed_years = res.stdout.splitlines()

    last_modified_year_real = int(changed_years[0])

    after_year_str = "NVIDIA CORPORATION & AFFILIATES. All rights reserved."

    if spec_is_range:
        created_year = int(curr_year_spec.split("-")[0])
        return f"# Copyright (c) {created_year}-{last_modified_year_real} {after_year_str}"

    if int(curr_year_spec) < last_modified_year_real:
        return f"# Copyright (c) {curr_year_spec}-{last_modified_year_real} {after_year_str}"

    return f"# Copyright (c) {last_modified_year_real} {after_year_str}"


@pytest.mark.parametrize("py_file", PY_FILES, ids=[str(f) for f in PY_FILES])
def test_src_copyright_header(py_file: Path):
    with py_file.open() as file:
        first_lines = [next(file).strip() for _ in range(HEADER_LINES)]

    assert (
        first_lines[0] == "# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES"
    ), "SPDX-FileCopyrightText is not valid"
    assert first_lines[1] == prepare_copyright_with_year(py_file, first_lines[1]), "Copyright year is not valid"
    assert "\n".join(first_lines[2:]) == "\n".join(HEADER.splitlines()[2:]), f"Header mismatch in {py_file}"


@pytest.mark.parametrize("toml_file", TOML_FILES, ids=[str(f) for f in TOML_FILES])
def test_toml_copyright_header(toml_file):
    with open(toml_file, "r") as file:
        first_lines = [next(file).strip() for _ in range(HEADER_LINES)]

    assert first_lines[0] == "# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES"
    assert first_lines[1] == prepare_copyright_with_year(toml_file, first_lines[1])
    assert "\n".join(first_lines[2:]) == "\n".join(HEADER.splitlines()[2:]), f"Header mismatch in {toml_file}"
