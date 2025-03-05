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


def prepare_copyright_with_year(file: Path) -> str:
    res = subprocess.run(
        ["git", "log", "--format=%ad", "--date=format:%Y", "--follow", file],
        capture_output=True,
        text=True,
    )
    changed_years = res.stdout.splitlines()
    created_year, last_modified_year = int(changed_years[-1]), int(changed_years[0])
    if created_year == last_modified_year:
        return f"# Copyright (c) {created_year} NVIDIA CORPORATION & AFFILIATES. All rights reserved."

    return f"# Copyright (c) {created_year}-{last_modified_year} NVIDIA CORPORATION & AFFILIATES. All rights reserved."


@pytest.mark.parametrize("py_file", PY_FILES, ids=[str(f) for f in PY_FILES])
def test_src_copyright_header(py_file: Path):
    with py_file.open() as file:
        first_lines = [next(file).strip() for _ in range(HEADER_LINES)]

    assert (
        first_lines[0] == "# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES"
    ), "SPDX-FileCopyrightText is not valid"
    assert first_lines[1] == prepare_copyright_with_year(py_file), "Copyright year is not valid"
    assert "\n".join(first_lines[2:]) == "\n".join(HEADER.splitlines()[2:]), f"Header mismatch in {py_file}"


@pytest.mark.parametrize("toml_file", TOML_FILES, ids=[str(f) for f in TOML_FILES])
def test_toml_copyright_header(toml_file):
    with open(toml_file, "r") as file:
        first_lines = [next(file).strip() for _ in range(HEADER_LINES)]

    assert first_lines[0] == "# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES"
    assert first_lines[1] == prepare_copyright_with_year(toml_file)
    assert "\n".join(first_lines[2:]) == "\n".join(HEADER.splitlines()[2:]), f"Header mismatch in {toml_file}"
