# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from datetime import datetime
from pathlib import Path

import pytest

pytestmark = pytest.mark.ci_only  # This test takes long time to run

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

HEADER_LINES = HEADER.split("\n")
HEADER_TAIL = "\n".join(HEADER_LINES[2:])

PY_FILES = list(Path().glob("./src/**/*.py")) + list(Path().glob("./tests/**/*.py"))
TOML_FILES = list(Path().glob("conf/**/*.toml")) + [Path("pyproject.toml"), Path(".taplo.toml")]

CURRENT_YEAR = datetime.now().year


def _format_years_to_ranges(years: list[int]) -> str:
    """Turn sorted unique years into a string with ranges for consecutive years.

    E.g. [2024, 2026] -> "2024, 2026"
         [2024, 2025, 2027] -> "2024-2025, 2027"
    """
    if not years:
        raise ValueError("Unexpected behavior. Expected at least one year in the list. If it's a new file - should be the current year")
    
    parts: list[str] = []
    range_start = years[0]
    range_end = years[0]
    for y in years[1:]:
        if y == range_end + 1:
            range_end = y
        else:
            parts.append(f"{range_start}-{range_end}" if range_start != range_end else str(range_start))
            range_start = y
            range_end = y
    parts.append(f"{range_start}-{range_end}" if range_start != range_end else str(range_start))
    return ", ".join(parts)


@pytest.mark.parametrize(
    ("years_list", "expected"),
    (
        ([2024, 2026], "2024, 2026"),
        ([2024, 2025, 2027], "2024-2025, 2027"),
        ([2024], "2024"),
        ([2024, 2025, 2026], "2024-2026"),
    ),
)
def test_format_years_to_ranges(years_list: list[int], expected: str):
    assert _format_years_to_ranges(years_list) == expected


def get_commit_years_for_file(path: Path) -> list[int]:
    """
    Return sorted list of years when the file was changed: from git log --follow
    (so renames/moves are included), plus current year if the file is new or has
    uncommitted changes (staged or unstaged).
    """
    path_str = path.as_posix()
    res = subprocess.run(
        ["git", "log", "--format=%ad", "--date=format:%Y", "--follow", "--", path_str],
        capture_output=True,
        text=True,
    )
    if not res.stdout.strip():
        res = subprocess.run(
            ["git", "log", "--format=%ad", "--date=format:%Y", "--", path_str],
            capture_output=True,
            text=True,
        )
    lines = [s.strip() for s in res.stdout.splitlines() if s.strip()]
    years = sorted(set(int(y) for y in lines)) if lines else [CURRENT_YEAR]

    # Include current year if file is new, modified (unstaged), or staged
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", path_str],
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        years = sorted(set(years) | {CURRENT_YEAR})
    return years


def prepare_copyright_with_year(years: list[int]) -> str:
    years_str = _format_years_to_ranges(years)
    after_year_str = "NVIDIA CORPORATION & AFFILIATES. All rights reserved."
    return f"# Copyright (c) {years_str} {after_year_str}"


def _assert_copyright_in_file(file: Path):
    with file.open() as f:
        try:
            actual_copyright_lines = [next(f).strip() for _ in range(len(HEADER_LINES))]
        except StopIteration:
            actual_copyright_lines = []
    
    assert len(actual_copyright_lines) >= len(HEADER_LINES), "Copyright is missing or incomplete"
    
    expected_years = get_commit_years_for_file(file)
    expected_years_line = prepare_copyright_with_year(expected_years)
    
    assert actual_copyright_lines[0] == HEADER_LINES[0], "SPDX-FileCopyrightText is not valid"
    assert actual_copyright_lines[1] == expected_years_line, "Copyright year is not valid"
    assert "\n".join(actual_copyright_lines[2:]) == HEADER_TAIL, f"Header mismatch in {file}"



@pytest.mark.parametrize("py_file", PY_FILES, ids=[str(f) for f in PY_FILES])
def test_src_copyright_header(py_file: Path):
    _assert_copyright_in_file(py_file)


@pytest.mark.parametrize("toml_file", TOML_FILES, ids=[str(f) for f in TOML_FILES])
def test_toml_copyright_header(toml_file: Path):
    _assert_copyright_in_file(toml_file)
