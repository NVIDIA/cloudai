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

PY_FILES = [*list(Path().glob("./src/**/*.py")), *list(Path().glob("./tests/**/*.py"))]
TOML_FILES = [*list(Path().glob("conf/**/*.toml")), Path("pyproject.toml"), Path(".taplo.toml")]

CURRENT_YEAR = datetime.now().year

_REC_SEP = "\x1e"


def _format_years_to_ranges(years: list[int]) -> str:
    """Turn sorted unique years into a string with ranges for consecutive years.

    E.g. [2024, 2026] -> "2024, 2026"
         [2024, 2025, 2027] -> "2024-2025, 2027"
    """
    if not years:
        raise ValueError(
            "Unexpected behavior. Expected at least one year in the list. If it's a new file - should be the "
            "current year"
        )

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


def test_empty_format_years_to_ranges():
    with pytest.raises(ValueError):
        _format_years_to_ranges([])


def run_git(cmd: list[str]) -> str:
    res = subprocess.run(["git", *cmd], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"git command {cmd[0]} failed: {res.stderr}")
    return res.stdout


def collect_years_same_file(path: Path) -> list[int]:
    """
    Using --follow but only following file movements (not copying) with more than 92% similarity.

    92% appears when one moves an empty __init__.py to another folder and updates years in the license header.
    """
    git_output = run_git(
        [
            "log",
            "--follow",
            "--name-status",
            "--find-renames=92%",
            "--format=%x1e%ad",
            "--date=format:%Y",
            "--",
            path.as_posix(),
        ],
    )

    years: set[int] = set()
    current_path = path.as_posix()

    for rec in git_output.split(_REC_SEP):
        rec = rec.strip()
        if not rec:
            continue

        lines = [ln.strip() for ln in rec.splitlines() if ln.strip()]

        year = int(lines[0])
        parts = lines[1].split("\t")
        commit_status = parts[0]

        # Follow only exact rename hops for the currently tracked path.
        if len(commit_status) == 4 and commit_status[0] == "R" and len(parts) == 3:
            percentage = int(commit_status[1:])
            old_path, new_path = parts[1], parts[2]
            if percentage >= 92 and new_path == current_path:
                current_path = old_path
                years.add(year)
                continue

        # Normal touch of current path.
        elif len(parts) >= 2 and parts[1] == current_path:
            years.add(year)
            continue

        break

    git_status = run_git(["status", "--porcelain", "--", path.as_posix()])
    if git_status.strip():
        years.add(CURRENT_YEAR)

    return sorted(years)


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

    expected_years = collect_years_same_file(file)
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
