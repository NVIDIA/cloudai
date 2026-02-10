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
from collections import defaultdict
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


def _path_key(file: Path) -> str:
    """Normalize path for cache lookup (e.g. ./src/foo.py and src/foo.py match)."""
    return file.as_posix().lstrip("./")


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


def test_format_years_to_ranges():
    """Check year range formatting: consecutive years become start-end, non-consecutive stay separate."""
    assert _format_years_to_ranges([2024, 2026]) == "2024, 2026"
    assert _format_years_to_ranges([2024, 2025, 2027]) == "2024-2025, 2027"
    assert _format_years_to_ranges([2024]) == "2024"
    assert _format_years_to_ranges([2024, 2025, 2026]) == "2024-2026"
    assert _format_years_to_ranges([]) == ""


@pytest.fixture(scope="session")
def git_commit_years_cache() -> dict[str, list[int]]:
    """
    Map all git-tracked files in this repo to the sorted list of years when they were touched in a commit

    Example:
        >>> git_commit_years_cache()
        {"src/cloudai/core.py": [2024, 2025, 2026]}
    """

    # 1: get all tracked files
    ls_out = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    files = ls_out.stdout.splitlines()

    # 2. Stream full history: each commit has a year line then affected files
    log_cmd = [
        "git",
        "-c",
        "core.quotepath=false",
        "log",
        "--format=Y:%ad",
        "--date=format:%Y",
        "--name-only",
        "--",
        "src/",
        "tests/",
        "conf/",
        "pyproject.toml",
        ".taplo.toml",
    ]
    file_years: dict[str, set[int]] = defaultdict(set)

    process = subprocess.Popen(log_cmd, stdout=subprocess.PIPE, text=True)
    if process.stdout is None:
        raise OSError("git subprocess: stdout wasn't initialized for the subprocess")
    
    with process:
        current_year: int | None = None
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Y:"):
                current_year = int(line[2:])
            elif current_year is not None and line in files:
                file_years[line].add(current_year)
    
    if process.returncode != 0:
        raise RuntimeError("git subprocess: unknown error, check output")

    return {path: sorted(years) for path, years in file_years.items()}


def prepare_copyright_with_year(file: Path, years_cache: dict[str, list[int]]) -> str:
    path_key = file.as_posix().replace("./", "", 1)
    years = years_cache.get(path_key, [CURRENT_YEAR])
    years_str = _format_years_to_ranges(years)
    after_year_str = "NVIDIA CORPORATION & AFFILIATES. All rights reserved."
    return f"# Copyright (c) {years_str} {after_year_str}"


@pytest.mark.parametrize("py_file", PY_FILES, ids=[str(f) for f in PY_FILES])
def test_src_copyright_header(py_file: Path, git_commit_years_cache: dict[str, list[int]]):
    with py_file.open() as file:
        actual_copyright_lines = [next(file).strip() for _ in range(len(HEADER_LINES))]

    assert actual_copyright_lines[0] == HEADER_LINES[0], "SPDX-FileCopyrightText is not valid"
    assert actual_copyright_lines[1] == prepare_copyright_with_year(py_file, git_commit_years_cache), "Copyright year is not valid"
    assert "\n".join(actual_copyright_lines[2:]) == HEADER_TAIL, f"Header mismatch in {py_file}"


@pytest.mark.parametrize("toml_file", TOML_FILES, ids=[str(f) for f in TOML_FILES])
def test_toml_copyright_header(toml_file: Path, git_commit_years_cache: dict[str, list[int]]):
    with toml_file.open() as file:
        actual_copyright_lines = [next(file).strip() for _ in range(len(HEADER_LINES))]

    assert actual_copyright_lines[0] == HEADER_LINES[0], "SPDX-FileCopyrightText is not valid"
    assert actual_copyright_lines[1] == prepare_copyright_with_year(toml_file, git_commit_years_cache), "Copyright year is not valid"
    assert "\n".join(actual_copyright_lines[2:]) == HEADER_TAIL, f"Header mismatch in {toml_file}"
