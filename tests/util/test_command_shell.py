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

import pytest

from cloudai.util.command_shell import CommandShell


@pytest.mark.parametrize("executable", ["/bin/bash", "/bin/sh"])
def test_execute_preserves_shell_syntax_and_process_results(executable: str):
    with CommandShell(Path(executable)).execute(
        "value='hello world'; printf '%s' \"$value\" | tr '[:lower:]' '[:upper:]'; printf 'error' >&2; exit 7"
    ) as process:
        stdout, stderr = process.communicate(timeout=10)

    assert stdout == "HELLO WORLD"
    assert stderr == "error"
    assert process.returncode == 7
