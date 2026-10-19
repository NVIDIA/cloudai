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

"""Load saved executions, with legacy file conversion kept at this boundary."""

from pathlib import Path
from typing import Any

import toml

from cloudai.models.workload import TestDefinition

from .test_run import TestRun


def _convert_legacy_run(data: dict[str, Any], fallback: TestDefinition | None) -> dict[str, Any]:
    """Translate the old TestRunDetails layout into TestRun fields."""
    definition = fallback.model_dump() if fallback is not None else {}
    definition.update(data["test_definition"])
    definition.update(id=data["name"], num_nodes=data["nnodes"], iterations=data["iterations"])
    return {
        "test": definition,
        "nodes": data["nodes"],
        "output_path": data["output_path"],
        "current_iteration": data["current_iteration"],
        "step": data["step"],
    }


def load_test_run(path: Path, fallback: TestDefinition | None = None) -> TestRun:
    """
    Read a saved execution; use scenario defaults only for incomplete legacy definitions.

    The containing folder determines the output path so copied result folders remain usable.
    Reporter registrations come from the supplied definition, never from the saved file.
    """
    with path.open() as stream:
        data = toml.load(stream)
    if "test_definition" in data:
        data = _convert_legacy_run(data, fallback)
    run = TestRun.model_validate(data)
    run.output_path = path.parent
    if fallback is not None:
        run.test = run.test.model_copy(update={"reporters": fallback.reporters})
    return run
