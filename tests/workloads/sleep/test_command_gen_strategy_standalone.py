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

from cloudai.core import CommandGenStrategy, TestRun
from cloudai.models.scenario import TestRunDetails
from cloudai.systems.standalone import StandaloneSystem
from cloudai.workloads.sleep import SleepCmdArgs, SleepStandaloneCommandGenStrategy, SleepTestDefinition


def test_gen_exec_command_writes_test_run_details(tmp_path: Path, standalone_system: StandaloneSystem) -> None:
    tdef = SleepTestDefinition(
        name="sleep_test",
        description="Simple sleep test",
        test_template_name="Sleep",
        cmd_args=SleepCmdArgs(seconds=60),
    )
    tr = TestRun(name="sleep-job", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path / "output")

    strategy = SleepStandaloneCommandGenStrategy(standalone_system, tr)
    command = strategy.gen_exec_command()

    assert command == "sleep 60"

    dump_path = tr.output_path / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME
    assert dump_path.is_file()
    details = TestRunDetails.model_validate(toml.load(dump_path))
    assert details.test_cmd == "sleep 60"
    assert details.full_cmd == "sleep 60"
