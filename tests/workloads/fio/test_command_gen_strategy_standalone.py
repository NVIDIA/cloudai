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
from cloudai.workloads.fio import FioCmdArgs, FioStandaloneCommandGenStrategy, FioTestDefinition


def test_gen_exec_command_redirects_output_and_writes_test_run(
    standalone_system: StandaloneSystem, tmp_path: Path
) -> None:
    tdef = FioTestDefinition(
        name="fio",
        description="fio test",
        test_template_name="Fio",
        cmd_args=FioCmdArgs(
            fio_binary="fio",
            args={"name": "smoke", "filename": "/tmp/file", "rw": "write", "bs": "128k"},
        ),
        extra_env_vars={"FIO_VAR": "1"},
    )
    tr = TestRun(name="fio", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path / "output")
    strategy = FioStandaloneCommandGenStrategy(standalone_system, tr)

    command = strategy.gen_exec_command()

    assert command.endswith(f"> {tr.output_path / 'stdout.txt'} 2> {tr.output_path / 'stderr.txt'}")
    assert 'export FIO_VAR="1"; fio --name=smoke --filename=/tmp/file --rw=write --bs=128k' in command

    dump_path = tr.output_path / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME
    assert dump_path.is_file()
    details = TestRunDetails.model_validate(toml.load(dump_path))
    assert details.test_cmd == 'export FIO_VAR="1"; fio --name=smoke --filename=/tmp/file --rw=write --bs=128k'
    assert details.full_cmd == command
