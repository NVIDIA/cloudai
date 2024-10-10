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
from typing import Dict

import pytest
from cloudai import Test, TestDefinition, TestRun, TestTemplate
from cloudai.schema.test_template.jax_toolbox.slurm_command_gen_strategy import JaxToolboxSlurmCommandGenStrategy
from cloudai.systems import SlurmSystem
from pydantic import BaseModel


class CmdArgsModel(BaseModel):
    test_setup_flags: Dict[str, str]
    output_path: str


class TestJaxToolboxSlurmCommandGenStrategy:
    @pytest.fixture
    def test_run(self, tmp_path: Path, slurm_system: SlurmSystem) -> TestRun:
        cmd_args = CmdArgsModel(
            test_setup_flags={"gpus_per_node": "4", "docker_workspace_dir": "/workspace"},
            output_path="/mock/output",
        )
        test_definition = TestDefinition(
            name="test1",
            description="Jax Toolbox test",
            test_template_name="jax_toolbox_test_template",
            cmd_args=cmd_args,
            extra_env_vars={"COMBINE_THRESHOLD": "1024"},
            extra_cmd_args={"max_steps": "100"},
        )
        test_template = TestTemplate(system=slurm_system, name="jax_toolbox_test_template")

        test = Test(test_definition=test_definition, test_template=test_template)

        tr = TestRun(
            test=test,
            num_nodes=2,
            nodes=["node1", "node2"],
            output_path=tmp_path / "output",
            name="test-job",
        )
        return tr

    @pytest.fixture
    def cmd_gen_strategy(self, slurm_system: SlurmSystem) -> JaxToolboxSlurmCommandGenStrategy:
        return JaxToolboxSlurmCommandGenStrategy(slurm_system, {})

    def test_gen_exec_command(self, cmd_gen_strategy: JaxToolboxSlurmCommandGenStrategy, test_run: TestRun):
        generated_command = cmd_gen_strategy.gen_exec_command(test_run)

        assert "srun" in generated_command
        assert "python3 -u -m paxml.main" in generated_command
        assert "--num_hosts=$SLURM_NTASKS" in generated_command
        assert "--fdl_config" in generated_command
        assert "PER_GPU_COMBINE_THRESHOLD" in generated_command
        assert "output_path" in generated_command

    def test_parse_slurm_args(self, cmd_gen_strategy: JaxToolboxSlurmCommandGenStrategy):
        env_vars = {"COMBINE_THRESHOLD": "1024"}
        cmd_args = {
            "test.setup_flags": {"docker_workspace_dir": "/workspace"},
            "output_path": "/mock/output",
        }
        num_nodes = 2
        nodes = ["node1", "node2"]

        slurm_args = cmd_gen_strategy._parse_slurm_args("JaxToolbox", env_vars, cmd_args, num_nodes, nodes)

        assert "container_mounts" in slurm_args
        assert "output" in slurm_args
        assert "error" in slurm_args
        assert slurm_args["container_mounts"] == "/mock/output:/workspace"

    def test_format_xla_flags(self, cmd_gen_strategy: JaxToolboxSlurmCommandGenStrategy):
        cmd_args = {
            "test.perf": {
                "XLA_FLAGS": {"xla_gpu_autotune_level": "2", "xla_gpu_all_reduce_combine_threshold_bytes": "2048"}
            }
        }
        stage = "perf"
        formatted_flags = cmd_gen_strategy._format_xla_flags(cmd_args, stage)

        assert "--xla_gpu_all_reduce_combine_threshold_bytes=$COMBINE_THRESHOLD" in formatted_flags
        assert "--xla_gpu_autotune_level=2" in formatted_flags
        assert "--xla_gpu_all_reduce_combine_threshold_bytes=2048" in formatted_flags

    def test_create_run_script(self, cmd_gen_strategy: JaxToolboxSlurmCommandGenStrategy, test_run: TestRun):
        env_vars = {"COMBINE_THRESHOLD": "1024"}
        cmd_args = {
            "test.setup_flags": {"docker_workspace_dir": "/workspace"},
            "output_path": "/mock/output",
        }
        extra_cmd_args = "--max-steps 100"

        run_script_path = cmd_gen_strategy._create_run_script(env_vars, cmd_args, extra_cmd_args)

        assert run_script_path == Path("/mock/output/run.sh")
        assert run_script_path.exists()
