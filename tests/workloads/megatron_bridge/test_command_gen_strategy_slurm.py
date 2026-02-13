# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import cast

import pytest

from cloudai.core import TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.megatron_bridge import (
    MegatronBridgeCmdArgs,
    MegatronBridgeSlurmCommandGenStrategy,
    MegatronBridgeTestDefinition,
)


class TestMegatronBridgeSlurmCommandGenStrategy:
    @pytest.fixture
    def test_run(self, tmp_path: Path) -> TestRun:
        # Create a fake local sqsh so container path selection can use it verbatim.
        sqsh = tmp_path / "img.sqsh"
        sqsh.write_text("x")

        args = MegatronBridgeCmdArgs(
            container_image=str(sqsh),
            hf_token="dummy_token",
            model_name="qwen3",
            model_size="30b_a3b",
            cuda_graph_scope="[moe_router,moe_preprocess]",
            compute_dtype="fp8_mx",
            num_gpus=8,
            gpus_per_node=4,
        )
        tdef = MegatronBridgeTestDefinition(
            name="mb",
            description="desc",
            test_template_name="MegatronBridge",
            cmd_args=args,
            extra_container_mounts=[],
            git_repos=[
                {
                    "url": "https://github.com/NVIDIA-NeMo/Megatron-Bridge.git",
                    "commit": "r0.2.0",
                    "mount_as": "/opt/Megatron-Bridge",
                }
            ],  # type: ignore[arg-type]
        )

        # Fake installed paths for installables so command-gen doesn't depend on real installs.
        (tmp_path / "run_repo").mkdir()
        (tmp_path / "run_venv").mkdir()
        (tmp_path / "mbridge_repo").mkdir()
        tdef.python_executable.git_repo.installed_path = tmp_path / "run_repo"
        tdef.python_executable.venv_path = tmp_path / "run_venv"
        tdef.megatron_bridge_repo.installed_path = tmp_path / "mbridge_repo"
        tdef.docker_image.installed_path = tmp_path / "cached.sqsh"

        return TestRun(
            test=tdef,
            name="tr",
            num_nodes=2,
            nodes=[],
            output_path=tmp_path / "out",
        )

    @pytest.fixture
    def cmd_gen(self, slurm_system: SlurmSystem, test_run: TestRun) -> MegatronBridgeSlurmCommandGenStrategy:
        slurm_system.account = "acct"
        slurm_system.default_partition = "gb300"
        return MegatronBridgeSlurmCommandGenStrategy(slurm_system, test_run)

    def test_hf_token_empty_is_rejected_by_schema(self) -> None:
        with pytest.raises(Exception, match=r"hf_token"):
            MegatronBridgeCmdArgs.model_validate({"hf_token": "", "model_name": "qwen3", "model_size": "30b_a3b"})

    @pytest.mark.parametrize("field_name", ["model_name", "model_size"])
    def test_model_fields_empty_string_rejected(self, field_name: str) -> None:
        data = {"hf_token": "dummy_token", "model_name": "qwen3", "model_size": "30b_a3b"}
        data[field_name] = ""
        with pytest.raises(Exception, match=field_name):
            MegatronBridgeCmdArgs.model_validate(data)

    @pytest.mark.parametrize("field_name", ["model_name", "model_size"])
    def test_model_fields_whitespace_only_rejected(self, field_name: str) -> None:
        data = {"hf_token": "dummy_token", "model_name": "qwen3", "model_size": "30b_a3b"}
        data[field_name] = "   \t  "
        with pytest.raises(Exception, match=rf"cmd_args\.{field_name} cannot be empty\."):
            MegatronBridgeCmdArgs.model_validate(data)

    def test_git_repos_can_pin_megatron_bridge_commit(self) -> None:
        args = MegatronBridgeCmdArgs(hf_token="dummy_token", model_name="qwen3", model_size="30b_a3b")
        tdef = MegatronBridgeTestDefinition(
            name="mb",
            description="desc",
            test_template_name="MegatronBridge",
            cmd_args=args,
            extra_container_mounts=[],
            git_repos=[
                {
                    "url": "https://github.com/NVIDIA-NeMo/Megatron-Bridge.git",
                    "commit": "abcdef1234567890",
                    "mount_as": "/opt/Megatron-Bridge",
                }
            ],  # type: ignore[arg-type]
        )
        assert tdef.megatron_bridge_repo.commit == "abcdef1234567890"

    def test_defaults_not_emitted_when_not_set_in_toml(self, slurm_system: SlurmSystem, tmp_path: Path) -> None:
        sqsh = tmp_path / "img.sqsh"
        sqsh.write_text("x")

        args = MegatronBridgeCmdArgs(
            container_image=str(sqsh),
            hf_token="dummy_token",
            model_name="qwen3",
            model_size="30b_a3b",
            num_gpus=8,
            gpus_per_node=4,
        )
        tdef = MegatronBridgeTestDefinition(
            name="mb",
            description="desc",
            test_template_name="MegatronBridge",
            cmd_args=args,
            extra_container_mounts=[],
            git_repos=[
                {
                    "url": "https://github.com/NVIDIA-NeMo/Megatron-Bridge.git",
                    "commit": "r0.2.0",
                    "mount_as": "/opt/Megatron-Bridge",
                }
            ],  # type: ignore[arg-type]
        )

        (tmp_path / "run_repo").mkdir()
        (tmp_path / "run_venv").mkdir()
        (tmp_path / "mbridge_repo").mkdir()
        tdef.python_executable.git_repo.installed_path = tmp_path / "run_repo"
        tdef.python_executable.venv_path = tmp_path / "run_venv"
        tdef.megatron_bridge_repo.installed_path = tmp_path / "mbridge_repo"
        tdef.docker_image.installed_path = tmp_path / "cached.sqsh"

        tr = TestRun(test=tdef, name="tr", num_nodes=1, nodes=[], output_path=tmp_path / "out")
        slurm_system.account = "acct"
        slurm_system.default_partition = "gb300"
        cmd_gen = MegatronBridgeSlurmCommandGenStrategy(slurm_system, tr)

        cmd = cmd_gen.gen_exec_command()
        assert "--use_tokendrop" not in cmd
        assert "--use_megatron_fsdp" not in cmd
        assert "--cuda_graph_impl" not in cmd
        assert " -ms " not in cmd

    def test_container_image_local_path_passed_verbatim(
        self, cmd_gen: MegatronBridgeSlurmCommandGenStrategy, test_run: TestRun
    ) -> None:
        tdef = cast(MegatronBridgeTestDefinition, test_run.test)
        local_img = Path(tdef.cmd_args.container_image)
        assert local_img.exists()

        cmd_gen.gen_exec_command()
        wrapper = test_run.output_path / "cloudai_megatron_bridge_submit_and_parse_jobid.sh"
        assert wrapper.exists()
        wrapper_content = wrapper.read_text()
        assert f"-i {local_img.absolute()}" in wrapper_content
        assert str(tdef.docker_image.installed_path) not in wrapper_content

    def test_cuda_graph_scope_normalization(self, cmd_gen: MegatronBridgeSlurmCommandGenStrategy) -> None:
        cmd_gen.gen_exec_command()
        wrapper = cmd_gen.test_run.output_path / "cloudai_megatron_bridge_submit_and_parse_jobid.sh"
        wrapper_content = wrapper.read_text()
        assert "--cuda_graph_scope moe_router,moe_preprocess" in wrapper_content

    @pytest.mark.parametrize(
        "detach, expected, not_expected",
        [
            (True, "--detach", "--no-detach"),
            (False, "--no-detach", "--detach"),
            (None, None, "--detach"),
        ],
    )
    def test_detach_flags(
        self,
        slurm_system: SlurmSystem,
        test_run: TestRun,
        detach: bool | None,
        expected: str | None,
        not_expected: str,
    ) -> None:
        tdef = cast(MegatronBridgeTestDefinition, test_run.test)

        data = tdef.cmd_args.model_dump(exclude_none=True)
        if detach is not None:
            data["detach"] = detach
        else:
            data.pop("detach", None)
        tdef.cmd_args = MegatronBridgeCmdArgs.model_validate(data)

        slurm_system.account = "acct"
        slurm_system.default_partition = "gb300"
        cmd_gen = MegatronBridgeSlurmCommandGenStrategy(slurm_system, test_run)
        cmd_gen.gen_exec_command()
        wrapper = test_run.output_path / "cloudai_megatron_bridge_submit_and_parse_jobid.sh"
        wrapper_content = wrapper.read_text()
        if detach is None:
            assert "--detach" not in wrapper_content
            assert "--no-detach" not in wrapper_content
        else:
            assert expected is not None
            assert expected in wrapper_content
            assert not_expected not in wrapper_content

    def test_generated_command_file_written(
        self, cmd_gen: MegatronBridgeSlurmCommandGenStrategy, test_run: TestRun
    ) -> None:
        cmd = cmd_gen.gen_exec_command()
        out_dir = test_run.output_path
        gen_file = out_dir / "cloudai_generated_command.sh"
        assert gen_file.exists()
        content = gen_file.read_text()
        assert cmd in content
        assert content.startswith("bash ")
        assert "cloudai_megatron_bridge_submit_and_parse_jobid.sh" in content
