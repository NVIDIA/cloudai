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
import os
from pathlib import Path
from typing import Any, Callable, Iterable, cast

import pytest

from cloudai.core import GitRepo, TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.megatron_bridge import (
    MegatronBridgeCmdArgs,
    MegatronBridgeSlurmCommandGenStrategy,
    MegatronBridgeTestDefinition,
)

WRAPPER_SCRIPT_NAME = "cloudai_megatron_bridge_submit_and_parse_jobid.sh"


class TestMegatronBridgeSlurmCommandGenStrategy:
    @staticmethod
    def _configure_fake_installs(tdef: MegatronBridgeTestDefinition, tmp_path: Path) -> None:
        # Fake installed paths for installables so command-gen doesn't depend on real installs.
        (tmp_path / "run_repo").mkdir(exist_ok=True)
        (tmp_path / "run_venv").mkdir(exist_ok=True)
        (tmp_path / "mbridge_repo").mkdir(exist_ok=True)
        tdef.python_executable.git_repo.installed_path = tmp_path / "run_repo"
        tdef.python_executable.venv_path = tmp_path / "run_venv"
        tdef.megatron_bridge_repo.installed_path = tmp_path / "mbridge_repo"
        tdef.docker_image.installed_path = tmp_path / "cached.sqsh"

    @staticmethod
    def _wrapper_content(cmd_gen: MegatronBridgeSlurmCommandGenStrategy) -> str:
        cmd_gen.gen_exec_command()
        wrapper = cmd_gen.test_run.output_path / WRAPPER_SCRIPT_NAME
        assert wrapper.exists()
        return wrapper.read_text()

    @staticmethod
    def _env_file_content(cmd_gen: MegatronBridgeSlurmCommandGenStrategy) -> str:
        cmd_gen.gen_exec_command()
        env_file = cmd_gen.test_run.output_path / "env_vars.sh"
        assert env_file.exists()
        return env_file.read_text()

    @pytest.fixture
    def configured_slurm_system(self, slurm_system: SlurmSystem) -> SlurmSystem:
        slurm_system.account = "acct"
        slurm_system.default_partition = "gb300"
        return slurm_system

    @pytest.fixture
    def make_test_run(self, tmp_path: Path) -> Callable[..., TestRun]:
        sqsh = tmp_path / "img.sqsh"
        sqsh.write_text("x")

        def _make(
            *,
            cmd_args_overrides: dict[str, Any] | None = None,
            git_commit: str = "r0.2.0",
            mount_as: str | None = "/opt/Megatron-Bridge",
            output_subdir: str = "out",
            num_nodes: int = 2,
        ) -> TestRun:
            cmd_args_data = {
                "container_image": str(sqsh),
                "hf_token": "dummy_token",
                "model_family_name": "qwen3",
                "model_recipe_name": "30b_a3b",
                "num_gpus": 8,
                "gpus_per_node": 4,
            }
            if cmd_args_overrides:
                cmd_args_data.update(cmd_args_overrides)

            repo_kwargs: dict[str, Any] = {
                "url": "https://github.com/NVIDIA-NeMo/Megatron-Bridge.git",
                "commit": git_commit,
            }
            if mount_as is not None:
                repo_kwargs["mount_as"] = mount_as

            tdef = MegatronBridgeTestDefinition(
                name="mb",
                description="desc",
                test_template_name="MegatronBridge",
                cmd_args=MegatronBridgeCmdArgs.model_validate(cmd_args_data),
                extra_container_mounts=[],
                git_repos=[GitRepo(**repo_kwargs)],
            )
            self._configure_fake_installs(tdef, tmp_path)
            (tmp_path / output_subdir).mkdir()
            return TestRun(
                test=tdef,
                name="tr",
                num_nodes=num_nodes,
                nodes=[],
                output_path=tmp_path / output_subdir,
            )

        return _make

    @pytest.fixture
    def test_run(self, make_test_run: Callable[..., TestRun]) -> TestRun:
        return make_test_run()

    @pytest.fixture
    def hf_token_env(self) -> Iterable[str]:
        old_hf_token = os.environ.get("HF_TOKEN")
        os.environ["HF_TOKEN"] = "dummy_token"
        yield "dummy_token"
        if old_hf_token:
            os.environ["HF_TOKEN"] = old_hf_token
        else:
            del os.environ["HF_TOKEN"]

    @pytest.fixture
    def no_hf_token_env(self) -> Iterable[None]:
        old_hf_token = os.environ.get("HF_TOKEN")
        if old_hf_token:
            del os.environ["HF_TOKEN"]
        yield
        if old_hf_token:
            os.environ["HF_TOKEN"] = old_hf_token

    @pytest.fixture
    def cmd_gen(
        self,
        configured_slurm_system: SlurmSystem,
        test_run: TestRun,
    ) -> MegatronBridgeSlurmCommandGenStrategy:
        return MegatronBridgeSlurmCommandGenStrategy(configured_slurm_system, test_run)

    @pytest.mark.usefixtures("no_hf_token_env")
    def test_hf_token_empty_is_rejected_by_schema(self) -> None:
        with pytest.raises(Exception, match=r"hf_token"):
            MegatronBridgeCmdArgs.model_validate(
                {"hf_token": "", "model_family_name": "qwen3", "model_recipe_name": "30b_a3b"}
            )
        with pytest.raises(Exception, match=r"hf_token"):
            MegatronBridgeCmdArgs.model_validate({"model_family_name": "qwen3", "model_recipe_name": "30b_a3b"})

    def test_hf_token_may_be_taken_from_env(self, hf_token_env: str) -> None:
        cmd_args = MegatronBridgeCmdArgs.model_validate(
            {"hf_token": "", "model_family_name": "qwen3", "model_recipe_name": "30b_a3b"}
        )
        assert cmd_args.hf_token == hf_token_env

    @pytest.mark.parametrize(
        ("field_name", "value", "match"),
        (
            ("model_family_name", "", "model_family_name"),
            ("model_recipe_name", "", "model_recipe_name"),
            ("model_family_name", "  \t   ", r"cmd_args\.model_family_name cannot be empty\."),
            ("model_recipe_name", "  \t   ", r"cmd_args\.model_recipe_name cannot be empty\."),
        ),
    )
    def test_model_fields_validation(self, field_name: str, value: str, match: str) -> None:
        data = {
            "hf_token": "dummy_token",
            "model_family_name": "qwen3",
            "model_recipe_name": "30b_a3b",
        } | {field_name: value}
        with pytest.raises(Exception, match=match):
            MegatronBridgeCmdArgs.model_validate(data)

    def test_defaults_not_emitted_when_not_set_in_toml(
        self, configured_slurm_system: SlurmSystem, make_test_run: Callable[..., TestRun]
    ) -> None:
        tr = make_test_run(num_nodes=1)
        cmd_gen = MegatronBridgeSlurmCommandGenStrategy(configured_slurm_system, tr)
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

        wrapper_content = self._wrapper_content(cmd_gen)
        assert f"-i {local_img.absolute()}" in wrapper_content
        assert str(tdef.docker_image.installed_path) not in wrapper_content

    def test_cuda_graph_scope_normalization(
        self, configured_slurm_system: SlurmSystem, make_test_run: Callable[..., TestRun]
    ) -> None:
        tr = make_test_run(cmd_args_overrides={"cuda_graph_scope": "[moe_router,moe_preprocess]"})
        cmd_gen = MegatronBridgeSlurmCommandGenStrategy(configured_slurm_system, tr)
        wrapper_content = self._wrapper_content(cmd_gen)
        assert "--cuda_graph_scope moe_router,moe_preprocess" in wrapper_content

    @pytest.mark.parametrize(
        ("log_content", "expected_is_successful"),
        (
            (None, False),
            ("", False),
            ("any\bthing", False),
            ("ain_fp8_mx/0 Step Time : 9.09s GPU utilization: 663.5MODEL_TFLOP/s/GPU", True),
        ),
    )
    def test_was_run_successful(
        self, make_test_run: Callable[..., TestRun], log_content: str | None, expected_is_successful: bool
    ) -> None:
        tr = make_test_run()
        tr.output_path.mkdir(parents=True, exist_ok=True)
        if log_content is not None:
            (tr.output_path / "cloudai_megatron_bridge_launcher.log").write_text(log_content)
        tdef = cast(MegatronBridgeTestDefinition, tr.test)
        result = tdef.was_run_successful(tr)
        assert result.is_successful is expected_is_successful

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

    @pytest.mark.parametrize(
        "use_recipes, expected_in_wrapper",
        [
            (True, True),
            (None, False),
        ],
    )
    def test_use_recipes_emitted_only_when_true(
        self,
        configured_slurm_system: SlurmSystem,
        make_test_run: Callable[..., TestRun],
        use_recipes: bool | None,
        expected_in_wrapper: bool,
    ) -> None:
        overrides = {} if use_recipes is None else {"use_recipes": use_recipes}
        output_subdir = "out_true" if expected_in_wrapper else "out_none"
        tr = make_test_run(cmd_args_overrides=overrides, output_subdir=output_subdir, num_nodes=1)
        cmd_gen = MegatronBridgeSlurmCommandGenStrategy(configured_slurm_system, tr)
        wrapper_content = self._wrapper_content(cmd_gen)
        assert ("--use_recipes" in wrapper_content) is expected_in_wrapper

    @pytest.mark.parametrize("mount_as", ("/opt/custom-megatron", None))
    def test_mount_as_adds_repo_to_container_mounts(
        self,
        configured_slurm_system: SlurmSystem,
        make_test_run: Callable[..., TestRun],
        tmp_path: Path,
        mount_as: str | None,
    ) -> None:
        tr = make_test_run(mount_as=mount_as, output_subdir="out_mount")
        tdef = cast(MegatronBridgeTestDefinition, tr.test)
        repo_path = tdef.megatron_bridge_repo.installed_path
        assert repo_path is not None

        cmd_gen = MegatronBridgeSlurmCommandGenStrategy(configured_slurm_system, tr)
        wrapper_content = self._wrapper_content(cmd_gen)
        if mount_as is not None:
            assert f"-cm {repo_path.absolute()}:/opt/custom-megatron" in wrapper_content
        else:
            assert f"{repo_path.absolute()}:" not in wrapper_content
            assert ":/opt/Megatron-Bridge" not in wrapper_content

    def test_extra_srun_args_forwarded(
        self, configured_slurm_system: SlurmSystem, make_test_run: Callable[..., TestRun]
    ) -> None:
        configured_slurm_system.extra_srun_args = "--reservation=my_reservation"
        tr = make_test_run(output_subdir="out_srun")
        tr.extra_srun_args = "--constraint=gpu"
        cmd_gen = MegatronBridgeSlurmCommandGenStrategy(configured_slurm_system, tr)
        wrapper_content = self._wrapper_content(cmd_gen)
        assert "--reservation=my_reservation" in wrapper_content
        assert "--constraint=gpu" in wrapper_content

    def test_profiling_ranks_string_format(
        self,
        configured_slurm_system: SlurmSystem,
        make_test_run: Callable[..., TestRun],
    ) -> None:
        tr = make_test_run(
            cmd_args_overrides={"profiling_ranks": "0,1,2,3", "enable_nsys": True},
            output_subdir="out_prof_str",
        )
        assert not tr.is_dse_job
        cmd_gen = MegatronBridgeSlurmCommandGenStrategy(configured_slurm_system, tr)
        wrapper_content = self._wrapper_content(cmd_gen)
        assert "--profiling_ranks 0,1,2,3" in wrapper_content

    @pytest.mark.parametrize(("input_vp", "expected_vp"), ((1, "None"), (2, "2")))
    def test_vp(
        self,
        configured_slurm_system: SlurmSystem,
        make_test_run: Callable[..., TestRun],
        input_vp: int,
        expected_vp: str,
    ):
        tr = make_test_run(cmd_args_overrides={"vp": input_vp})
        cmd_gen = MegatronBridgeSlurmCommandGenStrategy(configured_slurm_system, tr)
        wrapper_content = self._wrapper_content(cmd_gen)
        assert expected_vp in wrapper_content

    def test_env_vars(self, test_run: TestRun, configured_slurm_system: SlurmSystem):
        configured_slurm_system.global_env_vars = {"MELLANOX_VISIBLE_DEVICES": "0,1,2,3", "BLA": "bla"}
        cmd_gen = MegatronBridgeSlurmCommandGenStrategy(configured_slurm_system, test_run)
        cmd_gen.gen_exec_command()

        env_file = test_run.output_path / "env_vars.sh"
        assert env_file.is_file()
        assert env_file.read_text().splitlines() == [
            'export MELLANOX_VISIBLE_DEVICES="0,1,2,3"',
            'export BLA="bla"',
        ]
