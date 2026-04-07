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

    def test_git_repos_can_pin_megatron_bridge_commit(self, make_test_run: Callable[..., TestRun]) -> None:
        tr = make_test_run(git_commit="abcdef1234567890")
        tdef = cast(MegatronBridgeTestDefinition, tr.test)
        assert tdef.megatron_bridge_repo.commit == "abcdef1234567890"

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

    def test_env_vars_are_forwarded_via_custom_bash_cmds(
        self, configured_slurm_system: SlurmSystem, make_test_run: Callable[..., TestRun]
    ) -> None:
        tr = make_test_run()
        tdef = cast(MegatronBridgeTestDefinition, tr.test)
        tdef.extra_env_vars = {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "NCCL_DEBUG": "INFO"}

        cmd_gen = MegatronBridgeSlurmCommandGenStrategy(configured_slurm_system, tr)
        wrapper_content = self._wrapper_content(cmd_gen)
        assert "--custom_env_vars" not in wrapper_content
        assert "-cb 'export CUDA_VISIBLE_DEVICES=0,1,2,3'" in wrapper_content
        assert "-cb 'export NCCL_DEBUG=INFO'" in wrapper_content

    def test_container_runtime_env_vars_exported_in_wrapper_script(
        self, configured_slurm_system: SlurmSystem, make_test_run: Callable[..., TestRun]
    ) -> None:
        configured_slurm_system.global_env_vars = {
            "MELLANOX_VISIBLE_DEVICES": "0,1,4,5",
            "NCCL_IB_HCA": "roce_p0_r0,roce_p0_r1,roce_p0_r2,roce_p0_r3",
            "NCCL_IB_GID_INDEX": "3",
        }
        tr = make_test_run(output_subdir="out_container_rt")
        tdef = cast(MegatronBridgeTestDefinition, tr.test)
        tdef.extra_env_vars = {"NVIDIA_VISIBLE_DEVICES": "all", "NCCL_DEBUG": "INFO"}

        cmd_gen = MegatronBridgeSlurmCommandGenStrategy(configured_slurm_system, tr)
        wrapper_content = self._wrapper_content(cmd_gen)

        launcher_idx = wrapper_content.index("setup_experiment.py")

        assert "export MELLANOX_VISIBLE_DEVICES=0,1,4,5" in wrapper_content
        assert "export NVIDIA_VISIBLE_DEVICES=all" in wrapper_content
        mvd_idx = wrapper_content.index("export MELLANOX_VISIBLE_DEVICES=")
        nvd_idx = wrapper_content.index("export NVIDIA_VISIBLE_DEVICES=")
        assert mvd_idx < launcher_idx, "MELLANOX_VISIBLE_DEVICES must be exported before the launcher"
        assert nvd_idx < launcher_idx, "NVIDIA_VISIBLE_DEVICES must be exported before the launcher"

        assert "-cb 'export MELLANOX_VISIBLE_DEVICES=0,1,4,5'" in wrapper_content
        assert "-cb 'export NVIDIA_VISIBLE_DEVICES=all'" in wrapper_content
        assert "-cb 'export NCCL_IB_HCA=roce_p0_r0,roce_p0_r1,roce_p0_r2,roce_p0_r3'" in wrapper_content
        assert "-cb 'export NCCL_DEBUG=INFO'" in wrapper_content

        assert "export NCCL_IB_HCA=" not in wrapper_content.split("setup_experiment.py")[0]
        assert "export NCCL_DEBUG=" not in wrapper_content.split("setup_experiment.py")[0]

    def test_wrapper_emits_job_id_even_when_launcher_non_zero(
        self, configured_slurm_system: SlurmSystem, make_test_run: Callable[..., TestRun]
    ) -> None:
        tr = make_test_run()
        cmd_gen = MegatronBridgeSlurmCommandGenStrategy(configured_slurm_system, tr)
        wrapper_content = self._wrapper_content(cmd_gen)
        assert 'if [ "${LAUNCH_RC}" -ne 0 ]; then' in wrapper_content
        assert 'echo "Submitted batch job ${JOB_ID}"' in wrapper_content
        assert 'exit "${LAUNCH_RC}"' not in wrapper_content
        assert "Submitted batch job[ ]+[0-9]+" in wrapper_content

    def test_wrapper_installs_wandb_before_launcher(
        self, configured_slurm_system: SlurmSystem, make_test_run: Callable[..., TestRun]
    ) -> None:
        tr = make_test_run()
        cmd_gen = MegatronBridgeSlurmCommandGenStrategy(configured_slurm_system, tr)
        wrapper_content = self._wrapper_content(cmd_gen)

        assert "-m pip install wandb numpy==1.26.4" in wrapper_content
        wandb_idx = wrapper_content.index("-m pip install wandb")
        launcher_idx = wrapper_content.index("setup_experiment.py")
        assert wandb_idx < launcher_idx

    def test_wrapper_exits_when_wandb_install_fails(
        self, configured_slurm_system: SlurmSystem, make_test_run: Callable[..., TestRun]
    ) -> None:
        tr = make_test_run()
        cmd_gen = MegatronBridgeSlurmCommandGenStrategy(configured_slurm_system, tr)
        wrapper_content = self._wrapper_content(cmd_gen)

        assert 'if [ "${WANDB_INSTALL_RC}" -ne 0 ]; then' in wrapper_content
        assert (
            'echo "Failed to install runtime deps (wandb, numpy==1.26.4) in launcher venv (exit '
            '${WANDB_INSTALL_RC})." >&2'
        ) in wrapper_content
        assert 'exit "${WANDB_INSTALL_RC}"' in wrapper_content

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

    def test_mount_as_adds_repo_to_container_mounts(
        self, configured_slurm_system: SlurmSystem, make_test_run: Callable[..., TestRun], tmp_path: Path
    ) -> None:
        tr = make_test_run(mount_as="/opt/custom-megatron", output_subdir="out_mount")
        tdef = cast(MegatronBridgeTestDefinition, tr.test)
        repo_path = tdef.megatron_bridge_repo.installed_path
        assert repo_path is not None

        cmd_gen = MegatronBridgeSlurmCommandGenStrategy(configured_slurm_system, tr)
        wrapper_content = self._wrapper_content(cmd_gen)
        assert f"-cm {repo_path.absolute()}:/opt/custom-megatron" in wrapper_content

    def test_no_mount_as_skips_repo_container_mount(
        self, configured_slurm_system: SlurmSystem, make_test_run: Callable[..., TestRun]
    ) -> None:
        tr = make_test_run(mount_as=None, output_subdir="out_no_mount")
        tdef = cast(MegatronBridgeTestDefinition, tr.test)
        repo_path = tdef.megatron_bridge_repo.installed_path
        assert repo_path is not None

        cmd_gen = MegatronBridgeSlurmCommandGenStrategy(configured_slurm_system, tr)
        wrapper_content = self._wrapper_content(cmd_gen)
        assert f"{repo_path.absolute()}:" not in wrapper_content
        assert ":/opt/Megatron-Bridge" not in wrapper_content

    @pytest.mark.parametrize(("system_gpus_per_node", "expected_gpus"), ((None, None), (4, 4)))
    def test_gpus_per_node(
        self,
        configured_slurm_system: SlurmSystem,
        make_test_run: Callable[..., TestRun],
        system_gpus_per_node: int | None,
        expected_gpus: int | None,
    ) -> None:
        configured_slurm_system.supports_gpu_directives_cache = True
        configured_slurm_system.gpus_per_node = system_gpus_per_node
        tr = make_test_run(output_subdir="out_gpus")
        cmd_gen = MegatronBridgeSlurmCommandGenStrategy(configured_slurm_system, tr)
        wrapper_content = self._wrapper_content(cmd_gen)

        if expected_gpus is None:
            assert "--additional_slurm_params" not in wrapper_content
            assert "-gn" not in wrapper_content
        else:
            assert "--additional_slurm_params" in wrapper_content
            assert f"gpus-per-node={expected_gpus}" in wrapper_content
            assert f"gres=gpu:{expected_gpus}" in wrapper_content
            assert f"-gn {expected_gpus}" in wrapper_content

    def test_gpus_per_node_skipped_when_gpu_directives_unsupported(
        self, configured_slurm_system: SlurmSystem, make_test_run: Callable[..., TestRun]
    ) -> None:
        configured_slurm_system.supports_gpu_directives_cache = False
        tr = make_test_run(cmd_args_overrides={"gpus_per_node": 2}, output_subdir="out_no_gpu_directives")
        cmd_gen = MegatronBridgeSlurmCommandGenStrategy(configured_slurm_system, tr)
        wrapper_content = self._wrapper_content(cmd_gen)
        assert "gpus-per-node=2" not in wrapper_content
        assert "gres=gpu:2" not in wrapper_content

    def test_system_extra_srun_args_forwarded(
        self, configured_slurm_system: SlurmSystem, make_test_run: Callable[..., TestRun]
    ) -> None:
        configured_slurm_system.extra_srun_args = "--reservation my_reserv"
        tr = make_test_run(output_subdir="out_srun")
        cmd_gen = MegatronBridgeSlurmCommandGenStrategy(configured_slurm_system, tr)
        wrapper_content = self._wrapper_content(cmd_gen)
        assert "reservation=my_reserv" in wrapper_content

    def test_test_run_extra_srun_args_forwarded(
        self, configured_slurm_system: SlurmSystem, make_test_run: Callable[..., TestRun]
    ) -> None:
        tr = make_test_run(output_subdir="out_tr_srun")
        tr.extra_srun_args = "--constraint gpu"
        cmd_gen = MegatronBridgeSlurmCommandGenStrategy(configured_slurm_system, tr)
        wrapper_content = self._wrapper_content(cmd_gen)
        assert "constraint=gpu" in wrapper_content

    def test_parse_srun_args_as_slurm_params(self) -> None:
        result = MegatronBridgeSlurmCommandGenStrategy._parse_srun_args_as_slurm_params(
            "--reservation my_reserv --constraint=gpu"
        )
        assert result == ["reservation=my_reserv", "constraint=gpu"]

    def test_parse_srun_args_boolean_flags(self) -> None:
        result = MegatronBridgeSlurmCommandGenStrategy._parse_srun_args_as_slurm_params(
            "--exclusive --reservation my_reserv --overcommit"
        )
        assert result == ["exclusive", "reservation=my_reserv", "overcommit"]

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

    def test_vp(
        self,
        configured_slurm_system: SlurmSystem,
        make_test_run: Callable[..., TestRun],
    ):
        tr = make_test_run(cmd_args_overrides={"vp": 1})
        cmd_gen = MegatronBridgeSlurmCommandGenStrategy(configured_slurm_system, tr)
        wrapper_content = self._wrapper_content(cmd_gen)
        assert "-vp None" in wrapper_content
