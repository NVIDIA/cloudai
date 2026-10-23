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

import shlex
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from cloudai.core import GitRepo, TestRun, TestScenario
from cloudai.systems.slurm import SlurmCommandGenStrategy, SlurmSystem
from cloudai.workloads.megatron_bridge import (
    MegatronBridgeCmdArgs,
    MegatronBridgeSlurmCommandGenStrategy,
    MegatronBridgeTestDefinition,
)


@pytest.fixture
def strategy(slurm_system: SlurmSystem, tmp_path: Path) -> MegatronBridgeSlurmCommandGenStrategy:
    tdef = MegatronBridgeTestDefinition(
        name="bridge",
        description="test",
        test_template_name="MegatronBridge",
        cmd_args=MegatronBridgeCmdArgs(
            hf_token="dummy_token", model_family_name="qwen", model_recipe_name="qwen3_30b_a3b"
        ),
    )
    tdef.python_executable.venv_path = tmp_path / "nemo-run-venv"
    return MegatronBridgeSlurmCommandGenStrategy(
        slurm_system,
        TestRun(
            test_definition=tdef.model_copy(update={"id": "bridge", "num_nodes": 2}),
            nodes=[],
            output_path=tmp_path / "job",
        ),
    )


@pytest.mark.parametrize(
    "launcher_present,python_present", [(True, None), (False, False), (False, True), (True, False), (True, True)]
)
def test_launcher_dependencies(
    strategy: MegatronBridgeSlurmCommandGenStrategy, launcher_present: bool, python_present: bool | None
):
    tdef = strategy.test_run.test_definition
    assert isinstance(tdef, MegatronBridgeTestDefinition)
    tdef.git_repos = [GitRepo(url="https://github.com/NVIDIA-NeMo/Megatron-Bridge.git", commit="v0.4.0")]
    repo = strategy.system.install_path / tdef.megatron_bridge_repo.repo_name
    venv = strategy.system.install_path / tdef.python_executable.venv_name
    tdef.megatron_bridge_repo.installed_path = repo
    tdef.python_executable.venv_path = venv if python_present is not None else None
    for present, path in (
        (launcher_present, repo / "scripts/performance/setup_experiment.py"),
        (python_present, venv / "bin/python"),
    ):
        if present:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    with patch("subprocess.run") as run:
        run.return_value.returncode = 1
        if launcher_present and python_present:
            with pytest.raises(RuntimeError, match="Megatron-Bridge dry-run failed"):
                strategy._generate_mbridge_sbatch()
            run.assert_called_once()
        else:
            assert strategy._generate_mbridge_sbatch() is None
            run.assert_not_called()
            assert not list(strategy.test_run.output_path.glob("experiments/**/*_sbatch.sh"))

    if not (launcher_present and python_present):
        body = strategy.generate_test_command()[0]
        result = subprocess.run(["bash", "-c", body], capture_output=True, text=True)
        assert result.returncode == 1
        assert "Install MBridge and NeMo Run" in result.stderr


@pytest.mark.parametrize(
    "overrides,expected,absent",
    [
        (
            {},
            ["-m qwen", "-mr qwen3_30b_a3b", "-vb false"],
            ["--task", "-en", "--use_recipes", "--moe_flex_dispatcher_backend", "-cs"],
        ),
        (
            {
                "vp": 1,
                "use_recipes": True,
                "use_tokendrop": False,
                "use_megatron_fsdp": True,
                "nccl_ub": False,
                "moe_a2a_overlap": False,
                "moe_flex_dispatcher_backend": "None",
                "cuda_graph_scope": "[moe_router,moe_preprocess]",
                "recompute_modules": "['core_attn', 'mlp']",
                "dataset_paths": ["a", "b"],
                "profiling_ranks": [0, 8],
                "enable_nsys": True,
                "nsys_trace": "cuda,nvtx",
                "nsys_extra_args": "",
                "packed_sequence": True,
                "head_only": True,
                "profiling_gpu_metrics": True,
                "list_config_variants": True,
                "dryrun": True,
            },
            [
                "-vp None",
                "--use_recipes",
                "--use_tokendrop false",
                "--use_megatron_fsdp true",
                "--nccl_ub false",
                "--moe_a2a_overlap false",
                "--moe_flex_dispatcher_backend None",
                "--cuda_graph_scope moe_router,moe_preprocess",
                "--recompute_modules core_attn,mlp",
                "--dataset_paths a b",
                "--profiling_ranks 0,8",
                "-en",
                "--nsys_trace cuda,nvtx",
                "--packed_sequence",
                "--head_only",
                "--profiling_gpu_metrics",
                "--list_config_variants",
                "-d",
            ],
            ["--nsys_extra_args"],
        ),
        (
            {
                "vp": [1],
                "tp": [2],
                "pp": [2, 4],
                "ep": [],
                "recompute_modules": ["core_attn", "mlp"],
                "profiling_ranks": "0,8",
                "use_recipes": False,
            },
            ["-vp None", "-tp 2", "-pp 2,4", "--recompute_modules core_attn,mlp", "--profiling_ranks 0,8"],
            ["-ep", "--use_recipes"],
        ),
    ],
)
def test_launcher_argument_translation(
    strategy: MegatronBridgeSlurmCommandGenStrategy,
    overrides: dict[str, Any],
    expected: list[str],
    absent: list[str],
) -> None:
    tdef = strategy.test_run.test_definition
    assert isinstance(tdef, MegatronBridgeTestDefinition)
    tdef.cmd_args = MegatronBridgeCmdArgs.model_validate(tdef.cmd_args.model_dump(exclude_unset=True) | overrides)
    tdef.extra_cmd_args = {"logger.tensorboard_dir": "/custom tensorboard", "logger.log_memory_to_tensorboard": "false"}
    parts = strategy._build_launcher_parts(tdef.cmd_args, tdef, strategy.system.install_path, Path("launcher.py"))
    command = " ".join(shlex.split(" ".join(parts)))
    for fragment in expected:
        assert fragment in command
    for flag in absent:
        assert flag not in shlex.split(command)
    assert "logger.tensorboard_dir=/custom tensorboard" in shlex.split(" ".join(parts))
    assert "logger.log_memory_to_tensorboard=false" in command
    assert "logger.tensorboard_dir=/nemo_run/tb_logs" not in command


@pytest.mark.parametrize(
    "native_args,expected_native",
    [
        ({}, ""),
        ({"-cs": "--cpu-bind=cores"}, "--cpu-bind=cores,"),
        ({"--custom_srun_args": "--cpu-bind=cores"}, "--cpu-bind=cores,"),
        ({"-cs": "--cpu-bind=cores", "--custom_srun_args": "--label"}, "--cpu-bind=cores,--label,"),
    ],
)
def test_custom_srun_args(
    strategy: MegatronBridgeSlurmCommandGenStrategy, native_args: dict[str, str], expected_native: str
) -> None:
    tdef = strategy.test_run.test_definition
    assert isinstance(tdef, MegatronBridgeTestDefinition)
    tdef.extra_cmd_args = native_args.copy()
    strategy.system.extra_srun_args = "--comment='smoke test' --kill-on-bad-exit=1"
    tdef = tdef.model_copy(update={"extra_srun_args": ["--cpu-bind=none"]})
    strategy.test_run.test_definition = tdef
    strategy.test_run.pre_test = TestScenario(
        name="pre",
        test_runs=[TestRun(test_definition=tdef.model_copy(update={"id": "hook", "num_nodes": 2}), nodes=[])],
    )
    with patch.object(SlurmCommandGenStrategy, "pre_test_srun_extra_args", return_value=["--overlap"]):
        parts = strategy._build_launcher_parts(tdef.cmd_args, tdef, strategy.system.install_path, Path("launcher.py"))
    tokens = shlex.split(" ".join(parts))
    assert [arg for arg in tokens if arg.startswith(("-cs=", "--custom_srun_args="))] == [
        f"-cs={expected_native}--comment='smoke test' --kill-on-bad-exit=1,--cpu-bind=none,--overlap"
    ]
    assert tdef.extra_cmd_args == native_args


@pytest.mark.parametrize(
    "parser,overrides,expected",
    [
        (None, {}, {}),
        ("", {}, {}),
        ('add_argument("--packager")', {}, {"--packager": "none"}),
        ("add_argument('--packager')", {"--packager": "git"}, {"--packager": "git"}),
    ],
)
def test_version_dependent_packager(
    tmp_path: Path,
    parser: str | None,
    overrides: dict[str, str],
    expected: dict[str, str],
) -> None:
    if parser is not None:
        path = tmp_path / "scripts/performance/argument_parser.py"
        path.parent.mkdir(parents=True)
        path.write_text(parser)
    original = overrides.copy()
    assert MegatronBridgeSlurmCommandGenStrategy._with_default_packager(tmp_path, overrides) == expected
    assert overrides == original


@pytest.mark.parametrize("image_kind", ["local", "cached", "remote"])
def test_launcher_container_resolution(
    strategy: MegatronBridgeSlurmCommandGenStrategy, tmp_path: Path, image_kind: str
) -> None:
    tdef = strategy.test_run.test_definition
    assert isinstance(tdef, MegatronBridgeTestDefinition)
    image = tmp_path / "nemo.sqsh"
    tdef.cmd_args.container_image = str(image) if image_kind == "local" else "nvcr.io/nvidia/nemo:26.04.00"
    if image_kind == "local":
        image.touch()
    elif image_kind == "cached":
        tdef.docker_image.installed_path = image
    parts = strategy._build_launcher_parts(tdef.cmd_args, tdef, tmp_path, Path("launcher.py"))
    assert parts[parts.index("-i") + 1] == (tdef.cmd_args.container_image if image_kind == "remote" else str(image))


@pytest.mark.parametrize("cleanup_error", [False, True])
def test_artifact_cleanup(strategy: MegatronBridgeSlurmCommandGenStrategy, tmp_path: Path, cleanup_error: bool) -> None:
    strategy.cleanup_job_artifacts()
    root = strategy.test_run.output_path
    code = root / "experiments/run/code"
    code.mkdir(parents=True)
    (code / "source.py").write_text("packaged source")
    external = tmp_path / "external.log"
    external.write_text("dummy_token")
    (root / "linked.log").symlink_to(external)
    (root / "experiments/linked/code").parent.mkdir()
    (root / "experiments/linked/code").symlink_to(code, target_is_directory=True)
    untouched = root / "trace.bin"
    untouched.write_bytes(b"dummy_token")
    artifact = root / "training.log"
    # Two tokens, one straddling the streaming reader's chunk boundary.
    contents = b"x" * (1024 * 1024 - 4) + b"dummy_token dummy_token"
    artifact.write_bytes(contents)
    if cleanup_error:
        with (
            patch("shutil.rmtree", side_effect=OSError("denied")),
            pytest.raises(RuntimeError, match="Failed to clean 1"),
        ):
            strategy.cleanup_job_artifacts()
        assert code.exists()
    else:
        strategy.cleanup_job_artifacts()
        assert not code.exists()
    assert artifact.read_bytes() == contents.replace(b"dummy_token", b"XXXXXXXXXXX")
    assert untouched.read_bytes() == b"dummy_token"
    assert external.read_text() == "dummy_token"
