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

from typing import cast

import pytest

from cloudai.core import TestRun
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.workloads.nixl_kvbench import (
    NIXLKVBenchCmdArgs,
    NIXLKVBenchSlurmCommandGenStrategy,
    NIXLKVBenchTestDefinition,
)


@pytest.fixture
def kvbench() -> NIXLKVBenchTestDefinition:
    return NIXLKVBenchTestDefinition(
        name="n",
        description="d",
        test_template_name="NIXLKVBench",
        cmd_args=NIXLKVBenchCmdArgs(docker_image_url="docker://image/url"),
    )


@pytest.fixture
def kvbench_tr(kvbench: NIXLKVBenchTestDefinition, tmp_path) -> TestRun:
    output_path = tmp_path / "nixl-kvbench"
    output_path.mkdir(parents=True, exist_ok=True)
    return TestRun(name="nixl-bench", num_nodes=2, nodes=[], test=kvbench, output_path=output_path)


def test_gen_kvbench_ucx(kvbench_tr: TestRun, slurm_system: SlurmSystem):
    kvbench_tr.test.cmd_args = NIXLKVBenchCmdArgs.model_validate(
        {
            "docker_image_url": "docker://image/url",
            "model": "./model.yaml",
            "model_config": "./cfg.yaml",
            "backend": "UCX",
            "source": "src",
            "op_type": "READ",
        }
    )
    kvbench = cast(NIXLKVBenchTestDefinition, kvbench_tr.test)
    cmd_gen = NIXLKVBenchSlurmCommandGenStrategy(slurm_system, kvbench_tr)
    cmd = cmd_gen.gen_kvbench_command()
    assert cmd == [
        f"{kvbench.cmd_args.python_executable}",
        f"{kvbench.cmd_args.kvbench_script}",
        kvbench.cmd_args.command,
        "--backend UCX",
        "--model ./model.yaml",
        "--model_config ./cfg.yaml",
        "--source src",
        "--op_type READ",
        "--etcd_endpoints http://$NIXL_ETCD_ENDPOINTS",
    ]


def test_gen_kvbench_model_cfg_maps_to_model_config(kvbench_tr: TestRun, slurm_system: SlurmSystem):
    kvbench_tr.test.cmd_args = NIXLKVBenchCmdArgs.model_validate(
        {
            "docker_image_url": "docker://image/url",
            "model_cfg": "./cfg.yaml",
        }
    )
    cmd_gen = NIXLKVBenchSlurmCommandGenStrategy(slurm_system, kvbench_tr)
    cmd = cmd_gen.gen_kvbench_command()

    # Ensure the alias 'model_cfg' is mapped to '--model_config'
    assert "--model_config ./cfg.yaml" in cmd
    assert not any(arg.startswith("--model_cfg ") for arg in cmd)


def test_gen_kvbench_omits_none_values(kvbench_tr: TestRun, slurm_system: SlurmSystem):
    kvbench_tr.test.cmd_args = NIXLKVBenchCmdArgs.model_validate(
        {
            "docker_image_url": "docker://image/url",
            "model": "./model.yaml",
            "op_type": None,
            "source": None,
        }
    )
    cmd_gen = NIXLKVBenchSlurmCommandGenStrategy(slurm_system, kvbench_tr)
    cmd = cmd_gen.gen_kvbench_command()

    # Ensure None-valued args are omitted entirely
    assert not any(arg.startswith("--op_type ") for arg in cmd)
    assert not any(arg.startswith("--source ") for arg in cmd)


def test_gen_kvbench_command_includes_etcd_endpoints(kvbench_tr: TestRun, slurm_system: SlurmSystem):
    kvbench_tr.test.cmd_args = NIXLKVBenchCmdArgs.model_validate(
        {
            "docker_image_url": "docker://image/url",
            "etcd_image_url": "docker://etcd/url",
            "model": "./model.yaml",
        }
    )
    cmd_gen = NIXLKVBenchSlurmCommandGenStrategy(slurm_system, kvbench_tr)
    cmd = cmd_gen.gen_kvbench_command()

    assert "etcd_image_url" not in " ".join(cmd)
    assert "docker://etcd/url" not in " ".join(cmd)


def test_get_etcd_srun_command_with_etcd_image(kvbench_tr: TestRun, slurm_system: SlurmSystem):
    strategy = NIXLKVBenchSlurmCommandGenStrategy(slurm_system, kvbench_tr)
    tdef: NIXLKVBenchTestDefinition = cast(NIXLKVBenchTestDefinition, kvbench_tr.test)
    tdef.cmd_args.etcd_image_url = "docker.io/library/etcd:latest"

    cmd = " ".join(strategy.gen_etcd_srun_command(tdef.cmd_args.etcd_path))
    assert tdef.etcd_image is not None
    assert f"--container-image={tdef.etcd_image.installed_path}" in cmd


def test_kvbench_cleanup_command_uses_host_paths(kvbench_tr: TestRun, slurm_system: SlurmSystem):
    kvbench_tr.test.cmd_args = NIXLKVBenchCmdArgs.model_validate(
        {
            "docker_image_url": "docker://image/url",
            "backend": "GUSLI",
            "filepath": "/data",
            "device_list": "11:F:/store0.bin",
        }
    )
    strategy = NIXLKVBenchSlurmCommandGenStrategy(slurm_system, kvbench_tr)

    cmd = " ".join(strategy.gen_cleanup_command())
    filepath_dir = kvbench_tr.output_path / "filepath_mount"
    device_list_dir = kvbench_tr.output_path / "device_list_mounts"
    assert cmd == f"rm -rf {filepath_dir} {device_list_dir}"


def test_kvbench_gen_cleanup_command_empty_without_storage_args(kvbench_tr: TestRun, slurm_system: SlurmSystem):
    strategy = NIXLKVBenchSlurmCommandGenStrategy(slurm_system, kvbench_tr)

    assert strategy.gen_cleanup_command() == []


def test_kvbench_gen_srun_command_includes_host_cleanup(kvbench_tr: TestRun, slurm_system: SlurmSystem):
    kvbench_tr.test.cmd_args = NIXLKVBenchCmdArgs.model_validate(
        {
            "docker_image_url": "docker://image/url",
            "backend": "GUSLI",
            "filepath": "/data",
            "device_list": "11:F:/store0.bin",
        }
    )
    strategy = NIXLKVBenchSlurmCommandGenStrategy(slurm_system, kvbench_tr)

    cleanup_cmd = " ".join(strategy.gen_cleanup_command())
    cmd = strategy._gen_srun_command()

    assert cleanup_cmd in cmd
