# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.nixl_perftest import (
    NixlPerftestCmdArgs,
    NixlPerftestSlurmCommandGenStrategy,
    NixlPerftestTestDefinition,
)
from cloudai.workloads.nixl_perftest.nixl_perftest import MatgenCmdArgs


@pytest.fixture
def nixl_perftest() -> NixlPerftestTestDefinition:
    return NixlPerftestTestDefinition(
        name="nixl_perftest",
        description="NixlPerftest",
        test_template_name="NixlPerftest",
        cmd_args=NixlPerftestCmdArgs(
            docker_image_url="nvcr.io#nvidia/pytorch:24.02-py3",
            perftest_script="/workspace/nixl/benchmark/kvbench/main.py",
            subtest="sequential-ct-perftest",
            matgen_script="/workspace/nixl/benchmark/kvbench/main.py",
            python_executable="/usr/bin/python3",
            etcd_path="/usr/bin/etcd",
            num_user_requests=100,
            batch_size=10,
            num_prefill_nodes=1,
            num_decode_nodes=1,
            model="m",
        ),
    )


@pytest.fixture
def test_run(nixl_perftest: NixlPerftestTestDefinition, slurm_system: SlurmSystem) -> TestRun:
    return TestRun(name="test_run", num_nodes=2, nodes=[], test=nixl_perftest, output_path=slurm_system.output_path)


def test_gen_matrix_gen_srun_command(test_run: TestRun, slurm_system: SlurmSystem) -> None:
    strategy = NixlPerftestSlurmCommandGenStrategy(slurm_system, test_run)
    strategy.gen_matrix_gen_command = lambda: ["cmd"]
    cmd = strategy.gen_matrix_gen_srun_command()
    assert cmd == [
        *strategy.gen_srun_prefix(),
        "--ntasks-per-node=1",
        "--ntasks=1",
        "-N1",
        "bash",
        "-c",
        '"cmd"',
    ]


@pytest.mark.parametrize(
    "opt_arg,value",
    [(None, None), ("isl_mean", 1), ("isl_scale", 1)],
)
def test_gen_matrix_gen_command_with_model(
    test_run: TestRun, slurm_system: SlurmSystem, opt_arg: str | None, value: int | None
) -> None:
    strategy = NixlPerftestSlurmCommandGenStrategy(slurm_system, test_run)
    tdef = cast(NixlPerftestTestDefinition, test_run.test)
    if opt_arg:
        setattr(tdef.cmd_args, opt_arg, value)
    cmd = strategy.gen_matrix_gen_command()
    ref_cmd = [
        tdef.cmd_args.python_executable,
        tdef.cmd_args.matgen_script,
        "generate",
        f"--num-user-requests={tdef.cmd_args.num_user_requests}",
        f"--batch-size={tdef.cmd_args.batch_size}",
        f"--num-prefill-nodes={tdef.cmd_args.num_prefill_nodes}",
        f"--num-decode-nodes={tdef.cmd_args.num_decode_nodes}",
        "--results-dir=" + str(strategy.matrix_gen_path.absolute()),
        f"--prefill-tp={tdef.cmd_args.prefill_tp}",
        f"--prefill-pp={tdef.cmd_args.prefill_pp}",
        f"--prefill-cp={tdef.cmd_args.prefill_cp}",
        f"--decode-tp={tdef.cmd_args.decode_tp}",
        f"--decode-pp={tdef.cmd_args.decode_pp}",
        f"--decode-cp={tdef.cmd_args.decode_cp}",
        f"--model={tdef.cmd_args.model}",
    ]
    if opt_arg:
        ref_cmd.append(f"{strategy.prop_to_cli_arg(opt_arg)}={value}")

    assert cmd == ref_cmd


def test_gen_matrix_gen_command(test_run: TestRun, slurm_system: SlurmSystem) -> None:
    strategy = NixlPerftestSlurmCommandGenStrategy(slurm_system, test_run)
    tdef = cast(NixlPerftestTestDefinition, test_run.test)
    tdef.cmd_args.model = None
    tdef.cmd_args.hidden_size = 1
    tdef.cmd_args.num_layers = 2
    tdef.cmd_args.num_heads = 3
    tdef.cmd_args.num_kv_heads = 4
    tdef.cmd_args.dtype_size = 5
    cmd = strategy.gen_matrix_gen_command()
    for arg_val in ["--hidden-size=1", "--num-layers=2", "--num-heads=3", "--num-kv-heads=4", "--dtype-size=5"]:
        assert arg_val in cmd


def test_gen_matrix_gen_command_with_matgen_args(test_run: TestRun, slurm_system: SlurmSystem) -> None:
    strategy = NixlPerftestSlurmCommandGenStrategy(slurm_system, test_run)
    tdef = cast(NixlPerftestTestDefinition, test_run.test)
    tdef.cmd_args.matgen_args = MatgenCmdArgs.model_validate({"unknown": "unknown"})
    slurm_system.ntasks_per_node = 2
    cmd = strategy.gen_matrix_gen_command()
    assert cmd[:3] == [tdef.cmd_args.python_executable, tdef.cmd_args.matgen_script, "generate"]
    assert f"--ppn={slurm_system.ntasks_per_node}" in cmd
    assert "--unknown=unknown" in cmd


@pytest.mark.parametrize(
    "system_ppn,args_ppn,expected_ppn",
    [
        (None, None, None),
        (3, None, 3),
        (None, 2, 2),
        (3, 2, 2),
    ],
)
def test_gen_matrix_gen_command_with_matgen_args_ppn(
    test_run: TestRun, slurm_system: SlurmSystem, system_ppn: int | None, args_ppn: int | None, expected_ppn: int | None
) -> None:
    """Ensure that ppn setup logic follows these rules:
    - If args.ppn is set, then ppn is set to args.ppn. Always. That is user's decision.
    - If args.ppn is not set, then ppn is set to system.ntasks_per_node if it is defined.
    """
    strategy = NixlPerftestSlurmCommandGenStrategy(slurm_system, test_run)
    tdef = cast(NixlPerftestTestDefinition, test_run.test)
    tdef.cmd_args.matgen_args = MatgenCmdArgs(ppn=args_ppn)
    slurm_system.ntasks_per_node = system_ppn
    cmd = strategy.gen_matrix_gen_command()
    if expected_ppn is not None:
        assert f"--ppn={expected_ppn}" in cmd
    else:
        assert "--ppn=" not in cmd


def test_gen_perftest_srun_command(test_run: TestRun, slurm_system: SlurmSystem) -> None:
    strategy = NixlPerftestSlurmCommandGenStrategy(slurm_system, test_run)
    tdef = cast(NixlPerftestTestDefinition, test_run.test)
    test_run.output_path.mkdir(parents=True, exist_ok=True)
    cmd = strategy.gen_perftest_srun_command()
    assert cmd == [
        *strategy.gen_srun_prefix(),
        "--overlap",
        f'bash -c "source {(test_run.output_path / "env_vars.sh").absolute()}; ',
        tdef.cmd_args.python_executable,
        tdef.cmd_args.perftest_script,
        tdef.cmd_args.subtest,
        str(strategy.matrix_gen_path.absolute() / "metadata.yaml"),
        "--json-output-path=" + str(test_run.output_path.absolute() / "results.json"),
        '"',
    ]


@pytest.mark.parametrize(
    "decode_tp,dec_nodes,prefill_tp,prefill_nodes,res",
    [
        (1, 1, 1, 1, True),  # decode/prefill ratio is 1:1
        (1, 2, 4, 8, True),  # decode/prefill ratio is 2:2
        (1, 2, 1, 1, False),  # decode/prefill ratio is 1:2
    ],
)
def test_constraint_check(
    nixl_perftest: NixlPerftestTestDefinition,
    test_run: TestRun,
    decode_tp: int,
    dec_nodes: int,
    prefill_tp: int,
    prefill_nodes: int,
    res: bool,
) -> None:
    nixl_perftest.cmd_args.decode_tp = decode_tp
    nixl_perftest.cmd_args.num_decode_nodes = dec_nodes
    nixl_perftest.cmd_args.prefill_tp = prefill_tp
    nixl_perftest.cmd_args.num_prefill_nodes = prefill_nodes
    assert nixl_perftest.constraint_check(test_run) is res
