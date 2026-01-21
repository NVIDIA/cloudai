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

import copy
import re
from typing import Generator, cast
from unittest.mock import Mock

import pandas as pd
import pytest
import toml

from cloudai.core import Registry, TestRun, TestScenario
from cloudai.systems.slurm import SingleSbatchRunner, SlurmJob, SlurmJobMetadata, SlurmSystem
from cloudai.workloads.nccl_test import NCCLCmdArgs, NCCLTestDefinition
from cloudai.workloads.nccl_test.slurm_command_gen_strategy import NcclTestSlurmCommandGenStrategy
from cloudai.workloads.sleep import SleepCmdArgs, SleepTestDefinition


class MyNCCL(NCCLTestDefinition):
    def constraint_check(self, tr: TestRun) -> bool:
        return "CONSTRAINT" not in tr.test.extra_env_vars


@pytest.fixture
def nccl_tr(slurm_system: SlurmSystem) -> Generator[TestRun, None, None]:
    tr = TestRun(
        name="nccl_test",
        test=MyNCCL(
            name="nccl",
            description="desc",
            test_template_name="NcclTest",
            cmd_args=NCCLCmdArgs(docker_image_url="fake://url/nccl"),
        ),
        num_nodes=2,
        nodes=[],
        output_path=slurm_system.output_path / "nccl_test",
    )
    Registry().add_command_gen_strategy(SlurmSystem, MyNCCL, NcclTestSlurmCommandGenStrategy)
    yield tr
    del Registry().command_gen_strategies_map[(SlurmSystem, MyNCCL)]


@pytest.fixture
def sleep_tr(slurm_system: SlurmSystem) -> TestRun:
    tr = TestRun(
        name="sleep_test",
        test=SleepTestDefinition(name="sleep", description="desc", test_template_name="t", cmd_args=SleepCmdArgs()),
        num_nodes=1,
        nodes=[],
        output_path=slurm_system.output_path / "sleep_test",
    )
    tr.output_path.mkdir(parents=True, exist_ok=True)
    return tr


@pytest.mark.parametrize("gres_support", [True, False])
def test_sbatch_default(sleep_tr: TestRun, slurm_system: SlurmSystem, gres_support: bool) -> None:
    tc = TestScenario(name="tc", test_runs=[sleep_tr])
    runner = SingleSbatchRunner(mode="run", system=slurm_system, test_scenario=tc, output_path=slurm_system.output_path)

    runner.system.supports_gpu_directives_cache = gres_support

    expected = [
        f"#SBATCH -N {sleep_tr.num_nodes}",
        f"#SBATCH --job-name={runner.job_name}",
        f"#SBATCH --output={runner.scenario_root.absolute() / 'common.out'}",
        f"#SBATCH --error={runner.scenario_root.absolute() / 'common.err'}",
        f"#SBATCH --partition={slurm_system.default_partition}",
    ]
    if gres_support:
        expected.extend(
            [
                f"#SBATCH --gpus-per-node={slurm_system.gpus_per_node}",
                f"#SBATCH --gres=gpu:{slurm_system.gpus_per_node}",
            ]
        )

    sbatch_lines = runner.get_sbatch_directives()
    assert sbatch_lines == expected


def test_sbatch_system_fields(sleep_tr: TestRun, slurm_system: SlurmSystem) -> None:
    sleep_tr.time_limit = "00:00:10"
    tc = TestScenario(name="tc", test_runs=[sleep_tr])
    runner = SingleSbatchRunner(mode="run", system=slurm_system, test_scenario=tc, output_path=slurm_system.output_path)

    runner.system.account = "test_account"
    runner.system.distribution = "test_distribution"
    runner.system.supports_gpu_directives_cache = True
    runner.system.gpus_per_node = 2
    runner.system.ntasks_per_node = 4
    runner.system.extra_sbatch_args = ["--test-arg1", "--test-arg2"]
    sbatch_lines = runner.get_sbatch_directives()
    assert sbatch_lines == [
        f"#SBATCH -N {sleep_tr.num_nodes}",
        f"#SBATCH --job-name={runner.job_name}",
        f"#SBATCH --output={runner.scenario_root.absolute() / 'common.out'}",
        f"#SBATCH --error={runner.scenario_root.absolute() / 'common.err'}",
        f"#SBATCH --partition={slurm_system.default_partition}",
        f"#SBATCH --time={sleep_tr.time_limit}",
        f"#SBATCH --account={runner.system.account}",
        f"#SBATCH --distribution={runner.system.distribution}",
        f"#SBATCH --gpus-per-node={runner.system.gpus_per_node}",
        f"#SBATCH --gres=gpu:{runner.system.gpus_per_node}",
        f"#SBATCH --ntasks-per-node={runner.system.ntasks_per_node}",
        "#SBATCH --test-arg1",
        "#SBATCH --test-arg2",
    ]


class TestNodeSpec:
    def test_max_nodes(self, sleep_tr: TestRun, slurm_system: SlurmSystem) -> None:
        another_tr = copy.deepcopy(sleep_tr)
        another_tr.num_nodes = 3
        tc = TestScenario(name="tc", test_runs=[sleep_tr, another_tr])
        runner = SingleSbatchRunner(
            mode="run", system=slurm_system, test_scenario=tc, output_path=slurm_system.output_path
        )
        sbatch_lines = runner.get_sbatch_directives()
        assert sbatch_lines[0] == f"#SBATCH -N {another_tr.num_nodes}"

    def test_nodes_list_covers_max(self, sleep_tr: TestRun, slurm_system: SlurmSystem) -> None:
        another_tr = copy.deepcopy(sleep_tr)
        sleep_tr.nodes = []
        sleep_tr.num_nodes = 1
        another_tr.nodes = ["node-[030-035]"]
        tc = TestScenario(name="tc", test_runs=[sleep_tr, another_tr])
        runner = SingleSbatchRunner(
            mode="run", system=slurm_system, test_scenario=tc, output_path=slurm_system.output_path
        )
        sbatch_lines = runner.get_sbatch_directives()
        _, node_list = runner.system.get_nodes_by_spec(1, another_tr.nodes)
        assert sbatch_lines[0] == f"#SBATCH --nodelist={','.join(node_list)}"

    def test_nodes_list_does_not_cover_max(self, sleep_tr: TestRun, slurm_system: SlurmSystem) -> None:
        another_tr = copy.deepcopy(sleep_tr)
        sleep_tr.nodes = []
        sleep_tr.num_nodes = 1
        another_tr.nodes = ["node-[030-035]"]
        nnodes, _ = slurm_system.get_nodes_by_spec(1, another_tr.nodes)
        sleep_tr.num_nodes = nnodes + 1

        tc = TestScenario(name="tc", test_runs=[sleep_tr, another_tr])
        runner = SingleSbatchRunner(
            mode="run", system=slurm_system, test_scenario=tc, output_path=slurm_system.output_path
        )

        with pytest.raises(ValueError) as exc_info:
            runner.get_sbatch_directives()

        assert str(exc_info.value) == (
            f"Number of nodes in the nodes list ({nnodes}) does not cover the max number "
            f"of nodes ({sleep_tr.num_nodes})"
        )

    def test_mixed_spec(self, sleep_tr: TestRun, slurm_system: SlurmSystem) -> None:
        another_tr = copy.deepcopy(sleep_tr)
        sleep_tr.nodes = []
        sleep_tr.num_nodes = 1
        another_tr.nodes = ["node-[030-035]"]
        tc = TestScenario(name="tc", test_runs=[sleep_tr, another_tr])
        runner = SingleSbatchRunner(
            mode="run", system=slurm_system, test_scenario=tc, output_path=slurm_system.output_path
        )

        srun_cmd = runner.get_single_tr_block(sleep_tr)
        assert f"-N{sleep_tr.num_nodes}" in srun_cmd
        assert "--nodelist" not in srun_cmd

        srun_cmd = runner.get_single_tr_block(another_tr)
        assert f"-N{another_tr.num_nodes}" not in srun_cmd
        assert "--nodelist" in srun_cmd


class TestTimeLimit:
    def test_simple(self, sleep_tr: TestRun, slurm_system: SlurmSystem) -> None:
        sleep_tr.time_limit = "00:00:10"
        tc = TestScenario(name="tc", test_runs=[sleep_tr])
        runner = SingleSbatchRunner(
            mode="run", system=slurm_system, test_scenario=tc, output_path=slurm_system.output_path
        )
        assert runner.build_time_limit() == sleep_tr.time_limit

    def test_no_limits(self, sleep_tr: TestRun, slurm_system: SlurmSystem) -> None:
        another_tr = copy.deepcopy(sleep_tr)
        another_tr.time_limit = None
        sleep_tr.time_limit = None
        tc = TestScenario(name="tc", test_runs=[sleep_tr, another_tr])
        runner = SingleSbatchRunner(
            mode="run", system=slurm_system, test_scenario=tc, output_path=slurm_system.output_path
        )
        assert runner.build_time_limit() is None

    def test_mixed_limits(self, sleep_tr: TestRun, slurm_system: SlurmSystem) -> None:
        another_tr = copy.deepcopy(sleep_tr)
        another_tr.time_limit = None
        sleep_tr.time_limit = "00:00:10"
        tc = TestScenario(name="tc", test_runs=[sleep_tr, another_tr])
        runner = SingleSbatchRunner(
            mode="run", system=slurm_system, test_scenario=tc, output_path=slurm_system.output_path
        )

        with pytest.raises(ValueError) as exc_info:
            runner.build_time_limit()

        assert str(exc_info.value) == "All tests must have a time limit or none of them must have a time limit"

    def test_sum_limits(self, sleep_tr: TestRun, slurm_system: SlurmSystem) -> None:
        sleep_tr.time_limit = "00:00:10"
        another_tr = copy.deepcopy(sleep_tr)
        another_tr.time_limit = "1-00:10:20"
        tc = TestScenario(name="tc", test_runs=[sleep_tr, another_tr])
        runner = SingleSbatchRunner(
            mode="run", system=slurm_system, test_scenario=tc, output_path=slurm_system.output_path
        )
        assert runner.build_time_limit() == "1-00:10:30"


class TestAuxCommands:
    def test_bare_metal(self, sleep_tr: TestRun, slurm_system: SlurmSystem) -> None:
        tc = TestScenario(name="tc", test_runs=[sleep_tr])
        runner = SingleSbatchRunner(
            mode="run", system=slurm_system, test_scenario=tc, output_path=slurm_system.output_path
        )
        aux_cmds = runner.aux_commands()
        assert len(aux_cmds) == 2

        metadata_cmd = (
            f"srun --export=ALL --mpi=pmix -N{sleep_tr.num_nodes} --ntasks=1 --ntasks-per-node=1 "
            f"--output={runner.scenario_root}/metadata/node-%N.toml "
            f"--error={runner.scenario_root}/metadata/nodes.err "
            "bash "
            f"{runner.system.install_path}/slurm-metadata.sh"
        )
        assert aux_cmds[0] == metadata_cmd

        ranks_mapping_cmd = (
            f"srun --export=ALL --mpi=pmix -N{sleep_tr.num_nodes} --output={runner.scenario_root}/mapping-stdout.txt "
            f"--error={runner.scenario_root}/mapping-stderr.txt bash -c "
            r'"echo \$(date): \$(hostname):node \${SLURM_NODEID}:rank \${SLURM_PROCID}."'
        )
        assert aux_cmds[1] == ranks_mapping_cmd

    def test_container(self, nccl_tr: TestRun, slurm_system: SlurmSystem) -> None:
        tdef = cast(NCCLTestDefinition, nccl_tr.test)
        tc = TestScenario(name="tc", test_runs=[nccl_tr])
        runner = SingleSbatchRunner(
            mode="run", system=slurm_system, test_scenario=tc, output_path=slurm_system.output_path
        )
        aux_cmds = runner.aux_commands()
        assert len(aux_cmds) == 2

        mounts = (
            f"--container-mounts={runner.system.output_path.absolute()}:/cloudai_run_results,"
            f"{runner.system.install_path.absolute()}:/cloudai_install,"
            f"{runner.system.output_path.absolute()}"
        )
        metadata_cmd = (
            f"srun --export=ALL --mpi=pmix -N{nccl_tr.num_nodes} --container-image={tdef.docker_image.installed_path} "
            f"{mounts} --no-container-mount-home --ntasks=2 --ntasks-per-node=1 "
            f"--output={runner.scenario_root}/metadata/node-%N.toml "
            f"--error={runner.scenario_root}/metadata/nodes.err "
            "bash /cloudai_install/slurm-metadata.sh"
        )
        assert aux_cmds[0] == metadata_cmd

        ranks_mapping_cmd = (
            f"srun --export=ALL --mpi=pmix -N{nccl_tr.num_nodes} --container-image={tdef.docker_image.installed_path} "
            f"{mounts} --no-container-mount-home --output={runner.scenario_root}/mapping-stdout.txt "
            f"--error={runner.scenario_root}/mapping-stderr.txt bash -c "
            r'"echo \$(date): \$(hostname):node \${SLURM_NODEID}:rank \${SLURM_PROCID}."'
        )
        assert aux_cmds[1] == ranks_mapping_cmd

    def test_max_nodes_used_for_metadata(self, nccl_tr: TestRun, slurm_system: SlurmSystem) -> None:
        another_tr = copy.deepcopy(nccl_tr)
        another_tr.num_nodes = nccl_tr.nnodes + 1
        tc = TestScenario(name="tc", test_runs=[nccl_tr, another_tr])
        runner = SingleSbatchRunner(
            mode="run", system=slurm_system, test_scenario=tc, output_path=slurm_system.output_path
        )
        aux_cmds = runner.aux_commands()

        metadata_cmd = aux_cmds[0]
        assert f"--ntasks={another_tr.nnodes}" in metadata_cmd

    def test_max_nodes_used_for_metadata_dse(self, nccl_tr: TestRun, slurm_system: SlurmSystem) -> None:
        nccl_tr.num_nodes = [1, 2]
        tc = TestScenario(name="tc", test_runs=[nccl_tr])
        runner = SingleSbatchRunner(
            mode="run", system=slurm_system, test_scenario=tc, output_path=slurm_system.output_path
        )
        aux_cmds = runner.aux_commands()

        metadata_cmd = aux_cmds[0]
        assert f"--ntasks={max(nccl_tr.num_nodes)}" in metadata_cmd


def test_single_tr_block(sleep_tr: TestRun, slurm_system: SlurmSystem) -> None:
    tc = TestScenario(name="tc", test_runs=[sleep_tr])
    runner = SingleSbatchRunner(mode="run", system=slurm_system, test_scenario=tc, output_path=slurm_system.output_path)

    runner.system.global_env_vars["GLOBAL_VAR"] = "global_value"
    sleep_tr.test.extra_env_vars["LOCAL_VAR"] = "local_value"
    block = runner.get_single_tr_block(sleep_tr)

    tdef = cast(SleepTestDefinition, sleep_tr.test)
    assert block == (
        f"srun -N1 "
        f"--output={sleep_tr.output_path.absolute()}/stdout.txt "
        f"--error={sleep_tr.output_path.absolute()}/stderr.txt "
        f'--export=ALL --mpi=pmix -N{sleep_tr.num_nodes} bash -c "source {sleep_tr.output_path.absolute()}/env_vars.sh;'
        f' sleep {tdef.cmd_args.seconds}"'
    )


def test_unroll_dse(nccl_tr: TestRun, slurm_system: SlurmSystem) -> None:
    nccl_tr.test.extra_env_vars["NCCL_VAR"] = ["v1", "v2"]
    tc = TestScenario(name="tc", test_runs=[nccl_tr])
    runner = SingleSbatchRunner(mode="run", system=slurm_system, test_scenario=tc, output_path=slurm_system.output_path)

    dse_runs = list(runner.unroll_dse(nccl_tr))

    assert len(dse_runs) == 2
    assert dse_runs[0].test.extra_env_vars["NCCL_VAR"] == "v1"
    assert dse_runs[1].test.extra_env_vars["NCCL_VAR"] == "v2"


def test_unroll_dse_constraint_check(nccl_tr: TestRun, slurm_system: SlurmSystem) -> None:
    nccl_tr.test.extra_env_vars["CONSTRAINT"] = "1"
    tc = TestScenario(name="tc", test_runs=[nccl_tr])
    runner = SingleSbatchRunner(mode="run", system=slurm_system, test_scenario=tc, output_path=slurm_system.output_path)

    dse_runs = list(runner.unroll_dse(nccl_tr))
    assert len(dse_runs) == 0


class TestSbatch:
    def test_single_case(self, slurm_system: SlurmSystem, nccl_tr: TestRun) -> None:
        nccl_tr.test.extra_env_vars["NCCL_VAR"] = "nccl_value"
        tc = TestScenario(name="tc", test_runs=[nccl_tr])
        runner = SingleSbatchRunner(
            mode="run", system=slurm_system, test_scenario=tc, output_path=slurm_system.output_path
        )

        sbatch = runner.gen_sbatch_content()
        output_paths = [
            p.replace(f"{runner.scenario_root.absolute()}", "SCENARIO_ROOT")
            for p in re.findall(r"--output=(\S+)", sbatch)
        ]
        assert len(output_paths) == 4
        assert output_paths[0] == "SCENARIO_ROOT/common.out"
        assert output_paths[1] == "SCENARIO_ROOT/metadata/node-%N.toml"
        assert output_paths[2] == "SCENARIO_ROOT/mapping-stdout.txt"
        assert output_paths[3] == f"SCENARIO_ROOT/{nccl_tr.name}/0/stdout.txt"

    def test_with_two_cases(self, slurm_system: SlurmSystem, nccl_tr: TestRun, sleep_tr: TestRun) -> None:
        nccl_tr.test.extra_env_vars["NCCL_VAR"] = "nccl_value"
        sleep_tr.test.extra_env_vars["SLEEP_VAR"] = "sleep_value"
        tc = TestScenario(name="tc", test_runs=[nccl_tr, sleep_tr])
        runner = SingleSbatchRunner(
            mode="run", system=slurm_system, test_scenario=tc, output_path=slurm_system.output_path
        )

        sbatch = runner.gen_sbatch_content()
        sbatch = sbatch.replace(str(runner.scenario_root.absolute()), "SCENARIO_ROOT").replace(
            str(runner.system.install_path.absolute()), "INSTALL_PATH"
        )

        ref = "\n".join(
            [
                "#!/bin/bash",
                *runner.get_sbatch_directives(),
                "",
                *runner.aux_commands(),
                "",
                runner.get_global_env_vars(),
                "",
                runner.get_single_tr_block(nccl_tr),
                "",
                runner.get_single_tr_block(sleep_tr),
                "",
            ]
        )
        ref = ref.replace(str(runner.scenario_root.absolute()), "SCENARIO_ROOT").replace(
            str(runner.system.install_path.absolute()), "INSTALL_PATH"
        )

        assert sbatch == ref

        output_paths = [
            p.replace(f"{runner.scenario_root.absolute()}", "SCENARIO_ROOT")
            for p in re.findall(r"--output=(\S+)", sbatch)
        ]
        assert len(output_paths) == 5
        assert output_paths[-2] != output_paths[-1]

    def test_dse(self, nccl_tr: TestRun, slurm_system: SlurmSystem) -> None:
        nccl_tr.test.extra_env_vars["NCCL_VAR"] = ["v1", "v2"]
        tc = TestScenario(name="tc", test_runs=[nccl_tr])
        runner = SingleSbatchRunner(
            mode="run", system=slurm_system, test_scenario=tc, output_path=slurm_system.output_path
        )

        sbatch = runner.gen_sbatch_content()
        sbatch = sbatch.replace(str(runner.scenario_root.absolute()), "SCENARIO_ROOT").replace(
            str(runner.system.install_path.absolute()), "INSTALL_PATH"
        )

        dse_runs = list(runner.unroll_dse(nccl_tr))
        ref = "\n".join(
            [
                "#!/bin/bash",
                *runner.get_sbatch_directives(),
                "",
                *runner.aux_commands(),
                "",
                runner.get_global_env_vars(),
                "",
                runner.get_single_tr_block(dse_runs[0]),
                "",
                runner.get_single_tr_block(dse_runs[1]),
                "",
            ]
        )
        ref = ref.replace(str(runner.scenario_root.absolute()), "SCENARIO_ROOT").replace(
            str(runner.system.install_path.absolute()), "INSTALL_PATH"
        )

        assert sbatch == ref

        paths = set([str(tr.output_path.absolute()) for tr in dse_runs])
        assert len(paths) == len(dse_runs), "Output paths are not unique"

    def test_dse_and_non_dse(self, nccl_tr: TestRun, slurm_system: SlurmSystem) -> None:
        dse_nccl = copy.deepcopy(nccl_tr)
        dse_nccl.test.extra_env_vars["NCCL_VAR"] = ["v1", "v2"]

        tc = TestScenario(name="tc", test_runs=[dse_nccl, nccl_tr])
        runner = SingleSbatchRunner(
            mode="run", system=slurm_system, test_scenario=tc, output_path=slurm_system.output_path
        )

        runner.on_job_submit = Mock()
        sbatch = runner.gen_sbatch_content()
        sbatch = sbatch.replace(str(runner.scenario_root.absolute()), "SCENARIO_ROOT").replace(
            str(runner.system.install_path.absolute()), "INSTALL_PATH"
        )

        dse_runs = list(runner.unroll_dse(dse_nccl))
        ref = "\n".join(
            [
                "#!/bin/bash",
                *runner.get_sbatch_directives(),
                "",
                *runner.aux_commands(),
                "",
                runner.get_global_env_vars(),
                "",
                runner.get_single_tr_block(dse_runs[0]),
                "",
                runner.get_single_tr_block(dse_runs[1]),
                "",
                runner.get_single_tr_block(nccl_tr),
                "",
            ]
        )
        ref = ref.replace(str(runner.scenario_root.absolute()), "SCENARIO_ROOT").replace(
            str(runner.system.install_path.absolute()), "INSTALL_PATH"
        )
        assert sbatch == ref
        assert runner.on_job_submit.call_count == 3  # 2 dse runs + 1 non-dse run


def test_store_job_metadata(nccl_tr: TestRun, slurm_system: SlurmSystem) -> None:
    tc = TestScenario(name="tc", test_runs=[nccl_tr])
    runner = SingleSbatchRunner(mode="run", system=slurm_system, test_scenario=tc, output_path=slurm_system.output_path)
    nccl_tr.output_path.mkdir(parents=True, exist_ok=True)
    runner.mode = "dry-run"

    runner.store_job_metadata(SlurmJob(nccl_tr, id=1))

    out_file = runner.scenario_root / "slurm-job.toml"
    assert out_file.exists()
    sjm = SlurmJobMetadata.model_validate(toml.load(out_file))
    assert sjm.job_id == 1
    assert sjm.is_single_sbatch is True
    assert sjm.srun_cmd == "n/a for single sbatch run"
    assert sjm.test_cmd == "n/a for single sbatch run"
    assert sjm.job_root == runner.scenario_root.absolute()

    assert sjm == SlurmJobMetadata.model_validate(toml.loads(toml.dumps(sjm.model_dump())))


def test_pre_test(nccl_tr: TestRun, sleep_tr: TestRun, slurm_system: SlurmSystem) -> None:
    nccl_tr.pre_test = TestScenario(name="pre_test", test_runs=[sleep_tr])
    tc = TestScenario(name="tc", test_runs=[nccl_tr])
    runner = SingleSbatchRunner(mode="run", system=slurm_system, test_scenario=tc, output_path=slurm_system.output_path)

    pre_tests = runner.add_pre_tests(nccl_tr.pre_test, nccl_tr)
    tdef = cast(SleepTestDefinition, sleep_tr.test)

    assert pre_tests == "\n".join(
        [
            f"srun --output={sleep_tr.output_path.absolute()}/stdout.txt "
            f"--error={sleep_tr.output_path.absolute()}/stderr.txt "
            f"--export=ALL --mpi=pmix -N{sleep_tr.num_nodes} bash -c "
            f'"source {sleep_tr.output_path.absolute()}/env_vars.sh; sleep {tdef.cmd_args.seconds}"',
            "SUCCESS_0=$()",
            "PRE_TEST_SUCCESS=$( [ $SUCCESS_0 -eq 1 ] && echo 1 || echo 0 )",
            "if [ $PRE_TEST_SUCCESS -ne 1 ]; then",
            "    exit 1",
            "fi",
            "",
        ]
    )


def test_trajectory_saved(dse_tr: TestRun, slurm_system: SlurmSystem) -> None:
    tc = TestScenario(name="tc", test_runs=[dse_tr])
    runner = SingleSbatchRunner(mode="run", system=slurm_system, test_scenario=tc, output_path=slurm_system.output_path)
    dse_tr.output_path = slurm_system.output_path / dse_tr.name
    dse_tr.output_path.mkdir(parents=True, exist_ok=True)

    trajectory_path = runner.scenario_root / dse_tr.name / f"{dse_tr.current_iteration}" / "trajectory.csv"
    trajectory_path.unlink(missing_ok=True)
    runner.handle_dse()

    assert trajectory_path.exists()
    df = pd.read_csv(trajectory_path)
    assert df.shape[0] == len(dse_tr.all_combinations)
    assert df["step"].tolist() == list(range(1, len(dse_tr.all_combinations) + 1))
