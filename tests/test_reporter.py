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
import csv
import tarfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import toml

from cloudai import TestRun, TestScenario
from cloudai.cli.handlers import generate_reports
from cloudai.core import CommandGenStrategy, Registry, Reporter, System
from cloudai.models.scenario import ReportConfig, TestRunDetails
from cloudai.report_generator.dse_report import build_dse_summaries
from cloudai.reporter import DSEReporter, PerTestReporter, ReportItem, StatusReporter, TarballReporter
from cloudai.systems.slurm.slurm_metadata import (
    MetadataCUDA,
    MetadataMPI,
    MetadataNCCL,
    MetadataNetwork,
    MetadataSlurm,
    MetadataSystem,
    SlurmJobMetadata,
    SlurmStepMetadata,
    SlurmSystemMetadata,
)
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.systems.standalone.standalone_system import StandaloneSystem
from cloudai.workloads.nccl_test import NCCLCmdArgs, NCCLTestDefinition


class TestLoadTestTuns:
    def test_load_test_runs_behcnmark_sorted(self, slurm_system: SlurmSystem, benchmark_tr: TestRun) -> None:
        reporter = PerTestReporter(
            slurm_system,
            TestScenario(name="test_scenario", test_runs=[benchmark_tr]),
            slurm_system.output_path,
            ReportConfig(),
        )
        reporter.load_test_runs()

        assert len(reporter.trs) == benchmark_tr.iterations
        for i, tr in enumerate(reporter.trs):
            assert tr.name == benchmark_tr.name
            assert tr.current_iteration == i
            assert tr.step == 0
            assert tr.output_path == slurm_system.output_path / benchmark_tr.name / str(i)

    def test_load_test_runs_dse_sorted(self, slurm_system: SlurmSystem, dse_tr: TestRun) -> None:
        reporter = PerTestReporter(
            slurm_system,
            TestScenario(name="test_scenario", test_runs=[dse_tr]),
            slurm_system.output_path,
            ReportConfig(),
        )
        reporter.load_test_runs()

        assert len(reporter.trs) == dse_tr.test.agent_steps * dse_tr.iterations
        for i, tr in enumerate(reporter.trs):
            exp_iter = i // dse_tr.test.agent_steps
            exp_step = i % dse_tr.test.agent_steps
            assert tr.name == dse_tr.name
            assert tr.current_iteration == exp_iter
            assert tr.step == exp_step
            assert tr.output_path == slurm_system.output_path / dse_tr.name / str(exp_iter) / str(exp_step)


def test_create_tarball_preserves_full_name(tmp_path: Path, slurm_system: SlurmSystem) -> None:
    results_dir = tmp_path / "nemo2.0_llama3_70b_fp8_2025-04-16_14-27-45"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "dummy.txt").write_text("test content")

    reporter = TarballReporter(slurm_system, TestScenario(name="dummy", test_runs=[]), results_dir, ReportConfig())
    reporter.create_tarball(results_dir)

    tarball_path = tmp_path / "nemo2.0_llama3_70b_fp8_2025-04-16_14-27-45.tgz"
    assert tarball_path.exists()
    assert tarfile.is_tarfile(tarball_path)

    with tarfile.open(tarball_path, "r:gz") as tar:
        assert f"{results_dir.name}/dummy.txt" in tar.getnames()


@pytest.mark.parametrize(
    "system",
    [
        SlurmSystem(name="slurm", install_path=Path.cwd(), output_path=Path.cwd(), partitions=[], default_partition=""),
        StandaloneSystem(name="standalone", install_path=Path.cwd(), output_path=Path.cwd()),
    ],
)
def test_template_file_path(system: System) -> None:
    reporter = StatusReporter(
        system, TestScenario(name="test_scenario", test_runs=[]), system.output_path, ReportConfig()
    )
    assert (reporter.template_file_path / reporter.template_file).exists()


MY_REPORT_CALLED = 0


class MyReporter(Reporter):
    def generate(self) -> None:
        global MY_REPORT_CALLED
        MY_REPORT_CALLED += 1


class TestGenerateReport:
    @pytest.fixture(autouse=True, scope="class")
    def setup(self):
        reg = Registry()
        orig_reports = copy.deepcopy(reg.scenario_reports)
        orig_configs = copy.deepcopy(reg.report_configs)
        reg.scenario_reports.clear()
        reg.report_configs.clear()

        reg.add_scenario_report("sr1", MyReporter, ReportConfig(enable=True))

        yield

        reg.scenario_reports.clear()
        reg.scenario_reports.update(orig_reports)
        reg.report_configs.clear()
        reg.report_configs.update(orig_configs)

    @pytest.fixture(autouse=True)
    def reset(self):
        global MY_REPORT_CALLED
        MY_REPORT_CALLED = 0

    def test_default_flow(self, slurm_system: SlurmSystem) -> None:
        generate_reports(slurm_system, TestScenario(name="ts", test_runs=[]), slurm_system.output_path)
        assert MY_REPORT_CALLED == 1

    def test_disabled_by_default(self, slurm_system: SlurmSystem) -> None:
        Registry().update_scenario_report("sr1", MyReporter, ReportConfig(enable=False))
        generate_reports(slurm_system, TestScenario(name="ts", test_runs=[]), slurm_system.output_path)
        assert MY_REPORT_CALLED == 0

    def test_disabled_on_system_level(self, slurm_system: SlurmSystem) -> None:
        slurm_system.reports = {"sr1": ReportConfig(enable=False)}
        generate_reports(slurm_system, TestScenario(name="ts", test_runs=[]), slurm_system.output_path)
        assert MY_REPORT_CALLED == 0


class TestGenerateReportPriority:
    @pytest.fixture(autouse=True)
    def setup(self):
        reg = Registry()
        orig_reports = copy.deepcopy(reg.scenario_reports)
        orig_configs = copy.deepcopy(reg.report_configs)
        reg.scenario_reports.clear()
        reg.report_configs.clear()

        global MY_REPORT_CALLED
        MY_REPORT_CALLED = 0

        yield

        reg.scenario_reports.clear()
        reg.scenario_reports.update(orig_reports)
        reg.report_configs.clear()
        reg.report_configs.update(orig_configs)

    def test_non_registered_report_is_ignored(self, slurm_system: SlurmSystem) -> None:
        generate_reports(slurm_system, TestScenario(name="ts", test_runs=[]), slurm_system.output_path)
        assert MY_REPORT_CALLED == 0

    def test_report_is_enabled_on_system_level(self, slurm_system: SlurmSystem) -> None:
        Registry().add_scenario_report("sr1", MyReporter, ReportConfig(enable=True))
        slurm_system.reports = {"sr1": ReportConfig(enable=True)}
        generate_reports(slurm_system, TestScenario(name="ts", test_runs=[]), slurm_system.output_path)
        assert MY_REPORT_CALLED == 1

    def test_report_is_enabled_on_scenario_level(self, slurm_system: SlurmSystem) -> None:
        Registry().add_scenario_report("sr1", MyReporter, ReportConfig(enable=True))
        slurm_system.reports = {}
        generate_reports(
            slurm_system,
            TestScenario(name="ts", test_runs=[], reports={"sr1": ReportConfig(enable=True)}),
            slurm_system.output_path,
        )
        assert MY_REPORT_CALLED == 1

    def test_report_scenario_has_highest_priority(self, slurm_system: SlurmSystem) -> None:
        Registry().add_scenario_report("sr1", MyReporter, ReportConfig(enable=True))
        slurm_system.reports = {"sr1": ReportConfig(enable=False)}
        generate_reports(
            slurm_system,
            TestScenario(name="ts", test_runs=[], reports={"sr1": ReportConfig(enable=True)}),
            slurm_system.output_path,
        )
        assert MY_REPORT_CALLED == 1


@pytest.fixture
def slurm_metadata() -> SlurmSystemMetadata:
    return SlurmSystemMetadata(
        user="user",
        system=MetadataSystem(
            os_type="os_type",
            os_version="os_version",
            linux_kernel_version="linux_kernel_version",
            gpu_arch_type="gpu_arch_type",
            cpu_model_name="cpu_model_name",
            cpu_arch_type="cpu_arch_type",
        ),
        mpi=MetadataMPI(
            mpi_type="mpi_type",
            mpi_version="mpi_version",
            hpcx_version="hpcx_version",
        ),
        cuda=MetadataCUDA(
            cuda_build_version="cuda_build_version",
            cuda_runtime_version="cuda_runtime_version",
            cuda_driver_version="cuda_driver_version",
        ),
        network=MetadataNetwork(
            nics="nics",
            switch_type="switch_type",
            network_name="network_name",
            mofed_version="mofed_version",
            libfabric_version="libfabric_version",
        ),
        nccl=MetadataNCCL(
            version="1.1.1",
            commit_sha="abcdef15",
        ),
        slurm=MetadataSlurm(
            cluster_name="cluster_name",
            node_list="node1,node2",
            num_nodes="2",
            ntasks_per_node="8",
            ntasks="16",
            job_id="123456",
        ),
    )


class TestSlurmReportItem:
    def test_no_metadata_folder(self, slurm_system: SlurmSystem) -> None:
        run_dir = slurm_system.output_path / "run_dir"
        run_dir.mkdir(parents=True, exist_ok=True)
        tr = TestRun(
            name="run_dir",
            test=NCCLTestDefinition(
                name="nccl",
                description="NCCL test",
                test_template_name="NcclTest",
                cmd_args=NCCLCmdArgs(docker_image_url="fake://url/nccl"),
            ),
            num_nodes=1,
            nodes=["node1"],
            output_path=run_dir,
        )

        [report_item] = ReportItem.from_test_runs([tr], slurm_system.output_path)
        assert report_item.nodes is None

    def test_no_metadata_files(self, slurm_system: SlurmSystem) -> None:
        run_dir = slurm_system.output_path / "run_dir"
        (run_dir / "metadata").mkdir(parents=True, exist_ok=True)
        tr = TestRun(
            name="run_dir",
            test=NCCLTestDefinition(
                name="nccl",
                description="NCCL test",
                test_template_name="NcclTest",
                cmd_args=NCCLCmdArgs(docker_image_url="fake://url/nccl"),
            ),
            num_nodes=1,
            nodes=["node1"],
            output_path=run_dir,
        )

        [report_item] = ReportItem.from_test_runs([tr], slurm_system.output_path)
        assert report_item.nodes is None

    def test_metadata_file_in_run_dir(self, slurm_system: SlurmSystem, slurm_metadata: SlurmSystemMetadata) -> None:
        run_dir = slurm_system.output_path / "run_dir"
        (run_dir / "metadata").mkdir(parents=True, exist_ok=True)
        with open(run_dir / "metadata" / "node-0.toml", "w") as f:
            toml.dump(slurm_metadata.model_dump(), f)
        tr = TestRun(
            name="run_dir",
            test=NCCLTestDefinition(
                name="nccl",
                description="NCCL test",
                test_template_name="NcclTest",
                cmd_args=NCCLCmdArgs(docker_image_url="fake://url/nccl"),
            ),
            num_nodes=1,
            nodes=["node1"],
            output_path=run_dir,
        )

        [report_item] = ReportItem.from_test_runs([tr], slurm_system.output_path)
        assert report_item.nodes == slurm_metadata.slurm.node_list

    def test_metadata_for_single_sbatch(self, slurm_system: SlurmSystem, slurm_metadata: SlurmSystemMetadata) -> None:
        run_dir = slurm_system.output_path / "run_dir"
        run_dir.mkdir(parents=True, exist_ok=True)
        (slurm_system.output_path / "metadata").mkdir(parents=True, exist_ok=True)
        with open(slurm_system.output_path / "metadata" / "node-0.toml", "w") as f:
            toml.dump(slurm_metadata.model_dump(), f)
        tr = TestRun(
            name="run_dir",
            test=NCCLTestDefinition(
                name="nccl",
                description="NCCL test",
                test_template_name="NcclTest",
                cmd_args=NCCLCmdArgs(docker_image_url="fake://url/nccl"),
            ),
            num_nodes=1,
            nodes=["node1"],
            output_path=run_dir,
        )

        [report_item] = ReportItem.from_test_runs([tr], slurm_system.output_path)
        assert report_item.nodes == slurm_metadata.slurm.node_list


def test_report_order() -> None:
    reports = Registry().ordered_scenario_reports()
    assert reports[0][0] == "per_test"
    assert reports[-3][0] == "status"
    assert reports[-2][0] == "dse"
    assert reports[-1][0] == "tarball"


def _write_slurm_job(step_dir: Path, elapsed_time_sec: int) -> None:
    metadata = SlurmJobMetadata(
        job_id=12345,
        name=step_dir.name,
        state="COMPLETED",
        start_time="2026-03-24T12:00:00",
        end_time="2026-03-24T12:05:00",
        elapsed_time_sec=elapsed_time_sec,
        exit_code="0:0",
        srun_cmd="srun echo test",
        test_cmd="echo test",
        is_single_sbatch=False,
        job_root=step_dir,
        job_steps=[
            SlurmStepMetadata(
                job_id=12345,
                step_id="0",
                name=step_dir.name,
                state="COMPLETED",
                start_time="2026-03-24T12:00:00",
                end_time="2026-03-24T12:05:00",
                elapsed_time_sec=elapsed_time_sec,
                exit_code="0:0",
                submit_line="srun echo test",
            )
        ],
    )
    with (step_dir / "slurm-job.toml").open("w") as f:
        toml.dump(metadata.model_dump(mode="json"), f)


def _write_slurm_system_metadata(step_dir: Path, slurm_metadata: SlurmSystemMetadata) -> None:
    metadata_dir = step_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with (metadata_dir / "node-0.toml").open("w") as f:
        toml.dump(slurm_metadata.model_dump(), f)


def _create_non_dse_iteration(case: TestRun, iteration: int, results_root: Path) -> None:
    iteration_dir = results_root / case.name / str(iteration)
    iteration_dir.mkdir(parents=True, exist_ok=True)


def _create_dse_iteration(
    case: TestRun,
    iteration: int,
    results_root: Path,
    slurm_metadata: SlurmSystemMetadata,
    steps: list[dict[str, Any]],
) -> None:
    iteration_dir = results_root / case.name / str(iteration)
    iteration_dir.mkdir(parents=True, exist_ok=True)

    with (iteration_dir / "trajectory.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "action", "reward", "observation"])
        for step in steps:
            step_no = step["step"]
            writer.writerow([step_no, step["action"], step["reward"], step["observation"]])

            step_dir = iteration_dir / str(step_no)
            step_dir.mkdir(parents=True, exist_ok=True)
            _write_slurm_job(step_dir, int(step["elapsed_time_sec"]))
            _write_slurm_system_metadata(step_dir, slurm_metadata)

            # NCCLTestDefinition.was_run_successful
            (step_dir / "stdout.txt").write_text("# Out of bounds values# Avg bus bandwidth")

            step_tr = case.apply_params_set(step["action"])
            step_tr.current_iteration = iteration
            step_tr.step = step_no
            step_tr.output_path = step_dir
            with (step_dir / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME).open("w") as dump_file:
                toml.dump(TestRunDetails.from_test_run(step_tr, "", "").model_dump(mode="json"), dump_file)


def test_dse_reporter(
    slurm_system: SlurmSystem,
    slurm_metadata: SlurmSystemMetadata,
) -> None:
    slurm_metadata.system.gpu_arch_type = "NVIDIA H100 80GB HBM3"

    dse_case = TestRun(
        name="dse-case",
        test=NCCLTestDefinition(
            name="nccl",
            description="NCCL case",
            test_template_name="NcclTest",
            cmd_args=NCCLCmdArgs(docker_image_url="fake://url/nccl", ngpus=[1, 2]),
            extra_env_vars={"VAR1": ["value1", "value2"]},
            agent_steps=3,
        ),
        num_nodes=1,
        nodes=["node1"],
        iterations=1,
    )

    steps = [
        {
            "step": 0,
            "action": {"ngpus": 1, "extra_env_vars.VAR1": "value1"},
            "reward": -10.0,
            "observation": [10],
            "elapsed_time_sec": 60,
        },
        {
            "step": 1,
            "action": {"ngpus": 2, "extra_env_vars.VAR1": "value1"},
            "reward": -5.0,
            "observation": [5],
            "elapsed_time_sec": 120,
        },
        {
            "step": 2,
            "action": {"ngpus": 2, "extra_env_vars.VAR1": "value2"},
            "reward": -7.0,
            "observation": [7],
            "elapsed_time_sec": 180,
        },
    ]
    _create_dse_iteration(
        dse_case,
        iteration=0,
        results_root=slurm_system.output_path,
        slurm_metadata=slurm_metadata,
        steps=steps,
    )

    scenario = TestScenario(
        name="single-dse-scenario",
        test_runs=[dse_case],
    )
    reporter = DSEReporter(slurm_system, scenario, slurm_system.output_path, ReportConfig())
    reporter.load_test_runs()

    summaries = build_dse_summaries(
        system=slurm_system,
        results_root=slurm_system.output_path,
        loaded_test_runs=reporter.trs,
        test_cases=scenario.test_runs,
    )

    best_tr = dse_case.apply_params_set({"ngpus": 2, "extra_env_vars.VAR1": "value1"})
    best_tr.current_iteration = 0
    best_tr.step = 1
    expected = {
        "name": "dse-case-0",
        "saved_time": "2m",
        "saved_gpu_hours": "0.27",
        "saved_usd": "$0.80",
        "gpu_label": "NVIDIA H100 80GB HBM3",
        "avg_step_runtime": "2m",
        "observed_runtime": "6m",
        "efficiency_ratio": "~1.3x",
        "efficiency_steps": "3 / 4 steps",
        "best_config_toml": toml.dumps(TestRunDetails.from_test_run(best_tr, "", "").test_definition.model_dump()),
        "parameter_rows": [
            {
                "name": "ngpus",
                "values": [
                    {"text": "1", "is_best": False},
                    {"text": "2", "is_best": True},
                ],
            },
            {
                "name": "extra_env_vars.VAR1",
                "values": [
                    {"text": "value1", "is_best": True},
                    {"text": "value2", "is_best": False},
                ],
            },
        ],
        "reward_chart_data": {
            "labels": [0, 1, 2],
            "rewards": [-10.0, -5.0, -7.0],
            "observations": ["10", "5", "7"],
            "best_index": 0,
        },
    }
    assert len(summaries) == 1
    assert asdict(summaries[0]) == expected

    reporter.generate()

    assert (slurm_system.output_path / "single-dse-scenario-dse-report.html").exists()
    assert (slurm_system.output_path / dse_case.name / "0" / f"{dse_case.name}.toml").exists()
