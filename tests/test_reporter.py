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

import copy
import tarfile
from pathlib import Path

import pytest
import toml

from cloudai import TestRun, TestScenario
from cloudai.cli.handlers import generate_reports
from cloudai.core import Registry, Reporter, System
from cloudai.models.scenario import ReportConfig
from cloudai.reporter import PerTestReporter, SlurmReportItem, StatusReporter, TarballReporter
from cloudai.systems.slurm.slurm_metadata import (
    MetadataCUDA,
    MetadataMPI,
    MetadataNCCL,
    MetadataNetwork,
    MetadataSlurm,
    MetadataSystem,
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


def test_best_dse_config(dse_tr: TestRun, slurm_system: SlurmSystem) -> None:
    reporter = StatusReporter(
        slurm_system, TestScenario(name="test_scenario", test_runs=[dse_tr]), slurm_system.output_path, ReportConfig()
    )
    reporter.report_best_dse_config()
    best_config_path = (
        reporter.results_root / dse_tr.name / f"{dse_tr.current_iteration}" / reporter.best_dse_config_file_name(dse_tr)
    )
    assert best_config_path.exists()
    nccl = NCCLTestDefinition.model_validate(toml.load(best_config_path))
    assert isinstance(nccl.cmd_args, NCCLCmdArgs)
    assert nccl.agent_steps == 12


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
        reg.scenario_reports.clear()

        reg.add_scenario_report("sr1", MyReporter, ReportConfig(enable=True))

        yield

        reg.scenario_reports.clear()
        reg.scenario_reports.update(orig_reports)

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
        reg.scenario_reports.clear()

        global MY_REPORT_CALLED
        MY_REPORT_CALLED = 0

        yield

        reg.scenario_reports.clear()
        reg.scenario_reports.update(orig_reports)

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

        meta = SlurmReportItem.get_metadata(run_dir, slurm_system.output_path)
        assert meta is None

    def test_no_metadata_files(self, slurm_system: SlurmSystem) -> None:
        run_dir = slurm_system.output_path / "run_dir"
        (run_dir / "metadata").mkdir(parents=True, exist_ok=True)

        meta = SlurmReportItem.get_metadata(run_dir, slurm_system.output_path)
        assert meta is None

    def test_metadata_file_in_run_dir(self, slurm_system: SlurmSystem, slurm_metadata: SlurmSystemMetadata) -> None:
        run_dir = slurm_system.output_path / "run_dir"
        (run_dir / "metadata").mkdir(parents=True, exist_ok=True)
        with open(run_dir / "metadata" / "node-0.toml", "w") as f:
            toml.dump(slurm_metadata.model_dump(), f)

        meta = SlurmReportItem.get_metadata(run_dir, slurm_system.output_path)
        assert meta is not None
        assert meta.slurm.node_list == slurm_metadata.slurm.node_list

    def test_metadata_for_single_sbatch(self, slurm_system: SlurmSystem, slurm_metadata: SlurmSystemMetadata) -> None:
        run_dir = slurm_system.output_path / "run_dir"
        run_dir.mkdir(parents=True, exist_ok=True)
        (slurm_system.output_path / "metadata").mkdir(parents=True, exist_ok=True)
        with open(slurm_system.output_path / "metadata" / "node-0.toml", "w") as f:
            toml.dump(slurm_metadata.model_dump(), f)

        meta = SlurmReportItem.get_metadata(run_dir, slurm_system.output_path)
        assert meta is not None
        assert meta.slurm.node_list == slurm_metadata.slurm.node_list


def test_report_order() -> None:
    reports = Registry().ordered_scenario_reports()
    assert reports[0][0] == "per_test"
    assert reports[-2][0] == "status"
    assert reports[-1][0] == "tarball"
