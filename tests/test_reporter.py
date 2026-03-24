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
from pathlib import Path

import pytest
import toml

from cloudai import TestRun, TestScenario
from cloudai.cli.handlers import generate_reports
from cloudai.core import CommandGenStrategy, Registry, Reporter, System
from cloudai.models.scenario import ReportConfig, TestRunDetails
from cloudai.report_generator.status_report import DSEReportBuilder, ReportItem, _build_effort_chart_data, load_system_metadata
from cloudai.reporter import PerTestReporter, StatusReporter, TarballReporter
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


def _write_successful_nccl_stdout(step_dir: Path) -> None:
    (step_dir / "stdout.txt").write_text("# Out of bounds values\n# Avg bus bandwidth\n")


def _write_slurm_job_metadata(step_dir: Path, elapsed_time_sec: int) -> None:
    slurm_job = {
        "job_id": 123456,
        "name": "test-job",
        "state": "COMPLETED",
        "start_time": "2026-03-21T15:00:00",
        "end_time": "2026-03-21T15:05:00",
        "elapsed_time_sec": elapsed_time_sec,
        "exit_code": "0:0",
        "srun_cmd": "srun echo test",
        "test_cmd": "echo test",
        "is_single_sbatch": False,
        "job_root": str(step_dir),
        "job_steps": [],
    }
    with (step_dir / "slurm-job.toml").open("w") as f:
        toml.dump(slurm_job, f)


def _write_step_metadata(step_dir: Path, metadata: SlurmSystemMetadata) -> None:
    metadata_dir = step_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with (metadata_dir / "node-0.toml").open("w") as f:
        toml.dump(metadata.model_dump(), f)


def _create_dse_report_fixture(
    slurm_system: SlurmSystem,
    slurm_metadata: SlurmSystemMetadata,
    gpu_name: str = "NVIDIA H100 80GB HBM3",
) -> TestRun:
    test_definition = NCCLTestDefinition(
        name="dse-nccl",
        description="DSE summary sample",
        test_template_name="NcclTest",
        cmd_args=NCCLCmdArgs(
            docker_image_url="fake://url/nccl",
            subtest_name="all_reduce_perf_mpi",
            nthreads=[1, 2],
            datatype=["float", "uint8"],
            blocking=[0, 1],
        ),
        agent_steps=3,
    )
    tr = TestRun(
        name="dse-report",
        test=test_definition,
        num_nodes=2,
        nodes=["node1", "node2"],
        time_limit="00:05:00",
    )
    iter_dir = slurm_system.output_path / tr.name / "0"
    iter_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        (1, {"nthreads": 1, "datatype": "float", "blocking": 0}, 1.5, [2.5], 10),
        (2, {"nthreads": 2, "datatype": "uint8", "blocking": 1}, 3.0, [1.2], 20),
        (3, {"nthreads": 2, "datatype": "float", "blocking": 1}, 2.0, [1.8], 30),
    ]

    with (iter_dir / "trajectory.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "action", "reward", "observation"])
        for step, action, reward, observation, _elapsed in rows:
            writer.writerow([step, action, reward, observation])

    for step, action, _reward, _observation, elapsed in rows:
        step_dir = iter_dir / str(step)
        step_dir.mkdir(parents=True, exist_ok=True)
        step_tr = tr.apply_params_set(action)
        step_tr.step = step
        step_tr.output_path = step_dir

        with (step_dir / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME).open("w") as f:
            toml.dump(TestRunDetails.from_test_run(step_tr, "", "").model_dump(), f)

        _write_successful_nccl_stdout(step_dir)
        _write_slurm_job_metadata(step_dir, elapsed)

    metadata = slurm_metadata.model_copy(deep=True)
    metadata.system.gpu_arch_type = gpu_name
    _write_step_metadata(iter_dir / "2", metadata)
    (iter_dir / "analysis.csv").write_text("parameter,sensitivity,importance\nblocking,0.5,0.8\n")

    return tr


def _build_dse_summaries(
    slurm_system: SlurmSystem,
    dse_tr: TestRun,
    scenario_name: str = "dse_scenario",
) -> tuple[StatusReporter, list]:
    reporter = StatusReporter(
        slurm_system,
        TestScenario(name=scenario_name, test_runs=[dse_tr]),
        slurm_system.output_path,
        ReportConfig(),
    )
    reporter.load_test_runs()
    summaries = DSEReportBuilder(slurm_system, slurm_system.output_path, reporter.trs).build([dse_tr])
    return reporter, summaries


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
    reporter, summaries = _build_dse_summaries(slurm_system, dse_tr, scenario_name="test_scenario")
    assert len(summaries) == dse_tr.iterations
    best_config_path = (
        reporter.results_root
        / dse_tr.name
        / f"{dse_tr.current_iteration}"
        / DSEReportBuilder.best_config_file_name(dse_tr)
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
    assert (reporter.templates_dir / "general-report.jinja2").exists()


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


class TestLoadSystemMetadata:
    def test_no_metadata_folder(self, slurm_system: SlurmSystem) -> None:
        run_dir = slurm_system.output_path / "run_dir"
        run_dir.mkdir(parents=True, exist_ok=True)

        meta = load_system_metadata(run_dir, slurm_system.output_path)
        assert meta is None

    def test_no_metadata_files(self, slurm_system: SlurmSystem) -> None:
        run_dir = slurm_system.output_path / "run_dir"
        (run_dir / "metadata").mkdir(parents=True, exist_ok=True)

        meta = load_system_metadata(run_dir, slurm_system.output_path)
        assert meta is None

    def test_metadata_file_in_run_dir(self, slurm_system: SlurmSystem, slurm_metadata: SlurmSystemMetadata) -> None:
        run_dir = slurm_system.output_path / "run_dir"
        (run_dir / "metadata").mkdir(parents=True, exist_ok=True)
        with open(run_dir / "metadata" / "node-0.toml", "w") as f:
            toml.dump(slurm_metadata.model_dump(), f)

        meta = load_system_metadata(run_dir, slurm_system.output_path)
        assert meta is not None
        assert meta.slurm.node_list == slurm_metadata.slurm.node_list

    def test_metadata_for_single_sbatch(self, slurm_system: SlurmSystem, slurm_metadata: SlurmSystemMetadata) -> None:
        run_dir = slurm_system.output_path / "run_dir"
        run_dir.mkdir(parents=True, exist_ok=True)
        (slurm_system.output_path / "metadata").mkdir(parents=True, exist_ok=True)
        with open(slurm_system.output_path / "metadata" / "node-0.toml", "w") as f:
            toml.dump(slurm_metadata.model_dump(), f)

        meta = load_system_metadata(run_dir, slurm_system.output_path)
        assert meta is not None
        assert meta.slurm.node_list == slurm_metadata.slurm.node_list


def test_report_item_from_test_runs_includes_logs_and_metadata(
    slurm_system: SlurmSystem, benchmark_tr: TestRun, slurm_metadata: SlurmSystemMetadata
) -> None:
    run_dir = slurm_system.output_path / benchmark_tr.name / "0"
    metadata_dir = run_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_dir / "node-0.toml", "w") as f:
        toml.dump(slurm_metadata.model_dump(), f)

    benchmark_tr.output_path = run_dir
    items = ReportItem.from_test_runs([benchmark_tr], slurm_system.output_path)

    assert len(items) == 1
    assert items[0].logs_path == f"./{benchmark_tr.name}/0"
    assert items[0].nodes is not None
    assert items[0].nodes.slurm.node_list == slurm_metadata.slurm.node_list
    assert items[0].status_text == "FAILED"
    assert items[0].status_class == "failed"


def test_report_order() -> None:
    reports = Registry().ordered_scenario_reports()
    assert reports[0][0] == "per_test"
    assert reports[-2][0] == "status"
    assert reports[-1][0] == "tarball"


def test_dse_summary_and_best_config_artifacts(slurm_system: SlurmSystem, slurm_metadata: SlurmSystemMetadata) -> None:
    dse_tr = _create_dse_report_fixture(slurm_system, slurm_metadata)
    _, summaries = _build_dse_summaries(slurm_system, dse_tr)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.total_space == 8
    assert summary.executed_steps == 3
    assert summary.skipped_steps == 5
    assert summary.best_step == 2
    assert summary.best_reward == pytest.approx(3.0)
    assert summary.avg_step_duration_sec == pytest.approx(20.0)
    assert summary.total_runtime_sec == pytest.approx(60.0)
    assert summary.saved_runtime_sec == pytest.approx(100.0)
    assert summary.saved_gpu_hours == pytest.approx((100.0 / 3600.0) * 16)
    assert summary.estimated_saved_cost_usd == pytest.approx((summary.saved_gpu_hours or 0) * 4.5)
    assert summary.analysis_rel_path is not None
    assert summary.best_config_rel_path == f"./{dse_tr.name}/0/{dse_tr.name}.toml"
    assert summary.reward_chart_data is not None
    assert summary.reward_chart_data["labels"] == [1, 2, 3]
    assert summary.reward_chart_data["rewards"] == pytest.approx([1.5, 3.0, 2.0])
    assert summary.reward_chart_data["running_best"] == pytest.approx([1.5, 3.0, 3.0])
    assert summary.reward_chart_data["observations"] == ["2.5", "1.2", "1.8"]
    assert summary.effort_chart_data is not None
    assert summary.effort_chart_data["explored_ratio"] == pytest.approx(3 / 8)
    assert summary.effort_chart_data["explored_percent"] == pytest.approx(37.5)
    assert summary.effort_chart_data["avoided_percent"] == pytest.approx(62.5)
    assert summary.effort_chart_data["reduction_factor"] == pytest.approx(8 / 3)
    assert summary.effort_chart_data["executed_steps"] == 3
    assert summary.effort_chart_data["total_space"] == 8

    best_values = {row.name: row.best_value for row in summary.parameter_rows}
    assert best_values["nthreads"] == "2"
    assert best_values["datatype"] == "uint8"
    assert best_values["blocking"] == "1"

    best_config_path = slurm_system.output_path / dse_tr.name / "0" / DSEReportBuilder.best_config_file_name(dse_tr)
    assert best_config_path.exists()

    best_config = toml.load(best_config_path)
    assert best_config["agent_steps"] == 3
    assert best_config["cmd_args"]["datatype"] == "uint8"
    assert best_config["cmd_args"]["blocking"] == 1
    assert best_config["cmd_args"]["nthreads"] == 2

    inline_best_config = toml.loads(summary.best_config_toml or "")
    assert inline_best_config["cmd_args"]["datatype"] == "uint8"
    assert inline_best_config["cmd_args"]["blocking"] == 1
    assert inline_best_config["cmd_args"]["nthreads"] == 2


def test_dse_generate_scenario_report_renders_html(
    slurm_system: SlurmSystem, slurm_metadata: SlurmSystemMetadata
) -> None:
    dse_tr = _create_dse_report_fixture(slurm_system, slurm_metadata)
    reporter = StatusReporter(
        slurm_system,
        TestScenario(name="dse_scenario", test_runs=[dse_tr]),
        slurm_system.output_path,
        ReportConfig(),
    )

    reporter.generate()

    report_path = slurm_system.output_path / "dse_scenario.html"
    html = report_path.read_text()
    assert "cdn.jsdelivr.net/npm/chart.js" in html
    assert "Saved GPU-Hours" in html
    assert "Exploration Efficiency" in html
    assert "3 / 8 steps" in html
    assert "reduction in search space" in html
    assert "Reward Over Steps" in html
    assert "Best Test TOML" in html
    assert "Show best config TOML" in html
    assert "Copy TOML" in html
    assert "BO Analysis" in html
    assert "All Steps" in html
    assert "dse-report.toml" in html
    assert "efficiency-ratio" in html
    assert "js-reward-chart" in html
    assert "chart-shell" in html
    assert 'class="value-pill value-pill--selected"' in html
    assert "Execution Context" not in html
    assert "Exploration Mix" not in html
    assert "Skipped" not in html
    assert "Coverage" not in html
    assert "GPU Family" not in html
    assert "<th>Best</th>" not in html
    assert "status-pill--passed" in html
    assert "1m 40s" in html


def test_effort_chart_uses_break_for_large_search_space() -> None:
    chart_data = _build_effort_chart_data(30, 100_000)

    assert chart_data is not None
    assert chart_data["explored_percent"] == pytest.approx(0.03)
    assert chart_data["avoided_percent"] == pytest.approx(99.97)
    assert chart_data["reduction_factor"] == pytest.approx(100_000 / 30)


def test_dse_console_summary_is_compact(
    slurm_system: SlurmSystem, slurm_metadata: SlurmSystemMetadata, caplog: pytest.LogCaptureFixture
) -> None:
    dse_tr = _create_dse_report_fixture(slurm_system, slurm_metadata)
    reporter, summaries = _build_dse_summaries(slurm_system, dse_tr)
    with caplog.at_level("INFO"):
        reporter.to_console(summaries)

    assert "steps=3/8" in caplog.text
    assert "best_step=2" in caplog.text
    assert "dse-report.toml" in caplog.text
    assert "step=1" not in caplog.text


def test_unknown_gpu_family_omits_estimated_cost(
    slurm_system: SlurmSystem, slurm_metadata: SlurmSystemMetadata
) -> None:
    dse_tr = _create_dse_report_fixture(slurm_system, slurm_metadata, gpu_name="Mystery GPU")
    _reporter, summaries = _build_dse_summaries(slurm_system, dse_tr)

    assert summaries[0].estimated_saved_cost_usd is None
