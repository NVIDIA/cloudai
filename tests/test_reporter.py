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

import defusedxml.ElementTree as ET
import pytest
import toml

from cloudai import TestRun, TestScenario
from cloudai.core import CommandGenStrategy, Registry, Reporter, System
from cloudai.handlers import generate_reports
from cloudai.models.scenario import ReportConfig, TestRunDetails
from cloudai.report_generator.dse_report import build_dse_summaries
from cloudai.reporter import (
    DSEReporter,
    JUnitReporter,
    PerTestReporter,
    ReportItem,
    StatusReporter,
    TarballReporter,
)
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
from tests.conftest import selected_dump_cases


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

    def test_records_passing_status(self, slurm_system: SlurmSystem, monkeypatch: pytest.MonkeyPatch) -> None:
        from cloudai.core import JobStatusResult

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
        monkeypatch.setattr(type(tr.test), "was_run_successful", lambda self, tr: JobStatusResult(True, ""))

        [report_item] = ReportItem.from_test_runs([tr], slurm_system.output_path)
        assert report_item.is_successful is True
        assert report_item.error_message == ""

    def test_records_failing_status_and_message(
        self, slurm_system: SlurmSystem, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cloudai.core import JobStatusResult

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
        monkeypatch.setattr(
            type(tr.test), "was_run_successful", lambda self, tr: JobStatusResult(False, "command failed")
        )

        [report_item] = ReportItem.from_test_runs([tr], slurm_system.output_path)
        assert report_item.is_successful is False
        assert report_item.error_message == "command failed"


def test_scenario_report_shows_pass_fail_status(
    slurm_system: SlurmSystem, benchmark_tr: TestRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for the bug where the saved HTML report never recorded pass/fail,
    only the terminal summary did."""
    from cloudai.core import JobStatusResult

    monkeypatch.setattr(
        type(benchmark_tr.test), "was_run_successful", lambda self, tr: JobStatusResult(False, "command failed")
    )

    reporter = StatusReporter(
        slurm_system,
        TestScenario(name="test-scenario", test_runs=[benchmark_tr]),
        slurm_system.output_path,
        ReportConfig(),
    )
    reporter.generate()

    report_html = (slurm_system.output_path / "test-scenario.html").read_text()
    assert "FAILED" in report_html
    assert "command failed" in report_html


def test_scenario_report_escapes_error_message(
    slurm_system: SlurmSystem, benchmark_tr: TestRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Error messages are attacker/tool-controlled output and must be escaped before
    being embedded in the saved HTML report."""
    from cloudai.core import JobStatusResult

    monkeypatch.setattr(
        type(benchmark_tr.test),
        "was_run_successful",
        lambda self, tr: JobStatusResult(False, "<script>alert(1)</script>"),
    )

    reporter = StatusReporter(
        slurm_system,
        TestScenario(name="test-scenario", test_runs=[benchmark_tr]),
        slurm_system.output_path,
        ReportConfig(),
    )
    reporter.generate()

    report_html = (slurm_system.output_path / "test-scenario.html").read_text()
    assert "<script>alert(1)</script>" not in report_html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in report_html


def test_report_order() -> None:
    reports = Registry().ordered_scenario_reports()
    assert reports[0][0] == "per_test"
    assert reports[-4][0] == "status"
    assert reports[-3][0] == "dse"
    assert reports[-2][0] == "tarball"
    assert reports[-1][0] == "s3"


def test_junit_reporter_generates_testcases_with_status_logs_and_duration(
    slurm_system: SlurmSystem, benchmark_tr: TestRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dirs = [slurm_system.output_path / benchmark_tr.name / str(i) for i in range(benchmark_tr.iterations)]
    for i, run_dir in enumerate(run_dirs):
        (run_dir / "stdout.txt").write_text(f"stdout {i}\n")
        (run_dir / "stderr.txt").write_text(f"stderr {i}\n")
        _write_slurm_job(run_dir, i + 1)

    statuses = {
        run_dirs[0]: (True, ""),
        run_dirs[1]: (False, "benchmark failed\x01"),
        run_dirs[2]: (True, ""),
    }

    def was_run_successful(tr: TestRun):
        successful, message = statuses[tr.output_path]
        from cloudai.core import JobStatusResult

        return JobStatusResult(successful, message)

    monkeypatch.setattr(type(benchmark_tr.test), "was_run_successful", lambda self, tr: was_run_successful(tr))
    reporter = JUnitReporter(
        slurm_system,
        TestScenario(name="test-scenario", test_runs=[benchmark_tr]),
        slurm_system.output_path,
        ReportConfig(),
    )
    reporter.generate()

    root = ET.parse(slurm_system.output_path / "junit.xml").getroot()
    assert root is not None
    suite = root.find("testsuite")
    assert root.attrib == {
        "name": "test-scenario",
        "tests": "3",
        "failures": "1",
        "errors": "0",
        "skipped": "0",
        "time": "6.000",
    }
    assert suite is not None
    cases = suite.findall("testcase")
    assert [case.attrib for case in cases] == [
        {"name": "benchmark", "classname": "test-scenario", "time": "1.000"},
        {"name": "benchmark iter=1", "classname": "test-scenario", "time": "2.000"},
        {"name": "benchmark iter=2", "classname": "test-scenario", "time": "3.000"},
    ]
    assert cases[0].findtext("system-out") == "stdout 0\n"
    failure = cases[1].find("failure")
    assert failure is not None
    assert failure.attrib["message"] == "benchmark failed"
    assert cases[1].findtext("system-err") == "stderr 1\n"


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


@pytest.mark.parametrize("dump_conflict", [False, True, "nonwinning"])
def test_dse_reporter(
    slurm_system: SlurmSystem, slurm_metadata: SlurmSystemMetadata, dump_conflict: bool | str
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
    if dump_conflict:
        from cloudai.configurator import TrajectoryLoadError

        dump = (
            slurm_system.output_path
            / dse_case.name
            / "0"
            / ("1" if dump_conflict is True else "2")
            / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME
        )
        recorded = toml.loads(dump.read_text())
        recorded["test_definition"]["cmd_args"]["ngpus"] = 1
        dump.write_text(toml.dumps(recorded))
        export = dump.parent.parent / f"{dse_case.name}.toml"
        if dump_conflict == "nonwinning":
            export.write_text('preserve = "existing export"\n')
        before = {path: path.read_bytes() for path in dump.parent.parent.rglob("*") if path.is_file()}
        with pytest.raises(TrajectoryLoadError):
            reporter.report_best_dse_config()
        assert {path: path.read_bytes() for path in dump.parent.parent.rglob("*") if path.is_file()} == before
        return

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
            "best_index": 1,
        },
    }
    assert len(summaries) == 1
    assert asdict(summaries[0]) == expected

    reporter.generate()

    assert (slurm_system.output_path / "single-dse-scenario-dse-report.html").exists()
    assert (slurm_system.output_path / dse_case.name / "0" / f"{dse_case.name}.toml").exists()
    best_config = slurm_system.output_path / dse_case.name / "0" / f"{dse_case.name}.toml"
    assert toml.loads(best_config.read_text()) == toml.loads(expected["best_config_toml"])


@pytest.mark.parametrize(
    "archive,change",
    [
        *selected_dump_cases(),
        ("native", "unknown_template-generate"),
        ("native", "missing_template-generate"),
        ("native", "invalid_argument-generate"),
        ("native", "missing_definition-generate"),
        ("native", "relative_equivalent-generate"),
        ("native", "relative_unrelated-generate"),
        ("native", "omitted_nodes_nonempty-generate"),
        ("native", "omitted_nodes_empty-generate"),
        ("native", "malformed_toml-generate"),
        ("native", "read_error-generate"),
    ],
)
def test_best_config_export_verifies_recorded_settings_without_inventing_defaults(
    slurm_system: SlurmSystem, archive: str, change: str, selected_dump_case
) -> None:
    from cloudai.configurator import Trajectory, TrajectoryLoadError
    from cloudai.core import TestScenario
    from cloudai.models.scenario import ReportConfig

    use_generate = change.endswith("-generate")
    selected, case, iteration, expected, dump, export, previous, before, read_attempts, change, valid = (
        selected_dump_case(archive, change)
    )
    if valid:
        reporter = DSEReporter(
            slurm_system, TestScenario(name="export", test_runs=[case]), slurm_system.output_path, ReportConfig()
        )
        (reporter.generate if use_generate else reporter.report_best_dse_config)()
        actual = toml.loads(export.read_text())
        assert actual == expected
        assert change != "minimal" or NCCLTestDefinition.model_validate(actual).cmd_args.nthreads == 1
        if change == "credentials":
            exported = type(selected.test).model_validate(actual)
            projected = exported.model_dump(mode="json", context={"purpose": "trajectory"})
            original_projection = selected.test.model_dump(mode="json", context={"purpose": "trajectory"})
            assert toml.loads(toml.dumps(projected)) == toml.loads(toml.dumps(original_projection))
            assert selected.test.cmd_args.hf_token == "synthetic-export-hf"
            assert selected.test.cmd_args.wandb_key == "synthetic-export-wandb"
            if archive == "native":
                stored = Trajectory.load(iteration).records()[0].get("env_params", decode_numbers=True)
                assert {key: projected["cmd_args"][key] for key in ("hf_token", "wandb_key", "alias")} == {
                    key: stored[key] for key in ("hf_token", "wandb_key", "alias")
                }
    else:
        with pytest.raises(TrajectoryLoadError):
            reporter = DSEReporter(
                slurm_system, TestScenario(name="export", test_runs=[case]), slurm_system.output_path, ReportConfig()
            )
            (reporter.generate if use_generate else reporter.report_best_dse_config)()
        assert export.read_bytes() == previous
    assert change != "read_error" or read_attempts == [dump]
    assert all(path.read_bytes() == original for path, original in before.items())


@pytest.mark.parametrize("outcome", ["mixed", "all_failed", "no_run", "job_failed"])
def test_best_config_uses_eligible_measurements_without_removing_history(
    slurm_system: SlurmSystem, outcome: str
) -> None:
    from cloudai.configurator import Trajectory
    from cloudai.core import RewardOverrides
    from cloudai.models.output import Metric, Run

    case = TestRun(
        name="measurement-selection",
        test=NCCLTestDefinition(
            name="nccl",
            description="measurement selection",
            test_template_name="NcclTest",
            cmd_args=NCCLCmdArgs(docker_image_url="fake://url/nccl", ngpus=[1, 2]),
            agent_steps=3,
        ),
        num_nodes=1,
        nodes=["node1"],
    )
    iteration = slurm_system.output_path / case.name / "0"
    history = Trajectory(iteration_dir=iteration)
    runs = []
    for step, ngpus, reward in ((1, 1, -1.0), (2, 2, 100.0), (3, 1, -1.0)):
        selected = case.apply_params_set({"ngpus": ngpus})
        selected.step = step
        selected.output_path = iteration / str(step)
        selected.output_path.mkdir(parents=True)
        (selected.output_path / "stdout.txt").write_text("# Out of bounds values# Avg bus bandwidth")
        job = SlurmJobMetadata(
            job_id=step,
            name=case.name,
            state="FAILED" if outcome == "job_failed" else "COMPLETED",
            start_time="2026-01-01T00:00:00",
            end_time="2026-01-01T00:01:00",
            elapsed_time_sec=60,
            exit_code="1:0" if outcome == "job_failed" else "0:0",
            srun_cmd="srun test",
            test_cmd="test",
            job_root=selected.output_path,
            job_steps=[],
        )
        (selected.output_path / "slurm-job.toml").write_text(toml.dumps(job.model_dump(mode="json")))
        (selected.output_path / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME).write_text(
            toml.dumps(TestRunDetails.from_test_run(selected, "", "").model_dump(mode="json"))
        )
        cached = history.find(action={"ngpus": 1}) if step == 3 else None
        history.record(
            test_run=selected,
            action_space=case.param_space,
            action={"ngpus": ngpus},
            reward=reward,
            observation={"default": reward},
            system=slurm_system,
            rewards=RewardOverrides(),
            execution_error="failed after completed output" if step == 2 or outcome == "all_failed" else None,
            cached_result=cached,
            run_output=None
            if outcome == "no_run" or step == 3
            else Run(
                path=str(selected.output_path),
                jobid=f"job-{step}",
                status="failed" if outcome == "job_failed" else "completed",
                step=step,
                iteration=0,
                metrics=[Metric(name="default", value=reward, unit="unknown")],
            ),
        )
        runs.append(selected)
    records = Trajectory.load(iteration).records()
    assert [record.step for record in records] == [1, 2, 3]
    assert [not record.ineligibility_reasons("measurement") for record in records] == (
        [True, False, True] if outcome in ("mixed", "no_run") else [False, False, False]
    )
    assert not any(record.ineligibility_reasons("decision") for record in records)
    assert records[2].get("measurement", decode_numbers=True)["kind"] == "cached"
    assert (
        records[2].get("measurement", decode_numbers=True)["origin"]
        == records[0].get("measurement", decode_numbers=True)["origin"]
    )
    export = iteration / f"{case.name}.toml"
    previous = b'preserve = "existing export"\n'
    export.write_bytes(previous)
    before = {path: path.read_bytes() for path in iteration.rglob("*") if path.is_file() and path != export}
    reporter = DSEReporter(
        slurm_system, TestScenario(name="eligible", test_runs=[case]), slurm_system.output_path, ReportConfig()
    )
    reporter.report_best_dse_config()
    if outcome in ("mixed", "no_run"):
        assert toml.loads(export.read_text())["cmd_args"]["ngpus"] == 1
    else:
        assert export.read_bytes() == previous
    assert {path: path.read_bytes() for path in iteration.rglob("*") if path.is_file() and path != export} == before
    assert len(Trajectory.load(iteration).records()) == 3


@pytest.mark.parametrize("archive", ["legacy", "native"])
@pytest.mark.parametrize(
    "rewards,expected_step",
    [
        pytest.param((9, "010"), 2, id="mixed_leading_zero"),
        pytest.param((9, " 10"), 2, id="mixed_whitespace"),
        pytest.param(("09", "010"), 2, id="lexical_strings"),
        pytest.param((9007199254740992, 9007199254740993), 2, id="native_bigints"),
        pytest.param((10, 10), 1, id="native_tie"),
        pytest.param((10, "010"), 1, id="mixed_numeric_tie"),
        pytest.param((9, " 10.5"), 2, id="whitespace_fraction"),
        pytest.param((9007199254740992, " 9007199254740993"), 2, id="mixed_bigint_string"),
    ],
)
def test_best_config_ranks_numeric_rewards_without_changing_recorded_values(
    slurm_system: SlurmSystem, archive: str, rewards: tuple[int | str, int | str], expected_step: int
) -> None:
    from cloudai.configurator import Trajectory
    from cloudai.core import RewardOverrides, TestRun, TestScenario
    from cloudai.models.output import Metric, Run

    case = TestRun(
        name="reward-order",
        test=NCCLTestDefinition(
            name="nccl",
            description="reward order",
            test_template_name="NcclTest",
            cmd_args=NCCLCmdArgs(docker_image_url="fake://url/nccl", ngpus=[1, 2]),
            agent_steps=2,
        ),
        num_nodes=1,
        nodes=["node1"],
    )
    iteration = slurm_system.output_path / case.name / "0"
    history = Trajectory(iteration_dir=iteration)
    for step, reward in enumerate(rewards, 1):
        selected = case.apply_params_set({"ngpus": step})
        selected.step = step
        selected.output_path = iteration / str(step)
        selected.output_path.mkdir(parents=True)
        (selected.output_path / "stdout.txt").write_text("# Out of bounds values# Avg bus bandwidth")
        _write_slurm_job(selected.output_path, 60)
        (selected.output_path / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME).write_text(
            toml.dumps(TestRunDetails.from_test_run(selected, "", "").model_dump(mode="json"))
        )
        if archive == "native":
            history.record(
                test_run=selected,
                action_space=case.param_space,
                action={"ngpus": step},
                reward=reward,
                observation={"default": 1.0},
                system=slurm_system,
                rewards=RewardOverrides(),
                run_output=Run(
                    path=str(selected.output_path),
                    jobid=f"rank-{step}",
                    status="completed",
                    step=step,
                    iteration=0,
                    metrics=[Metric(name="default", value=1.0, unit="unknown")],
                ),
            )
        else:
            history.append(step=step, action={"ngpus": step}, reward=reward, observation=[1.0])
    records = Trajectory.load(iteration).records()
    assert [record["reward"] for record in records] == list(rewards)
    assert [type(record["reward"]) for record in records] == [type(reward) for reward in rewards]
    assert not any(record.ineligibility_reasons("measurement") for record in records)
    getters = [(record.get("reward"), record.get("reward", decode_numbers=True)) for record in records]
    before = {path: path.read_bytes() for path in iteration.rglob("*") if path.is_file()}
    reporter = DSEReporter(
        slurm_system, TestScenario(name="reward-order", test_runs=[case]), slurm_system.output_path, ReportConfig()
    )
    reporter.report_best_dse_config()
    assert all(path.read_bytes() == data for path, data in before.items())
    reloaded = Trajectory.load(iteration).records()
    assert [(record.get("reward"), record.get("reward", decode_numbers=True)) for record in reloaded] == getters
    assert [type(record["reward"]) for record in reloaded] == [type(reward) for reward in rewards]
    actual = toml.loads((iteration / f"{case.name}.toml").read_text())
    assert actual["cmd_args"]["ngpus"] == expected_step


@pytest.mark.parametrize("archive", ["legacy", "native"])
def test_node_only_sweeps_export_selected_workload_and_preserve_resource_evidence(
    slurm_system: SlurmSystem, archive: str
) -> None:
    from cloudai.configurator import Trajectory
    from cloudai.core import RewardOverrides
    from cloudai.models.output import Metric, Run

    case = TestRun(
        name="node-only",
        test=NCCLTestDefinition(
            name="nccl",
            description="node-only sweep",
            test_template_name="NcclTest",
            cmd_args=NCCLCmdArgs(docker_image_url="fake://url/nccl", ngpus=1),
            agent_steps=2,
        ),
        num_nodes=[2, 4],
        nodes=["node1"],
    )
    assert case.is_dse_job and not case.test.is_dse_job
    assert case.param_space == {"NUM_NODES": [2, 4]}
    iteration = slurm_system.output_path / case.name / "0"
    history = Trajectory(iteration_dir=iteration)
    for step, nnodes in enumerate((2, 4), 1):
        selected = case.apply_params_set({"NUM_NODES": nnodes})
        selected.step = step
        selected.output_path = iteration / str(step)
        selected.output_path.mkdir(parents=True)
        (selected.output_path / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME).write_text(
            toml.dumps(TestRunDetails.from_test_run(selected, "", "").model_dump(mode="json"))
        )
        if archive == "native":
            history.record(
                test_run=selected,
                action_space=case.param_space,
                action={"NUM_NODES": nnodes},
                reward=step,
                observation={"default": 1.0},
                system=slurm_system,
                rewards=RewardOverrides(),
                run_output=Run(
                    path=str(selected.output_path),
                    jobid=f"nodes-{step}",
                    status="completed",
                    step=step,
                    iteration=0,
                    metrics=[Metric(name="default", value=1.0, unit="unknown")],
                ),
            )
        else:
            history.append(step=step, action={"NUM_NODES": nnodes}, reward=step, observation=[1.0])
    before = {path: path.read_bytes() for path in iteration.rglob("*") if path.is_file()}
    reporter = DSEReporter(
        slurm_system, TestScenario(name="node-only", test_runs=[case]), slurm_system.output_path, ReportConfig()
    )
    reporter.report_best_dse_config()
    exported = toml.loads((iteration / f"{case.name}.toml").read_text())
    dump = toml.loads((iteration / "2" / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME).read_text())
    assert exported == dump["test_definition"]
    assert NCCLTestDefinition.model_validate(exported).cmd_args.ngpus == 1
    assert all(path.read_bytes() == content for path, content in before.items())
    record = Trajectory.load(iteration, recovery="recorded").records()[1]
    assert record.get("action", decode_numbers=True) == {"NUM_NODES": 4}
    details = record.get("run_details", decode_numbers=True)
    assert details["launch"]["num_nodes"] == 4
    assert dump["nnodes"] == 4


@pytest.mark.parametrize("archive", ["legacy", "native"])
@pytest.mark.parametrize("node_only", [False, True], ids=["command", "nodes"])
def test_every_existing_dse_iteration_exports_and_corrupt_history_propagates(
    slurm_system: SlurmSystem, archive: str, node_only: bool
) -> None:
    from cloudai.configurator import Trajectory, TrajectoryLoadError
    from cloudai.core import RewardOverrides
    from cloudai.models.output import Metric, Run

    case = TestRun(
        name="all-iterations",
        test=NCCLTestDefinition(
            name="nccl",
            description="all DSE iterations",
            test_template_name="NcclTest",
            cmd_args=NCCLCmdArgs(docker_image_url="fake://url/nccl", ngpus=1 if node_only else [1, 2]),
            agent_steps=2,
        ),
        num_nodes=[2, 4] if node_only else 1,
        nodes=["node1"],
        iterations=3,
    )
    dimension = "NUM_NODES" if node_only else "ngpus"
    expected = {}
    inputs = {}
    for iteration in (0, 2):
        directory = slurm_system.output_path / case.name / str(iteration)
        history = Trajectory(iteration_dir=directory)
        for step in (1, 2):
            action = {dimension: step * 2 if node_only else step}
            selected = case.apply_params_set(action)
            selected.step = step
            selected.current_iteration = iteration
            selected.output_path = directory / str(step)
            selected.output_path.mkdir(parents=True)
            dump = TestRunDetails.from_test_run(selected, "", "").model_dump(mode="json")
            (selected.output_path / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME).write_text(toml.dumps(dump))
            if archive == "native":
                history.record(
                    test_run=selected,
                    action_space=case.param_space,
                    action=action,
                    reward=step,
                    observation={"default": 1.0},
                    system=slurm_system,
                    rewards=RewardOverrides(),
                    run_output=Run(
                        path=str(selected.output_path),
                        jobid=f"iteration-{iteration}-{step}",
                        status="completed",
                        step=step,
                        iteration=iteration,
                        metrics=[Metric(name="default", value=1.0, unit="unknown")],
                    ),
                )
            else:
                history.append(step=step, action=action, reward=step, observation=[1.0])
        expected[directory / f"{case.name}.toml"] = toml.loads(
            (directory / "2" / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME).read_text()
        )["test_definition"]
        inputs.update({path: path.read_bytes() for path in directory.rglob("*") if path.is_file()})
    reporter = DSEReporter(
        slurm_system, TestScenario(name=case.name, test_runs=[case]), slurm_system.output_path, ReportConfig()
    )

    reporter.report_best_dse_config()

    assert {path: toml.loads(path.read_text()) for path in expected} == expected
    assert not (slurm_system.output_path / case.name / "1").exists()
    assert all(path.read_bytes() == content for path, content in inputs.items())
    corrupt = slurm_system.output_path / case.name / "1"
    corrupt.mkdir()
    (corrupt / Trajectory.file_name).write_text("step,action,reward,observation\n1,{},1,[],extra\n")
    with pytest.raises(TrajectoryLoadError, match="row 2, expected 4 fields, got 5"):
        reporter.report_best_dse_config()


@pytest.fixture
def recorded_dse_case(slurm_system: SlurmSystem):
    from cloudai.configurator import Trajectory
    from cloudai.core import RewardOverrides
    from cloudai.models.output import Metric, Run

    def create(archive="native", timing=True, rewards=(2, 1)):
        case = TestRun(
            name="coherent-recording",
            test=NCCLTestDefinition(
                name="nccl",
                description="coherent recording",
                test_template_name="NcclTest",
                cmd_args=NCCLCmdArgs(docker_image_url="fake://nccl", ngpus=[1, 2]),
                agent_steps=3,
            ),
            num_nodes=1,
            nodes=["node1"],
            iterations=2,
        )
        iteration = slurm_system.output_path / case.name / "0"
        history = Trajectory(iteration_dir=iteration)

        def commit(step: int, ngpus: int, reward: int):
            selected = case.apply_params_set({"ngpus": ngpus})
            selected.step = step
            selected.output_path = iteration / str(step)
            selected.output_path.mkdir(parents=True)
            (selected.output_path / "stdout.txt").write_text("# Out of bounds values# Avg bus bandwidth")
            (selected.output_path / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME).write_text(
                toml.dumps(TestRunDetails.from_test_run(selected, "", "").model_dump(mode="json"))
            )
            if timing:
                _write_slurm_job(selected.output_path, 60)
            if archive == "native":
                history.record(
                    test_run=selected,
                    action_space=case.param_space,
                    action={"ngpus": ngpus},
                    reward=reward,
                    observation={"default": 1.0},
                    system=slurm_system,
                    rewards=RewardOverrides(),
                    run_output=Run(
                        path=str(selected.output_path),
                        jobid=f"coherent-{step}",
                        status="completed",
                        step=step,
                        iteration=0,
                        metrics=[Metric(name="default", value=1.0, unit="unknown")],
                    ),
                )
            else:
                history.append(step=step, action={"ngpus": ngpus}, reward=reward, observation=[1.0])
            return selected

        runs = [commit(step, step, reward) for step, reward in enumerate(rewards, 1)]
        return case, history, runs, commit

    return create


@pytest.mark.parametrize("archive", ["native", "legacy"])
def test_generate_exports_valid_history_without_timing_metadata(
    slurm_system: SlurmSystem, recorded_dse_case, archive: str
) -> None:
    case, history, runs, _ = recorded_dse_case(archive=archive, timing=False)
    before = {path: path.read_bytes() for path in history.output_path.parent.rglob("*") if path.is_file()}
    assert build_dse_summaries(slurm_system, slurm_system.output_path, runs, [case]) == []
    DSEReporter(
        slurm_system, TestScenario(name="without-timing", test_runs=[case]), slurm_system.output_path, ReportConfig()
    ).generate()
    export = history.output_path.parent / f"{case.name}.toml"
    assert export.is_file()
    assert toml.loads(export.read_text())["cmd_args"]["ngpus"] == 1
    assert not (slurm_system.output_path / "without-timing-dse-report.html").exists()
    assert all(path.read_bytes() == data for path, data in before.items())


def test_generate_uses_one_readonly_snapshot_per_configured_iteration(
    slurm_system: SlurmSystem, recorded_dse_case, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cloudai.reporter as reporter_module
    from cloudai.configurator import Trajectory

    case, history, _, commit = recorded_dse_case()
    actual_load = Trajectory.load
    actual_summaries = build_dse_summaries
    snapshots = []
    summaries = []

    def interleaved_load(path, **kwargs):
        loaded = actual_load(path, **kwargs)
        snapshots.append((Path(path), [record.step for record in loaded.records()]))
        with pytest.raises(RuntimeError):
            loaded.append(step=99, action={"ngpus": 1}, reward=99, observation=[1.0])
        if len(snapshots) == 1:
            commit(3, 2, 9)
        return loaded

    def capture_summaries(*args, **kwargs):
        result = actual_summaries(*args, **kwargs)
        summaries.extend(result)
        return result

    monkeypatch.setattr(Trajectory, "load", staticmethod(interleaved_load))
    monkeypatch.setattr(reporter_module, "build_dse_summaries", capture_summaries)
    reporter = DSEReporter(
        slurm_system, TestScenario(name="coherent", test_runs=[case]), slurm_system.output_path, ReportConfig()
    )
    reporter.generate()
    export = history.output_path.parent / f"{case.name}.toml"
    assert len(summaries) == 1
    assert toml.loads(summaries[0].best_config_toml)["cmd_args"]["ngpus"] == 1
    assert toml.loads(export.read_text())["cmd_args"]["ngpus"] == 1
    assert snapshots == [
        (slurm_system.output_path / case.name / "0" / Trajectory.file_name, [1, 2]),
        (slurm_system.output_path / case.name / "1" / Trajectory.file_name, []),
    ]
    assert (slurm_system.output_path / "coherent-dse-report.html").is_file()
    assert [record.step for record in actual_load(history.output_path.parent).records()] == [1, 2, 3]
    reporter.report_best_dse_config()
    assert toml.loads(export.read_text())["cmd_args"]["ngpus"] == 2


def test_summary_parameter_projection_work_is_independent_of_record_count(
    slurm_system: SlurmSystem, recorded_dse_case, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cloudai.reporter as reporter_module

    case, _, _runs, _ = recorded_dse_case()
    actual_param_space = TestRun.param_space.fget
    assert actual_param_space is not None
    reads = []

    def counted_param_space(test_run: TestRun):
        if test_run is case:
            reads.append(test_run.name)
        return actual_param_space(test_run)

    monkeypatch.setattr(TestRun, "param_space", property(counted_param_space))
    summaries = []

    def capture_summaries(*args, **kwargs):
        result = build_dse_summaries(*args, **kwargs)
        summaries.extend(result)
        return result

    monkeypatch.setattr(reporter_module, "build_dse_summaries", capture_summaries)
    DSEReporter(
        slurm_system, TestScenario(name="projection", test_runs=[case]), slurm_system.output_path, ReportConfig()
    ).generate()
    assert len(summaries) == 1
    assert [row.name for row in summaries[0].parameter_rows] == ["ngpus"]
    assert len(reads) <= 3


def test_summary_retains_exact_selection_for_adjacent_large_integers(
    slurm_system: SlurmSystem, recorded_dse_case, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cloudai.reporter as reporter_module

    case, _, _runs, _ = recorded_dse_case(rewards=(2**53, 2**53 + 1))
    summaries = []

    def capture_summaries(*args, **kwargs):
        result = build_dse_summaries(*args, **kwargs)
        summaries.extend(result)
        return result

    monkeypatch.setattr(reporter_module, "build_dse_summaries", capture_summaries)
    DSEReporter(
        slurm_system, TestScenario(name="projection", test_runs=[case]), slurm_system.output_path, ReportConfig()
    ).generate()
    assert len(summaries) == 1
    chart = summaries[0].reward_chart_data
    assert chart is not None
    assert chart["rewards"] == [2**53, 2**53 + 1]
    assert chart["best_index"] == 1
    assert toml.loads(summaries[0].best_config_toml)["cmd_args"]["ngpus"] == 2


@pytest.mark.parametrize("condition", ["opaque_label", "malformed_status"])
def test_reports_use_recovered_measurement_eligibility(
    slurm_system: SlurmSystem, recorded_dse_case, tmp_path: Path, condition: str
) -> None:
    import json

    from cloudai.configurator import Trajectory
    from cloudai.systems.slurm.slurm_metadata import SlurmJobMetadata

    case, original_history, original_runs, _ = recorded_dse_case(archive="legacy", rewards=(2, 3))
    case.iterations = 1
    case.test.cmd_args = NCCLCmdArgs.model_validate(case.test.cmd_args.model_dump() | {"history_label": "nan"})
    results_root = tmp_path / "recovered-reports"
    iteration = results_root / case.name / "0"
    legacy_root = tmp_path / "historical-input"
    legacy = Trajectory(iteration_dir=legacy_root)
    for record, original_run in zip(original_history.records(), original_runs, strict=True):
        selected = case.apply_params_set(record["action"])
        selected.step = record.step
        selected.output_path = iteration / str(record.step)
        selected.output_path.mkdir(parents=True)
        legacy_step = legacy_root / str(record.step)
        legacy_step.mkdir(parents=True)
        details = TestRunDetails.from_test_run(selected, "", "").model_dump(mode="json")
        for folder in (legacy_step, selected.output_path):
            (folder / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME).write_text(toml.dumps(details))
        timing = toml.loads((original_run.output_path / "slurm-job.toml").read_text())
        timing["job_root"] = str(selected.output_path)
        (selected.output_path / "slurm-job.toml").write_text(
            toml.dumps(SlurmJobMetadata.model_validate(timing).model_dump(mode="json"))
        )
        legacy.append(
            step=record.step,
            action=record["action"],
            reward=record["reward"],
            observation=record["observation"],
            env_params={"history_label": "nan"},
        )
    before_legacy = {path: path.read_bytes() for path in legacy_root.rglob("*") if path.is_file()}
    recovered = Trajectory.load(legacy_root, recovery="recorded")
    archived = Trajectory(iteration_dir=iteration)
    for record in recovered.records():
        metadata = {name: record.get(name) for name in record.metadata}
        metadata["env_params"] = record.get("env_params", decode_numbers=True)
        if condition == "malformed_status" and record.step == 2:
            measurement = record.get("measurement", decode_numbers=True)
            measurement["execution_status"] = []
            metadata["measurement"] = json.dumps(measurement)
        archived.append(
            step=record.step,
            action=record["action"],
            reward=record["reward"],
            observation=record["observation"],
            **metadata,
        )
    snapshot = Trajectory.load(iteration)
    before = {path: path.read_bytes() for path in iteration.rglob("*") if path.is_file()}

    summaries = build_dse_summaries(slurm_system, results_root, [], [case], histories={iteration: snapshot})
    DSEReporter(
        slurm_system, TestScenario(name="recovered", test_runs=[case]), results_root, ReportConfig()
    ).report_best_dse_config(histories={iteration: snapshot})

    expected = 1 if condition == "malformed_status" else 2
    assert len(summaries) == 1
    assert toml.loads(summaries[0].best_config_toml)["cmd_args"]["ngpus"] == expected
    assert toml.loads((iteration / f"{case.name}.toml").read_text())["cmd_args"]["ngpus"] == expected
    chart = summaries[0].reward_chart_data
    assert chart is not None
    assert chart["labels"] == [1, 2]
    assert chart["best_index"] == expected - 1
    assert {path: path.read_bytes() for path in before} == before
    assert {path: path.read_bytes() for path in legacy_root.rglob("*") if path.is_file()} == before_legacy
