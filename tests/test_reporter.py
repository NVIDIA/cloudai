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
            "best_index": 0,
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
        ("native", "matching"),
        ("native", "minimal"),
        ("native", "command"),
        ("native", "held_action"),
        ("native", "string_action"),
        ("native", "float_action"),
        ("native", "bool_action"),
        ("native", "static"),
        ("native", "resource"),
        ("native", "bool_resource"),
        ("native", "nodes"),
        ("native", "name"),
        ("native", "iteration"),
        ("native", "step"),
        ("native", "bool_step"),
        ("native", "output_path"),
        ("native", "missing_action"),
        ("legacy", "matching"),
        ("legacy", "minimal"),
        ("legacy", "command"),
        ("native", "credentials"),
        ("legacy", "credentials"),
        ("native", "bool_static"),
        ("native", "none_static"),
        ("native", "float_resource"),
        ("native", "string_resource"),
        ("native", "float_step"),
        ("native", "string_step"),
        ("native", "date"),
        ("native", "none_array"),
        ("native", "list_order"),
    ],
)
def test_best_config_export_verifies_recorded_settings_without_inventing_defaults(
    slurm_system: SlurmSystem, archive: str, change: str
) -> None:
    from copy import deepcopy
    from datetime import date

    from cloudai.configurator import Trajectory, TrajectoryLoadError
    from cloudai.core import RewardOverrides, TestScenario
    from cloudai.models.output import Metric, Run
    from cloudai.models.scenario import ReportConfig

    case = TestRun(
        name="typed-export",
        test=NCCLTestDefinition(
            name="nccl",
            description="typed export",
            test_template_name="NcclTest",
            cmd_args=NCCLCmdArgs.model_validate(
                {
                    "docker_image_url": "fake://url/nccl",
                    **(
                        {"fixed_array": {"none_array": [None], "list_order": [1, 2]}[change]}
                        if change in ("none_array", "list_order")
                        else {}
                    ),
                    **({"date_alias": date(2026, 1, 2)} if change == "date" else {}),
                    "ngpus": [1, 2],
                    "iters": [10, 20],
                    "payload": {"label": "0001", "enabled": True, "missing": None, "a.b": ["001", "true"]},
                }
            ),
            dse_excluded_args=["cmd_args.payload"]
            + (["cmd_args.fixed_array"] if change in ("none_array", "list_order") else []),
            agent_steps=1,
            extra_env_vars={"FIXED": "0001"},
        ),
        num_nodes=1,
        nodes=["node1"],
        iterations=1,
    )
    if change == "credentials":
        from cloudai.core import GitRepo
        from cloudai.workloads.megatron_bridge.megatron_bridge import (
            MegatronBridgeCmdArgs,
            MegatronBridgeTestDefinition,
        )

        case.test = MegatronBridgeTestDefinition(
            name="bridge",
            description="credential export",
            test_template_name="MegatronBridge",
            cmd_args=MegatronBridgeCmdArgs.model_validate(
                {
                    "ngpus": [1, 2],
                    "iters": [10, 20],
                    "hf_token": "synthetic-export-hf",
                    "wandb_key": "synthetic-export-wandb",
                    "alias": "synthetic-export-hf:synthetic-export-wandb",
                    "model_family_name": "qwen3",
                    "model_recipe_name": "30b_a3b",
                }
            ),
            git_repos=[GitRepo(url="https://github.com/NVIDIA-NeMo/Megatron-Bridge.git", commit="r0.2.0")],
            agent_steps=1,
        )
    iteration = slurm_system.output_path / case.name / "0"
    selected = case.apply_params_set({"ngpus": 2, "iters": 10})
    assert "fixed_array" not in case.param_space
    assert selected.test.cmd_args.model_dump().get("fixed_array") == {"none_array": [None], "list_order": [1, 2]}.get(
        change
    )
    selected.step = 1
    selected.output_path = iteration / "1"
    selected.output_path.mkdir(parents=True)
    (selected.output_path / "stdout.txt").write_text("# Out of bounds values# Avg bus bandwidth")
    (selected.output_path / "cloudai_megatron_bridge_launcher.log").write_text(
        "Step Time : 1.0 s GPU utilization: 2.0 TFLOP/s/GPU"
    )
    (selected.output_path / "cloudai_megatron_bridge_evaluation.exitcode").write_text("0")
    job = SlurmJobMetadata(
        job_id=1,
        name=case.name,
        state="COMPLETED",
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        elapsed_time_sec=60,
        exit_code="0:0",
        srun_cmd="srun test",
        test_cmd="test",
        job_root=selected.output_path,
        job_steps=[],
    )
    (selected.output_path / "slurm-job.toml").write_text(toml.dumps(job.model_dump(mode="json")))
    history = Trajectory(iteration_dir=iteration)
    if archive == "native":
        history.record(
            test_run=selected,
            action_space=case.param_space,
            action={"ngpus": 2},
            reward=1.0,
            observation={"default": 1.0},
            system=slurm_system,
            rewards=RewardOverrides(),
            run_output=Run(
                path=str(selected.output_path),
                jobid="export",
                status="completed",
                step=1,
                iteration=0,
                metrics=[Metric(name="default", value=1.0, unit="unknown")],
            ),
        )
    else:
        history.append(step=1, action={"ngpus": 2, "iters": 10}, reward=1.0, observation=1.0)
    raw = toml.loads(toml.dumps(TestRunDetails.from_test_run(selected, "", "").model_dump()))
    if change == "minimal":
        raw["test_definition"] = {
            key: raw["test_definition"][key] for key in ("name", "description", "test_template_name", "cmd_args")
        }
        raw["test_definition"]["cmd_args"] = {
            key: raw["test_definition"]["cmd_args"][key] for key in ("docker_image_url", "ngpus", "iters", "payload")
        }
    updates = {
        "none_array": (("test_definition", "cmd_args", "fixed_array"), []),
        "list_order": (("test_definition", "cmd_args", "fixed_array"), [2, 1]),
        "command": (("test_definition", "cmd_args", "ngpus"), 1),
        "held_action": (("test_definition", "cmd_args", "iters"), 999),
        "string_action": (("test_definition", "cmd_args", "iters"), "10"),
        "float_action": (("test_definition", "cmd_args", "iters"), 10.0),
        "bool_action": (("test_definition", "cmd_args", "iters"), True),
        "static": (("test_definition", "extra_env_vars", "FIXED"), "changed"),
        "resource": (("nnodes",), 2),
        "bool_resource": (("nnodes",), True),
        "nodes": (("nodes",), ["other-node"]),
        "name": (("name",), "other-run"),
        "iteration": (("current_iteration",), 1),
        "step": (("step",), 2),
        "bool_step": (("step",), True),
        "output_path": (("output_path",), str(iteration / "other" / "1")),
        "bool_static": (("test_definition", "cmd_args", "payload", "enabled"), 1),
        "none_static": (("test_definition", "cmd_args", "payload", "missing"), "not-none"),
        "float_resource": (("nnodes",), 1.0),
        "string_resource": (("nnodes",), "1"),
        "float_step": (("step",), 1.0),
        "string_step": (("step",), "1"),
    }
    if change in updates:
        path, value = updates[change]
        target = raw
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    elif change == "missing_action":
        del raw["test_definition"]["cmd_args"]["iters"]
    expected = deepcopy(raw["test_definition"])
    dump = selected.output_path / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME
    dump.write_text(toml.dumps(raw))
    export = iteration / f"{case.name}.toml"
    previous = b'preserve = "existing export"\n'
    export.write_bytes(previous)
    before = {
        path: path.read_bytes() for path in (history.output_path, history.output_path.with_name("metadata.csv"), dump)
    }
    if change in ("matching", "minimal", "credentials", "date"):
        reporter = DSEReporter(
            slurm_system, TestScenario(name="export", test_runs=[case]), slurm_system.output_path, ReportConfig()
        )
        reporter.report_best_dse_config()
        actual = toml.loads(export.read_text())
        assert actual == expected
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
            reporter.report_best_dse_config()
        assert export.read_bytes() == previous
    assert all(path.read_bytes() == original for path, original in before.items())
