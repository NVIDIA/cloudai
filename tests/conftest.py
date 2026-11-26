# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import csv
from pathlib import Path
from typing import Callable

import pytest
import toml
import yaml

from cloudai.core import CommandGenStrategy, TestRun
from cloudai.models.scenario import TestRunDetails
from cloudai.models.workload import CmdArgs, TestDefinition
from cloudai.systems.kubernetes import KubernetesSystem
from cloudai.systems.runai import RunAISystem
from cloudai.systems.slurm import SlurmGroup, SlurmPartition, SlurmSystem
from cloudai.systems.standalone import StandaloneSystem
from cloudai.workloads.nccl_test.nccl import NCCLCmdArgs, NCCLTestDefinition


@pytest.fixture(scope="session", autouse=True)
def cleanup():
    yield

    for f in {"env_vars.sh", "hostfile.txt", "start_server_wrapper.sh"}:
        (Path.cwd() / f).unlink(missing_ok=True)


@pytest.fixture
def write_parseable_test() -> Callable[[Path, str], None]:
    def write(path: Path, name: str) -> None:
        path.write_text(
            f'name = "{name}"\ndescription = "{name}"\ntest_template_name = "Sleep"\n\n[cmd_args]\nseconds = 1\n'
        )

    return write


@pytest.fixture
def write_unregistered_agent_test() -> Callable[[Path, str], None]:
    def write(path: Path, name: str) -> None:
        path.write_text(
            f'name = "{name}"\n'
            f'description = "{name}"\n'
            'test_template_name = "Sleep"\n'
            'agent = "agent_from_an_extra_nobody_installed"\n'
            "\n"
            "[cmd_args]\n"
            "seconds = 1\n"
        )

    return write


@pytest.fixture
def write_scenario() -> Callable[[Path, str], None]:
    def write(path: Path, test_name: str) -> None:
        path.write_text(f'name = "{path.stem}"\n\n[[Tests]]\nid = "Tests.1"\ntest_name = "{test_name}"\n')

    return write


@pytest.fixture
def slurm_system(tmp_path: Path) -> SlurmSystem:
    system = SlurmSystem(
        name="test_system",
        install_path=tmp_path / "install",
        output_path=tmp_path / "output",
        cache_docker_images_locally=True,
        default_partition="main",
        gpus_per_node=8,
        status_retry_pause_seconds=0,
        partitions=[
            SlurmPartition(
                name="main",
                groups=[
                    SlurmGroup(name="group1", nodes=["node-[033-048]"]),
                    SlurmGroup(name="group2", nodes=["node-[049-064]"]),
                ],
            ),
            SlurmPartition(
                name="backup",
                groups=[
                    SlurmGroup(name="group1", nodes=["node0[1-4]"]),
                    SlurmGroup(name="group2", nodes=["node0[5-8]"]),
                ],
            ),
        ],
    )
    system.scheduler = "slurm"
    system.monitor_interval = 0
    system.supports_gpu_directives_cache = True
    return system


@pytest.fixture
def k8s_system(tmp_path: Path) -> KubernetesSystem:
    kube_config = tmp_path / "kubeconfig"
    config = {
        "apiVersion": "v1",
        "kind": "Config",
        "current-context": "test-context",
        "clusters": [{"name": "test-cluster", "cluster": {"server": "https://test-server:6443"}}],
        "contexts": [{"name": "test-context", "context": {"cluster": "test-cluster", "user": "test-user"}}],
        "users": [{"name": "test-user", "user": {"token": "test-token"}}],
    }
    kube_config.write_text(yaml.dump(config))

    system = KubernetesSystem(
        name="test_kubernetes_system",
        install_path=tmp_path / "install",
        output_path=tmp_path / "output",
        scheduler="kubernetes",
        global_env_vars={},
        monitor_interval=1,
        gpus_per_node=8,
        use_host_network=True,
        default_namespace="test-namespace",
        kube_config_path=kube_config,
    )
    return system


@pytest.fixture
def runai_system(tmp_path: Path) -> RunAISystem:
    system = RunAISystem(
        name="test_runai_system",
        install_path=tmp_path / "install",
        output_path=tmp_path / "output",
        base_url="http://runai.example.com",
        app_id="test_app_id",
        app_secret="test_app_secret",
        project_id="test_project_id",
        cluster_id="test_cluster_id",
        scheduler="runai",
        global_env_vars={},
        monitor_interval=60,
        user_email="test_user@example.com",
    )
    return system


@pytest.fixture
def standalone_system(tmp_path: Path) -> StandaloneSystem:
    return StandaloneSystem(
        name="standalone",
        scheduler="standalone",
        install_path=tmp_path / "install",
        output_path=tmp_path / "output",
    )


@pytest.fixture
def base_tr(tmp_path: Path) -> TestRun:
    return TestRun(
        name="tr-name",
        test=TestDefinition(name="n", description="d", test_template_name="tt", cmd_args=CmdArgs()),
        num_nodes=1,
        nodes=[],
        output_path=tmp_path / "output" / "tr-name",
    )


def create_test_directories(slurm_system: SlurmSystem, test_run: TestRun) -> None:
    test_dir = slurm_system.output_path / test_run.name
    for iteration in range(test_run.iterations):
        folder = test_dir / str(iteration)
        folder.mkdir(exist_ok=True, parents=True)
        if test_run.is_dse_job:
            with open(folder / "trajectory.csv", "w") as _f_csv:
                csw_writer = csv.writer(_f_csv)
                csw_writer.writerow(["step", "action", "reward", "observation"])

                for step in range(test_run.test.agent_steps):
                    step_folder = folder / str(step)
                    step_folder.mkdir(exist_ok=True, parents=True)
                    trd = TestRunDetails.from_test_run(test_run, "", "")
                    csw_writer.writerow([step, {}, step * 2.1, [step]])
                    with open(step_folder / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME, "w") as _f_trd:
                        toml.dump(trd.model_dump(), _f_trd)


@pytest.fixture
def benchmark_tr(slurm_system: SlurmSystem) -> TestRun:
    test_definition = NCCLTestDefinition(
        name="nccl",
        description="NCCL test",
        test_template_name="NcclTest",
        cmd_args=NCCLCmdArgs(docker_image_url="fake://url/nccl"),
    )
    tr = TestRun(name="benchmark", test=test_definition, num_nodes=1, nodes=["node1"], iterations=3)
    create_test_directories(slurm_system, tr)
    return tr


@pytest.fixture
def dse_tr(slurm_system: SlurmSystem) -> TestRun:
    test_definition = NCCLTestDefinition(
        name="nccl",
        description="NCCL test",
        test_template_name="NcclTest",
        cmd_args=NCCLCmdArgs(docker_image_url="fake://url/nccl"),
        extra_env_vars={"VAR1": ["value1", "value2"]},
        agent_steps=12,
    )

    tr = TestRun(name="dse", test=test_definition, num_nodes=1, nodes=["node1"], iterations=12)
    create_test_directories(slurm_system, tr)
    return tr


@pytest.fixture
def nccl_tr(tmp_path: Path) -> TestRun:
    test = NCCLTestDefinition(
        name="nccl",
        description="desc",
        test_template_name="t",
        cmd_args=NCCLCmdArgs(docker_image_url="fake://url/nccl"),
    )
    tr = TestRun(name="nccl_test", test=test, num_nodes=2, nodes=[], output_path=tmp_path)

    stdout_content = """# Rank  0 Group  0 Pid 1000 on node1 device  0 [0xaa] NVIDIA H100
# Rank  1 Group  0 Pid 1001 on node1 device  1 [0xbb] NVIDIA H100
# Rank  2 Group  0 Pid 1002 on node1 device  2 [0xcc] NVIDIA H100
# Rank  3 Group  0 Pid 1003 on node1 device  3 [0xdd] NVIDIA H100
# Rank  4 Group  0 Pid 1004 on node1 device  4 [0xee] NVIDIA H100
# Rank  5 Group  0 Pid 1005 on node1 device  5 [0xff] NVIDIA H100
# Rank  6 Group  0 Pid 1006 on node1 device  6 [0x11] NVIDIA H100
# Rank  7 Group  0 Pid 1007 on node1 device  7 [0x22] NVIDIA H100
# Rank  8 Group  0 Pid 2000 on node2 device  0 [0xaa] NVIDIA H100
# Rank  9 Group  0 Pid 2001 on node2 device  1 [0xbb] NVIDIA H100
# Rank 10 Group  0 Pid 2002 on node2 device  2 [0xcc] NVIDIA H100
# Rank 11 Group  0 Pid 2003 on node2 device  3 [0xdd] NVIDIA H100
# Rank 12 Group  0 Pid 2004 on node2 device  4 [0xee] NVIDIA H100
# Rank 13 Group  0 Pid 2005 on node2 device  5 [0xff] NVIDIA H100
# Rank 14 Group  0 Pid 2006 on node2 device  6 [0x11] NVIDIA H100
# Rank 15 Group  0 Pid 2007 on node2 device  7 [0x22] NVIDIA H100
#
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
     1000000       1000000     float     sum      -1    1.11   10.10   20.20      0    1.12   10.11   20.21      0
     2000000       2000000     float     sum      -1    2.22   20.20   30.30      0    2.23   20.21   30.31      0
     12000000      12000000     float     sum      -1   13.13  120.30  130.40      0   13.14  120.31  130.41      0
# Avg bus bandwidth    : 111.111
"""
    (tr.output_path / "stdout.txt").write_text(stdout_content)

    return tr


@pytest.fixture
def selected_dump_read_failure(monkeypatch: pytest.MonkeyPatch):
    from pathlib import Path

    unreadable: set[Path] = set()
    attempts: list[Path] = []
    actual_open = Path.open

    def open_path(path: Path, mode="r", **kwargs):
        if path in unreadable and mode == "r":
            attempts.append(path)
            raise OSError("selected dump read failed")
        return actual_open(path, mode, **kwargs)

    monkeypatch.setattr(Path, "open", open_path)
    return unreadable, attempts


@pytest.fixture
def selected_dump_case(slurm_system: SlurmSystem, selected_dump_read_failure):
    import os
    from contextlib import nullcontext
    from copy import deepcopy
    from datetime import date

    from pydantic import ValidationError

    from cloudai.configurator import Trajectory
    from cloudai.core import RewardOverrides
    from cloudai.models.output import Metric, Run
    from cloudai.systems.slurm.slurm_metadata import SlurmJobMetadata

    def create(archive: str, change: str):
        change = change.removesuffix("-generate")
        nested_action = {"x": None, "keep": 1} if archive == "native" and change == "nested_matching" else {"x": 1}
        case = TestRun(
            name="typed-export",
            test=NCCLTestDefinition(
                name="nccl",
                description="typed export",
                test_template_name="NcclTest",
                cmd_args=NCCLCmdArgs.model_validate(
                    {
                        "docker_image_url": "fake://url/nccl",
                        "nthreads": {"omitted_different_default": 4}.get(change, 1),
                        **(
                            {"fixed_array": {"none_array": [None], "list_order": [1, 2]}[change]}
                            if change in ("none_array", "list_order")
                            else {}
                        ),
                        **({"date_alias": date(2026, 1, 2)} if change == "date" else {}),
                        "ngpus": [1, 2],
                        "iters": [10, 20],
                        "payload": (
                            [nested_action, {"x": 2}]
                            if change.startswith("nested_")
                            else {"label": "0001", "enabled": True, "missing": None, "a.b": ["001", "true"]}
                        ),
                    }
                ),
                dse_excluded_args=([] if change.startswith("nested_") else ["cmd_args.payload"])
                + (["cmd_args.fixed_array"] if change in ("none_array", "list_order") else []),
                agent_steps=1,
                extra_env_vars={"FIXED": "0001"},
            ),
            num_nodes=1,
            nodes=[] if change == "omitted_nodes_empty" else ["node1"],
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
        selected = case.apply_params_set(
            {"ngpus": 2, "iters": 10} | ({"payload": nested_action} if change.startswith("nested_") else {})
        )
        assert "fixed_array" not in case.param_space
        assert selected.test.cmd_args.model_dump().get("fixed_array") == {
            "none_array": [None],
            "list_order": [1, 2],
        }.get(change)
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
                action={"ngpus": 2} | ({"payload": nested_action} if change.startswith("nested_") else {}),
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
            history.append(
                step=1,
                action={"ngpus": 2, "iters": 10} | ({"payload": nested_action} if change.startswith("nested_") else {}),
                reward=1.0,
                observation=1.0,
            )
        raw = toml.loads(toml.dumps(TestRunDetails.from_test_run(selected, "", "").model_dump()))
        if change == "minimal":
            raw["test_definition"] = {
                key: raw["test_definition"][key] for key in ("name", "description", "test_template_name", "cmd_args")
            }
            raw["test_definition"]["cmd_args"] = {
                key: raw["test_definition"]["cmd_args"][key]
                for key in ("docker_image_url", "ngpus", "iters", "payload")
            }
        updates = {
            "unknown_template": (("test_definition", "test_template_name"), "UnregisteredTrajectoryWorkload"),
            "missing_template": (("test_definition", "test_template_name"), None),
            "invalid_argument": (("test_definition", "cmd_args", "nthreads"), "not-an-integer"),
            "missing_definition": (("test_definition",), None),
            "relative_equivalent": (("output_path",), os.path.relpath(selected.output_path, Path.cwd())),
            "relative_unrelated": (("output_path",), "unrelated-run/0/99"),
            "omitted_nodes_nonempty": (("nodes",), None),
            "omitted_nodes_empty": (("nodes",), None),
            "nested_extra": (("test_definition", "cmd_args", "payload", "y"), 2),
            "nested_missing": (("test_definition", "cmd_args", "payload"), {}),
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
        elif change in ("missing_action", "omitted_different_default"):
            del raw["test_definition"]["cmd_args"][
                {"missing_action": "iters", "omitted_different_default": "nthreads"}[change]
            ]
        ordinary_error = {
            "unknown_template": KeyError,
            "missing_template": KeyError,
            "invalid_argument": ValidationError,
            "missing_definition": ValidationError,
        }.get(change)
        with pytest.raises(ordinary_error) if ordinary_error is not None else nullcontext():
            _ = ordinary_error is None or TestRunDetails.model_validate(toml.loads(toml.dumps(raw)))
        expected = deepcopy(raw["test_definition"])
        dump = selected.output_path / CommandGenStrategy.TEST_RUN_DUMP_FILE_NAME
        dump.write_text('name = "unterminated' if change == "malformed_toml" else toml.dumps(raw))
        export = iteration / f"{case.name}.toml"
        previous = b'preserve = "existing export"\n'
        export.write_bytes(previous)
        before = {
            path: path.read_bytes()
            for path in (history.output_path, history.output_path.with_name("metadata.csv"), dump)
        }
        unreadable, read_attempts = selected_dump_read_failure
        unreadable.update([dump] if change == "read_error" else [])
        valid = change in (
            "matching",
            "minimal",
            "credentials",
            "date",
            "nested_matching",
            "relative_equivalent",
            "omitted_nodes_empty",
        )
        return selected, case, iteration, expected, dump, export, previous, before, read_attempts, change, valid

    return create


def selected_dump_cases() -> list[tuple[str, str]]:
    return [
        ("native", "matching"),
        ("native", "minimal"),
        ("native", "omitted_different_default"),
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
        ("native", "nested_matching"),
        ("native", "nested_extra"),
        ("native", "nested_missing"),
        ("legacy", "nested_matching"),
        ("legacy", "nested_extra"),
        ("legacy", "nested_missing"),
        ("native", "unknown_template"),
        ("native", "missing_template"),
        ("native", "invalid_argument"),
        ("native", "missing_definition"),
        ("native", "relative_equivalent"),
        ("native", "relative_unrelated"),
        ("native", "omitted_nodes_nonempty"),
        ("native", "omitted_nodes_empty"),
        ("native", "malformed_toml"),
        ("native", "read_error"),
    ]
