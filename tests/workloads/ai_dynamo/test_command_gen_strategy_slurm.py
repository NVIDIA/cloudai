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
from pathlib import Path
from typing import cast

import pytest
import yaml

from cloudai._core.test_scenario import TestRun
from cloudai.core import GitRepo
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.ai_dynamo import (
    LMCACHE_CONFIG_BACKUP_FILE_NAME,
    LMCACHE_CONFIG_FILE_NAME,
    AIDynamoArgs,
    AIDynamoCmdArgs,
    AIDynamoSlurmCommandGenStrategy,
    AIDynamoTestDefinition,
    AIPerf,
    AIPerfAccuracy,
    AIPerfPhase,
    GenAIPerf,
    LMCacheController,
    WorkerBaseArgs,
    WorkerConfig,
)


@pytest.fixture
def cmd_args() -> AIDynamoCmdArgs:
    return AIDynamoCmdArgs(
        docker_image_url="url",
        dynamo=AIDynamoArgs(
            model="model",
            workspace_path="/workspace",
            prefill_worker=WorkerConfig(
                cmd="python3 -m dynamo.vllm --is-prefill-worker",
                worker_initialized_regex="VllmWorker.*has.been.initialized",
                **{
                    "num-nodes": 1,
                    "args": WorkerBaseArgs(
                        **{
                            "gpu-memory-utilization": 0.95,
                            "tensor-parallel-size": 8,
                        }
                    ),
                },
            ),
            decode_worker=WorkerConfig(
                cmd="python3 -m dynamo.vllm",
                worker_initialized_regex="VllmWorker.*has.been.initialized",
                **{
                    "num-nodes": 1,
                    "args": WorkerBaseArgs(
                        **{
                            "gpu-memory-utilization": 0.95,
                            "tensor-parallel-size": 8,
                        }
                    ),
                },
            ),
        ),
        genai_perf=GenAIPerf(
            **{
                "endpoint-type": "chat",
                "streaming": True,
                "warmup-request-count": 10,
                "random-seed": 42,
                "synthetic-input-tokens-mean": 128,
                "synthetic-input-tokens-stddev": 32,
                "output-tokens-mean": 256,
                "output-tokens-stddev": 64,
                "extra-inputs": None,
                "concurrency": 2,
                "request-count": 10,
            }
        ),
    )


@pytest.fixture
def test_run(tmp_path: Path, cmd_args: AIDynamoCmdArgs) -> TestRun:
    dynamo_repo_path = tmp_path / "dynamo_repo"
    dynamo_repo_path.mkdir()

    tdef = AIDynamoTestDefinition(
        name="test",
        description="desc",
        test_template_name="template",
        cmd_args=cmd_args,
        repo=GitRepo(
            url="https://github.com/ai-dynamo/dynamo.git",
            commit="f7e468c7e8ff0d1426db987564e60572167e8464",
            installed_path=dynamo_repo_path,
        ),
    )

    return TestRun(name="run", test=tdef, nodes=["n0", "n1"], num_nodes=2, output_path=tmp_path)


@pytest.fixture
def strategy(slurm_system: SlurmSystem, test_run: TestRun) -> AIDynamoSlurmCommandGenStrategy:
    return AIDynamoSlurmCommandGenStrategy(slurm_system, test_run)


def test_container_mounts(strategy: AIDynamoSlurmCommandGenStrategy, test_run: TestRun) -> None:
    mounts = strategy._container_mounts()

    td = test_run.test
    expected = [
        f"{strategy.system.hf_home_path.absolute()}:{strategy.CONTAINER_MOUNT_HF_HOME}",
    ]
    if td.cmd_args.storage_cache_dir:
        expected.append(f"{td.cmd_args.storage_cache_dir}:{td.cmd_args.storage_cache_dir}")
    assert mounts == expected


@pytest.mark.parametrize(
    "module, config, expected",
    [
        ("graphs.agg:Frontend", "cfg.yaml", "dynamo serve graphs.agg:Frontend -f cfg.yaml"),
        (
            "components.worker:VllmPrefillWorker",
            "prefill.yaml",
            "dynamo serve components.worker:VllmPrefillWorker -f prefill.yaml",
        ),
        (
            "components.worker:VllmDecodeWorker",
            "decode.yaml",
            "dynamo serve components.worker:VllmDecodeWorker -f decode.yaml",
        ),
    ],
)
def test_dynamo_cmd(
    strategy: AIDynamoSlurmCommandGenStrategy,
    module: str,
    config: str,
    expected: str,
) -> None:
    result = strategy.gen_dynamo_cmd(module, Path(config))
    assert result.strip() == expected


def test_gen_script_args_contains_split_aiperf_accuracy_args(strategy: AIDynamoSlurmCommandGenStrategy) -> None:
    td = cast(AIDynamoTestDefinition, strategy.test_run.test)
    td.cmd_args.workloads = "aiperf.sh"
    setup_cmd = "python -m pip install --break-system-packages --upgrade aiperf==0.8.0"
    cli = (
        "--model {model} "
        "--url {url} "
        "--endpoint-type chat "
        "--streaming "
        "--artifact-dir {artifact_dir} "
        "--no-server-metrics "
        "--accuracy-benchmark mmlu "
        "--accuracy-n-shots 5 "
        "--accuracy-tasks abstract_algebra "
        "--concurrency 10 "
        '--extra-inputs \'{"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}\' '
        "--num-requests 100"
    )
    td.cmd_args.aiperf = AIPerf.model_validate(
        {
            "args": {
                "concurrency": 2,
                "request-count": 50,
                "synthetic-input-tokens-mean": 300,
                "output-tokens-mean": 500,
            },
        }
    )
    td.cmd_args.aiperf_accuracy = AIPerfAccuracy.model_validate(
        {
            "setup-cmd": setup_cmd,
            "cli": cli,
        }
    )

    result = strategy._gen_script_args(td)

    script = (strategy.test_run.output_path / "aiperf.sh").read_text()
    assert "--request-count 50" in script
    assert "--synthetic-input-tokens-mean 300" in script
    assert "--output-tokens-mean 500" in script
    assert f'--aiperf_accuracy-setup-cmd "{setup_cmd}"' in result
    assert '--aiperf_accuracy-name "aiperf_accuracy"' in result
    assert '--aiperf_accuracy-entrypoint "aiperf profile"' in result
    assert '--aiperf_accuracy-artifact-dir-name "aiperf_accuracy_artifacts"' in result
    assert f"--aiperf_accuracy-cli {shlex.quote(cli)}" in result


def test_gen_script_args_contains_custom_aiperf_accuracy_args(strategy: AIDynamoSlurmCommandGenStrategy) -> None:
    td = cast(AIDynamoTestDefinition, strategy.test_run.test)
    cli = "--model {model} --url {url} --endpoint {endpoint} --artifact-dir {artifact_dir} --prompt ping"
    td.cmd_args.aiperf_accuracy = AIPerfAccuracy.model_validate(
        {
            "entrypoint": "python /custom_accuracy/dummy_accuracy.py",
            "cli": cli,
        }
    )

    result = strategy._gen_script_args(td)

    assert '--aiperf_accuracy-entrypoint "python /custom_accuracy/dummy_accuracy.py"' in result
    assert f'--aiperf_accuracy-cli "{cli}"' in result


def test_gen_script_args_writes_resolved_aiperf_script(strategy: AIDynamoSlurmCommandGenStrategy) -> None:
    td = cast(AIDynamoTestDefinition, strategy.test_run.test)
    td.cmd_args.workloads = "aiperf.sh"
    td.cmd_args.aiperf = AIPerf.model_validate(
        {
            "setup-cmd": "python -m pip install --upgrade aiperf",
            "args": {
                "concurrency": 2,
                "request-count": 50,
                "synthetic-input-tokens-mean": 300,
                "output-tokens-mean": 500,
            },
        }
    )
    td.cmd_args.aiperf_phases = [
        AIPerfPhase.model_validate({"name": "round_1", "args": {"concurrency": 1}}),
        AIPerfPhase.model_validate({"name": "round_2", "args": {"request-count": 10}}),
    ]

    result = strategy._gen_script_args(td)

    assert f"--aiperf-script {strategy.CONTAINER_MOUNT_OUTPUT}/aiperf.sh" in result
    script = (strategy.test_run.output_path / "aiperf.sh").read_text()
    assert "bash -lc 'python -m pip install --upgrade aiperf'" in script
    assert ': "${FRONTEND_URL:?FRONTEND_URL is not set}"' in script
    assert '--url "$FRONTEND_URL"' in script
    assert f"--artifact-dir {strategy.CONTAINER_MOUNT_OUTPUT}/aiperf_artifacts/round_1" in script
    assert f"--artifact-dir {strategy.CONTAINER_MOUNT_OUTPUT}/aiperf_artifacts/round_2" in script
    assert "--concurrency 1 --request-count 50" in script
    assert "--concurrency 2 --request-count 10" in script
    assert f"{strategy.CONTAINER_MOUNT_OUTPUT}/aiperf_round_1.log" in script
    assert f"{strategy.CONTAINER_MOUNT_OUTPUT}/aiperf_round_1_report.csv" in script
    assert f"{strategy.CONTAINER_MOUNT_OUTPUT}/aiperf_round_2_report.csv" in script
    assert f"{strategy.CONTAINER_MOUNT_OUTPUT}/aiperf_report.csv" in script


def test_generated_aiperf_script_supports_core_overrides_and_server_metrics_auto(
    strategy: AIDynamoSlurmCommandGenStrategy,
) -> None:
    td = cast(AIDynamoTestDefinition, strategy.test_run.test)
    td.cmd_args.workloads = "aiperf.sh"
    td.cmd_args.aiperf = AIPerf.model_validate(
        {
            "args": {
                "model": "custom-model",
                "endpoint-type": "completions",
                "streaming": False,
                "server-metrics": "auto",
                "request-count": 10,
            },
        }
    )

    strategy._gen_script_args(td)

    script = (strategy.test_run.output_path / "aiperf.sh").read_text()
    assert "--model custom-model" in script
    assert "--endpoint-type completions" in script
    assert "--streaming" not in script
    assert '--server-metrics "$AIPERF_SERVER_METRICS_URLS"' in script
    assert "--no-server-metrics" not in script


def test_dcgm_exporter_generates_launcher_and_runtime_flags(strategy: AIDynamoSlurmCommandGenStrategy) -> None:
    td = cast(AIDynamoTestDefinition, strategy.test_run.test)
    td.cmd_args.dynamo.dcgm_exporter.enabled = True
    td.cmd_args.dynamo.dcgm_exporter.docker_image_url = "nvcr.io/test/dcgm:latest"
    td.cmd_args.dynamo.dcgm_exporter.port = 9501

    args = strategy._gen_script_args(td)
    block = strategy._gen_dcgm_launcher_block()

    assert '--dynamo-dcgm-exporter-enabled "True"' in args
    assert '--dynamo-dcgm-exporter-port "9501"' in args
    assert any("--container-image=nvcr.io/test/dcgm:latest" in line for line in block)
    assert any("DCGM_EXPORTER_LISTEN=:9501" in line for line in block)
    assert not any("docker run" in line for line in block)


def test_dcgm_exporter_adds_configured_docker_image_installable(cmd_args: AIDynamoCmdArgs) -> None:
    cmd_args.dynamo.dcgm_exporter.enabled = True
    cmd_args.dynamo.dcgm_exporter.docker_image_url = "nvcr.io/test/dcgm:latest"
    tdef = AIDynamoTestDefinition(
        name="test",
        description="desc",
        test_template_name="template",
        cmd_args=cmd_args,
    )

    assert tdef.dcgm_exporter_image is not None
    assert tdef.dcgm_exporter_image.url == "nvcr.io/test/dcgm:latest"
    assert tdef.dcgm_exporter_image in tdef.installables


def test_aiperf_phase_roundtrip_does_not_emit_default_report_name(strategy: AIDynamoSlurmCommandGenStrategy) -> None:
    td = cast(AIDynamoTestDefinition, strategy.test_run.test)
    td.cmd_args.workloads = "aiperf.sh"
    td.cmd_args.aiperf_phases = [
        AIPerfPhase.model_validate({"name": "round_1"}),
        AIPerfPhase.model_validate({"name": "round_2"}),
    ]

    roundtripped = AIDynamoTestDefinition.model_validate(td.model_dump())
    strategy.test_run.test = roundtripped

    assert roundtripped.cmd_args.aiperf_phases is not None
    assert [phase.report_name for phase in roundtripped.cmd_args.aiperf_phases] == [None, None]

    strategy._gen_script_args(roundtripped)

    script = (strategy.test_run.output_path / "aiperf.sh").read_text()
    assert f"{strategy.CONTAINER_MOUNT_OUTPUT}/aiperf_round_1_report.csv" in script
    assert f"{strategy.CONTAINER_MOUNT_OUTPUT}/aiperf_round_2_report.csv" in script


def test_single_aiperf_phase_keeps_legacy_artifact_defaults(strategy: AIDynamoSlurmCommandGenStrategy) -> None:
    td = cast(AIDynamoTestDefinition, strategy.test_run.test)
    td.cmd_args.workloads = "aiperf.sh"
    td.cmd_args.aiperf_phases = [AIPerfPhase.model_validate({"name": "round_1", "args": {"request-count": 10}})]

    strategy._gen_script_args(td)

    script = (strategy.test_run.output_path / "aiperf.sh").read_text()
    assert f"{strategy.CONTAINER_MOUNT_OUTPUT}/aiperf_artifacts" in script
    assert f"{strategy.CONTAINER_MOUNT_OUTPUT}/aiperf_report.csv" in script
    assert f"{strategy.CONTAINER_MOUNT_OUTPUT}/aiperf_round_1.log" not in script


def test_aiperf_phase_names_must_be_unique(cmd_args: AIDynamoCmdArgs) -> None:
    with pytest.raises(ValueError, match="AIPerf phase names must be unique"):
        AIDynamoCmdArgs(
            docker_image_url=cmd_args.docker_image_url,
            dynamo=cmd_args.dynamo,
            aiperf_phases=[
                AIPerfPhase.model_validate({"name": "round_1"}),
                AIPerfPhase.model_validate({"name": "round_1"}),
            ],
        )


def test_gen_script_args_quotes_worker_json_args(strategy: AIDynamoSlurmCommandGenStrategy) -> None:
    td = cast(AIDynamoTestDefinition, strategy.test_run.test)
    config = '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
    td.cmd_args.dynamo.prefill_worker.args = WorkerBaseArgs.model_validate({"kv-transfer-config": config})
    td.cmd_args.dynamo.decode_worker.args = WorkerBaseArgs.model_validate({"kv-transfer-config": config})

    result = strategy._gen_script_args(td)

    assert f"--prefill-args-kv-transfer-config '{config}'" in result
    assert f"--decode-args-kv-transfer-config '{config}'" in result


def test_gen_script_args_writes_lmcache_object_as_yaml(strategy: AIDynamoSlurmCommandGenStrategy) -> None:
    td = cast(AIDynamoTestDefinition, strategy.test_run.test)
    td.cmd_args.lmcache = {
        "chunk_size": 512,
        "local_cpu": True,
        "controller_pull_url": "{frontend_node}:8300",
        "controller_reply_url": "{frontend_node}:8400",
        "lmcache_worker_ports": [8788, 8789, 8790, 8791],
        "extra_config": {
            "enable_nixl_storage": False,
            "nixl_backend": "POSIX",
            "nixl_path": "{storage_cache_dir}",
        },
    }

    result = strategy._gen_script_args(td)

    config_path = strategy.test_run.output_path / LMCACHE_CONFIG_FILE_NAME
    backup_path = strategy.test_run.output_path / LMCACHE_CONFIG_BACKUP_FILE_NAME
    config = yaml.safe_load(config_path.read_text())
    backup_config = yaml.safe_load(backup_path.read_text())
    assert (
        strategy.final_env_vars["LMCACHE_CONFIG_FILE"]
        == f"{strategy.CONTAINER_MOUNT_OUTPUT}/{LMCACHE_CONFIG_FILE_NAME}"
    )
    assert config["chunk_size"] == 512
    assert config["local_cpu"] is True
    assert config["controller_pull_url"] == "{frontend_node}:8300"
    assert config["controller_reply_url"] == "{frontend_node}:8400"
    assert config["lmcache_worker_ports"] == [8788, 8789, 8790, 8791]
    assert config["extra_config"]["enable_nixl_storage"] is False
    assert config["extra_config"]["nixl_backend"] == "POSIX"
    assert config["extra_config"]["nixl_path"] == "{storage_cache_dir}"
    assert backup_config == config
    assert not any(arg.startswith("--lmcache") for arg in result)


def test_lmcache_config_supports_dse_with_excluded_prefix(test_run: TestRun) -> None:
    td = cast(AIDynamoTestDefinition, test_run.test)
    td.dse_excluded_args = ["cmd_args.lmcache.lmcache_worker_ports"]
    td.cmd_args.lmcache = {
        "chunk_size": [256, 512],
        "lmcache_worker_ports": [8788, 8789, 8790, 8791],
    }

    assert test_run.is_dse_job is True
    assert test_run.param_space["lmcache.chunk_size"] == [256, 512]
    assert "lmcache.lmcache_worker_ports" not in test_run.param_space

    new_test_run = test_run.apply_params_set({"lmcache.chunk_size": 512})

    assert cast(AIDynamoTestDefinition, new_test_run.test).cmd_args.lmcache["chunk_size"] == 512  # type: ignore


def test_gen_script_args_passes_lmcache_controller_cmd(strategy: AIDynamoSlurmCommandGenStrategy) -> None:
    td = cast(AIDynamoTestDefinition, strategy.test_run.test)
    cmd = "lmcache_controller --host 0.0.0.0 --port 9000 --monitor-port 9001"
    td.cmd_args.lmcache_controller = LMCacheController(cmd=cmd)

    result = strategy._gen_script_args(td)

    assert f"--lmcache-controller-cmd {shlex.quote(cmd)}" in result
