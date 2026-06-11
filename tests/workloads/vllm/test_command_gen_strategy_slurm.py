# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path
from typing import cast

import pytest

from cloudai.core import TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.vllm import (
    VllmArgs,
    VllmBenchCmdArgs,
    VllmCmdArgs,
    VllmSemanticEvalCmdArgs,
    VllmSlurmCommandGenStrategy,
    VllmTestDefinition,
)


@pytest.fixture
def vllm() -> VllmTestDefinition:
    return VllmTestDefinition(
        name="vllm_test",
        description="vLLM benchmark test",
        test_template_name="Vllm",
        cmd_args=VllmCmdArgs(docker_image_url="nvcr.io/nvidia/vllm:latest", model="Qwen/Qwen3-0.6B", port=8000),
        extra_env_vars={"CUDA_VISIBLE_DEVICES": "0"},
    )


@pytest.fixture
def vllm_tr(vllm: VllmTestDefinition, tmp_path: Path) -> TestRun:
    return TestRun(test=vllm, num_nodes=1, nodes=[], output_path=tmp_path, name="vllm-job")


@pytest.fixture
def vllm_cmd_gen_strategy(vllm_tr: TestRun, slurm_system: SlurmSystem) -> VllmSlurmCommandGenStrategy:
    return VllmSlurmCommandGenStrategy(slurm_system, vllm_tr)


@pytest.fixture
def vllm_disagg_tr(vllm: VllmTestDefinition, tmp_path: Path) -> TestRun:
    """TestRun for disaggregated mode with 4 GPUs."""
    vllm.extra_env_vars = {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    vllm.cmd_args.prefill = VllmArgs()
    return TestRun(test=vllm, num_nodes=1, nodes=[], output_path=tmp_path, name="vllm-disagg-job")


@pytest.fixture
def vllm_disagg_2node_tr(vllm: VllmTestDefinition, tmp_path: Path) -> TestRun:
    """TestRun for disaggregated mode with 2 nodes and a shared local GPU view."""
    vllm.extra_env_vars = {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    vllm.cmd_args.prefill = VllmArgs()
    return TestRun(test=vllm, num_nodes=2, nodes=[], output_path=tmp_path, name="vllm-disagg-2node-job")


def test_container_mounts(vllm_cmd_gen_strategy: VllmSlurmCommandGenStrategy) -> None:
    assert vllm_cmd_gen_strategy._container_mounts() == [
        f"{vllm_cmd_gen_strategy.system.hf_home_path.absolute()}:/root/.cache/huggingface"
    ]


def test_sweep_detection(vllm: VllmTestDefinition) -> None:
    assert vllm.is_dse_job is False
    vllm.cmd_args.decode.gpu_ids = ["1"]
    assert vllm.is_dse_job is True


class TestGpuDetection:
    """Tests for GPU detection logic."""

    def test_prefill_nodes_set(self, vllm_tr: TestRun, slurm_system: SlurmSystem) -> None:
        slurm_system.gpus_per_node = 4
        vllm_tr.test.extra_env_vars = {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
        vllm_tr.test.cmd_args.prefill = VllmArgs(gpu_ids="0,3")
        strategy = VllmSlurmCommandGenStrategy(slurm_system, vllm_tr)
        assert strategy.prefill_gpu_ids == [0, 3]

    def test_decode_nodes_set(self, vllm_tr: TestRun, slurm_system: SlurmSystem) -> None:
        slurm_system.gpus_per_node = 4
        vllm_tr.test.extra_env_vars = {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
        vllm_tr.test.cmd_args.decode.gpu_ids = "1,2"
        strategy = VllmSlurmCommandGenStrategy(slurm_system, vllm_tr)
        assert strategy.decode_gpu_ids == [1, 2]

    def test_multinode_disagg_uses_shared_gpu_ids_per_role(
        self, vllm_disagg_2node_tr: TestRun, slurm_system: SlurmSystem
    ) -> None:
        strategy = VllmSlurmCommandGenStrategy(slurm_system, vllm_disagg_2node_tr)
        assert strategy.prefill_gpu_ids == [0, 1, 2, 3]
        assert strategy.decode_gpu_ids == [0, 1, 2, 3]


class TestVllmServeCommand:
    @pytest.mark.parametrize(
        "decode_nthreads,prefill_nthreads",
        [
            (None, None),
            (4, 2),
            (None, 2),
            (4, None),
        ],
    )
    def test_nixl_threads(
        self,
        decode_nthreads: int | None,
        prefill_nthreads: int | None,
        vllm_cmd_gen_strategy: VllmSlurmCommandGenStrategy,
    ) -> None:
        tdef = cast(VllmTestDefinition, vllm_cmd_gen_strategy.test_run.test)
        tdef.cmd_args.prefill = VllmArgs(nixl_threads=prefill_nthreads)
        tdef.cmd_args.decode.nixl_threads = decode_nthreads

        commands = vllm_cmd_gen_strategy.get_serve_commands()

        assert len(commands) == 2

        prefill_cmd = " ".join(commands[0])
        assert "--nixl-threads" not in prefill_cmd
        if prefill_nthreads is not None:
            assert "kv_connector_extra_config" in prefill_cmd
            assert f'"num_threads":{prefill_nthreads}' in prefill_cmd
        else:
            assert all(arg not in prefill_cmd for arg in ["num_threads", "kv_connector_extra_config"])

        decode_cmd = " ".join(commands[1])
        assert "--nixl-threads" not in decode_cmd
        if decode_nthreads is not None:
            assert "kv_connector_extra_config" in decode_cmd
            assert f'"num_threads":{decode_nthreads}' in decode_cmd
        else:
            assert all(arg not in decode_cmd for arg in ["num_threads", "kv_connector_extra_config"])


class TestVllmBenchCommand:
    def test_get_vllm_bench_command_with_extra_args(
        self, vllm: VllmTestDefinition, vllm_tr: TestRun, slurm_system: SlurmSystem
    ) -> None:
        vllm.bench_cmd_args = VllmBenchCmdArgs.model_validate({"extra1": 1, "extra-2": 2, "extra_3": 3})
        vllm_tr.test = vllm
        vllm_cmd_gen_strategy = VllmSlurmCommandGenStrategy(slurm_system, vllm_tr)

        cmd = vllm_cmd_gen_strategy.get_bench_command()

        assert "--extra1 1" in cmd
        assert "--extra-2 2" in cmd
        assert "--extra-3 3" in cmd


class TestVllmSemanticEvalCommand:
    def test_get_vllm_semantic_eval_command_defaults(self, vllm_cmd_gen_strategy: VllmSlurmCommandGenStrategy):
        vllm_test = cast(VllmTestDefinition, vllm_cmd_gen_strategy.test_run.test)
        vllm_test.semantic_eval_cmd_args = VllmSemanticEvalCmdArgs()

        command = vllm_cmd_gen_strategy.get_semantic_eval_command()

        assert command == [
            "python3 /opt/vllm/tests/evals/gsm8k/gsm8k_eval.py",
            "--host http://${NODE} --port 8000 "
            "--num-questions 200 --save-results "
            f"{vllm_cmd_gen_strategy.test_run.output_path.absolute()}/vllm-gsm8k.json",
        ]

    def test_get_vllm_semantic_eval_command_supports_custom_entrypoint_and_cli(
        self, vllm_cmd_gen_strategy: VllmSlurmCommandGenStrategy
    ) -> None:
        vllm_test = cast(VllmTestDefinition, vllm_cmd_gen_strategy.test_run.test)
        vllm_test.semantic_eval_cmd_args = VllmSemanticEvalCmdArgs(
            entrypoint="python3 /custom/eval.py",
            cli="--model {model} --api {url} --out {result_dir}/vllm-gsm8k.json",
        )

        command = vllm_cmd_gen_strategy.get_semantic_eval_command()

        assert command == [
            "python3 /custom/eval.py",
            f"--model Qwen/Qwen3-0.6B --api http://${{NODE}}:8000 "
            f"--out {vllm_cmd_gen_strategy.test_run.output_path.absolute()}/vllm-gsm8k.json",
        ]

    def test_gen_srun_command_contains_vllm_semantic_eval(
        self, vllm_cmd_gen_strategy: VllmSlurmCommandGenStrategy
    ) -> None:
        vllm_test = cast(VllmTestDefinition, vllm_cmd_gen_strategy.test_run.test)
        vllm_test.semantic_eval_cmd_args = VllmSemanticEvalCmdArgs()

        srun_command = vllm_cmd_gen_strategy._gen_srun_command()

        assert "Running benchmark..." in srun_command
        assert "Running semantic validation..." in srun_command
        assert (
            "--output=" + str((vllm_cmd_gen_strategy.test_run.output_path / "vllm-semantic-eval.log").absolute())
            in srun_command
        )
        assert "python3 /opt/vllm/tests/evals/gsm8k/gsm8k_eval.py --host http://${NODE} --port 8000" in srun_command

    def test_gen_srun_command_contains_vllm_semantic_eval_in_disagg(
        self, vllm_disagg_tr: TestRun, slurm_system: SlurmSystem
    ) -> None:
        vllm_disagg_test = cast(VllmTestDefinition, vllm_disagg_tr.test)
        vllm_disagg_test.semantic_eval_cmd_args = VllmSemanticEvalCmdArgs()
        strategy = VllmSlurmCommandGenStrategy(slurm_system, vllm_disagg_tr)

        srun_command = strategy._gen_srun_command()

        assert "Running semantic validation..." in srun_command
        assert (
            "python3 /opt/vllm/tests/evals/gsm8k/gsm8k_eval.py --host http://${PREFILL_NODE} --port 8000"
            in srun_command
        )


class TestVllmAggregatedMode:
    """Tests for vLLM non-disaggregated mode with 1 GPU."""

    def test_get_vllm_serve_commands_convert_boolean_flags(
        self, vllm: VllmTestDefinition, vllm_tr: TestRun, slurm_system: SlurmSystem
    ) -> None:
        vllm.cmd_args.decode = VllmArgs.model_validate({"enable_expert_parallel": True})
        vllm_tr.test = vllm
        vllm_cmd_gen_strategy = VllmSlurmCommandGenStrategy(slurm_system, vllm_tr)

        commands = vllm_cmd_gen_strategy.get_serve_commands()

        assert commands[0] == [
            "vllm",
            "serve",
            vllm.cmd_args.model,
            "--host",
            vllm.cmd_args.host,
            "--enable-expert-parallel",
            "--port",
            str(vllm.cmd_args.port),
        ]

    def test_generate_wait_for_health_function(self, vllm_cmd_gen_strategy: VllmSlurmCommandGenStrategy) -> None:
        cmd_args = vllm_cmd_gen_strategy.test_run.test.cmd_args

        func = vllm_cmd_gen_strategy.generate_wait_for_health_function()

        expected = f"""\
wait_for_health() {{
    local endpoints=("$@")
    local timeout={cmd_args.serve_wait_seconds}
    local interval=5
    local end_time=$(($(date +%s) + timeout))

    while [ "$(date +%s)" -lt "$end_time" ]; do
        for endpoint in "${{endpoints[@]}}"; do
            if curl -sf "$endpoint" > /dev/null 2>&1; then
                echo "Health check passed: $endpoint"
                return 0
            fi
        done
        sleep "$interval"
    done

    echo "Timeout waiting for: ${{endpoints[*]}}"
    return 1
}}"""

        assert func == expected

    def test_custom_bash_string_wraps_aggregated_serve_and_benchmark(
        self, vllm_cmd_gen_strategy: VllmSlurmCommandGenStrategy
    ) -> None:
        tdef = cast(VllmTestDefinition, vllm_cmd_gen_strategy.test_run.test)
        tdef.custom_bash = "echo setup"

        srun_command = vllm_cmd_gen_strategy._gen_srun_command()

        assert srun_command.count("bash -c ") == 2
        assert "echo setup; exec vllm serve" in srun_command
        assert "echo setup; exec vllm bench serve" in srun_command

    def test_custom_bash_regex_can_target_only_aggregated_benchmark(
        self, vllm_cmd_gen_strategy: VllmSlurmCommandGenStrategy
    ):
        tdef = cast(VllmTestDefinition, vllm_cmd_gen_strategy.test_run.test)
        tdef.custom_bash = {"vllm bench serve": "echo bench setup"}

        srun_command = vllm_cmd_gen_strategy._gen_srun_command()

        assert srun_command.count("bash -c ") == 1
        assert "echo bench setup; exec vllm bench serve" in srun_command
        assert "echo bench setup; exec vllm serve" not in srun_command

    def test_custom_healthcheck_endpoints(
        self, vllm: VllmTestDefinition, vllm_tr: TestRun, slurm_system: SlurmSystem
    ) -> None:
        vllm.cmd_args.healthcheck = "/ready"
        vllm_tr.test = vllm
        aggregated = VllmSlurmCommandGenStrategy(slurm_system, vllm_tr)._gen_srun_command()
        assert 'wait_for_health "http://${NODE}:8000/ready"' in aggregated
        assert 'wait_for_health "http://${NODE}:8000/ready" "http://${NODE}:8000/healthcheck"' not in aggregated

        vllm.cmd_args.prefill = VllmArgs()
        vllm.cmd_args.proxy_healthcheck = "/router-ready"
        vllm_tr.num_nodes = 2
        disaggregated = VllmSlurmCommandGenStrategy(slurm_system, vllm_tr)._gen_srun_command()
        assert 'wait_for_health "http://${PREFILL_NODE}:8000/router-ready"' in disaggregated
        assert (
            'wait_for_health "http://${PREFILL_NODE}:8000/router-ready" "http://${PREFILL_NODE}:8000/ready"'
            not in disaggregated
        )

    def test_disagg_custom_healthcheck_preserves_legacy_proxy_endpoint(
        self, vllm: VllmTestDefinition, vllm_tr: TestRun, slurm_system: SlurmSystem
    ) -> None:
        vllm.cmd_args.healthcheck = "/legacy-ready"
        vllm.cmd_args.prefill = VllmArgs()
        vllm_tr.test = vllm
        vllm_tr.num_nodes = 2

        disaggregated = VllmSlurmCommandGenStrategy(slurm_system, vllm_tr)._gen_srun_command()

        assert 'wait_for_health "http://${PREFILL_NODE}:8000/legacy-ready"' in disaggregated


class TestVllmDisaggregatedMode:
    """Tests for vLLM disaggregated mode with multiple GPUs."""

    def test_custom_bash_regex_can_target_disaggregated_commands(
        self, vllm_disagg_tr: TestRun, slurm_system: SlurmSystem
    ) -> None:
        tdef = cast(VllmTestDefinition, vllm_disagg_tr.test)
        tdef.custom_bash = {
            r'CUDA_VISIBLE_DEVICES="0,1".*vllm serve': "echo prefill setup",
            r'CUDA_VISIBLE_DEVICES="2,3".*vllm serve': "echo decode setup",
            "toy_proxy_server": "echo router setup",
            "vllm bench serve": "echo bench setup",
        }
        strategy = VllmSlurmCommandGenStrategy(slurm_system, vllm_disagg_tr)

        srun_command = strategy._gen_srun_command()

        assert srun_command.count("bash -c ") == 4
        assert "echo prefill setup; exec env" in srun_command
        assert "echo decode setup; exec env" in srun_command
        assert "echo router setup; exec python3" in srun_command
        assert "echo bench setup; exec vllm bench serve" in srun_command

    def test_disagg_more_than_two_nodes_is_rejected(self, vllm_disagg_tr: TestRun, slurm_system: SlurmSystem) -> None:
        vllm_disagg_tr.num_nodes = 3
        strategy = VllmSlurmCommandGenStrategy(slurm_system, vllm_disagg_tr)

        with pytest.raises(ValueError, match=r"requires both prefill\.num_nodes and decode\.num_nodes"):
            _ = strategy._gen_srun_command()

    def test_gen_srun_command_disagg_four_nodes_uses_role_ray_clusters(
        self, vllm_disagg_tr: TestRun, slurm_system: SlurmSystem
    ) -> None:
        tdef = cast(VllmTestDefinition, vllm_disagg_tr.test)
        assert tdef.cmd_args.prefill is not None
        tdef.cmd_args.prefill.num_nodes = 2
        tdef.cmd_args.decode.num_nodes = 2
        vllm_disagg_tr.num_nodes = 4
        strategy = VllmSlurmCommandGenStrategy(slurm_system, vllm_disagg_tr)

        srun_command = strategy._gen_srun_command()

        assert "--distributed-executor-backend ray" in srun_command
        assert "export PREFILL_RAY_PORT=$((6379 + PORT_OFFSET))" in srun_command
        assert "export DECODE_RAY_PORT=$((7379 + PORT_OFFSET))" in srun_command
        assert 'PREFILL_NODES=( "${NODES[@]:0:2}" )' in srun_command
        assert 'DECODE_NODES=( "${NODES[@]:2:2}" )' in srun_command
        assert "PREFILL_RAY_PID=$!" in srun_command
        assert "DECODE_RAY_PID=$!" in srun_command
        assert srun_command.count('sum(node["Alive"] for node in ray.nodes())') == 2
        assert "ray.init(address=" not in srun_command
        assert 'env RAY_ADDRESS="${PREFILL_NODE}:${PREFILL_RAY_PORT}"' in srun_command
        assert 'env RAY_ADDRESS="${DECODE_NODE}:${DECODE_RAY_PORT}"' in srun_command
