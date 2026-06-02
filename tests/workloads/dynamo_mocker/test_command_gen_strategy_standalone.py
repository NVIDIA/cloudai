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

import pytest

from cloudai.core import TestRun
from cloudai.systems.standalone import StandaloneSystem
from cloudai.workloads.dynamo_mocker.dynamo_mocker import (
    DynamoMockerCmdArgs,
    DynamoMockerTestDefinition,
    MockerAIPerfArgs,
    MockerEngineArgs,
    MockerFrontendArgs,
    MockerGenAIPerfArgs,
    MockerWorkerConfig,
    MockerWorkerInstance,
    MockerWorkerInstanceArgs,
)
from cloudai.workloads.dynamo_mocker.standalone_command_gen_strategy import (
    DynamoMockerStandaloneCommandGenStrategy,
)

_VENV_PYTHON = Path("/venv/bin/python")


@pytest.fixture
def standalone_system(tmp_path: Path) -> StandaloneSystem:
    return StandaloneSystem(
        name="test_system",
        install_path=tmp_path / "install",
        output_path=tmp_path / "output",
    )


def _make_strategy(
    system: StandaloneSystem,
    tmp_path: Path,
    cmd_args: DynamoMockerCmdArgs | None = None,
) -> DynamoMockerStandaloneCommandGenStrategy:
    if cmd_args is None:
        cmd_args = DynamoMockerCmdArgs()
    tdef = DynamoMockerTestDefinition(
        name="dynamo_mocker",
        description="test",
        test_template_name="DynamoMockerTest",
        cmd_args=cmd_args,
    )
    # Simulate post-install state: venv_path is set on the cached instance.
    tdef.python_environment.venv_path = system.install_path / tdef.python_environment.venv_name
    tr = TestRun(
        name="test-run",
        test=tdef,
        num_nodes=1,
        nodes=[],
        output_path=tmp_path / "output" / "test-run",
    )
    return DynamoMockerStandaloneCommandGenStrategy(system=system, test_run=tr)


class TestBuildScriptArgsCombined:
    """_build_script_args in combined (disaggregation_mode=none) mode."""

    def test_contains_required_flags(self, standalone_system: StandaloneSystem, tmp_path: Path):
        strategy = _make_strategy(standalone_system, tmp_path)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--model-path" in args
        assert "--disaggregation-mode" in args
        assert "none" in args
        assert "--num-workers" in args

    def test_no_disaggregated_flags(self, standalone_system: StandaloneSystem, tmp_path: Path):
        strategy = _make_strategy(standalone_system, tmp_path)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--num-prefill-workers" not in args
        assert "--num-decode-workers" not in args
        assert "--kv-transfer-bandwidth" not in args

    def test_venv_python_is_first(self, standalone_system: StandaloneSystem, tmp_path: Path):
        result_dir = tmp_path / "result"
        strategy = _make_strategy(standalone_system, tmp_path)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, result_dir, _VENV_PYTHON)
        assert args[0] == "--venv-python"
        assert args[1] == str(_VENV_PYTHON)
        assert "--result-dir" in args
        assert args[args.index("--result-dir") + 1] == str(result_dir)


class TestBuildScriptArgsDisaggregated:
    """_build_script_args in disaggregated (disaggregation_mode=prefill_decode) mode."""

    def test_contains_disaggregated_flags(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(worker=MockerWorkerConfig(disaggregation_mode="prefill_decode"))
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--disaggregation-mode" in args
        assert "prefill_decode" in args
        assert "--prefill-num-nodes" in args
        assert "--decode-num-nodes" in args
        assert "--kv-transfer-bandwidth" in args
        assert "--prefill-initialized-regex" in args
        assert "--decode-initialized-regex" in args

    def test_no_combined_flag(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(worker=MockerWorkerConfig(disaggregation_mode="prefill_decode"))
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        # --num-workers is combined-mode only
        assert "--num-workers" not in args


class TestBuildScriptArgsBenchmarkTool:
    """benchmark_tool selection routes to the right config section."""

    def test_default_is_genai_perf(self, standalone_system: StandaloneSystem, tmp_path: Path):
        strategy = _make_strategy(standalone_system, tmp_path)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--benchmark-tool" in args
        idx = args.index("--benchmark-tool")
        assert args[idx + 1] == "genai_perf"

    def test_aiperf_sets_benchmark_tool_flag(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(benchmark_tool="aiperf")
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        idx = args.index("--benchmark-tool")
        assert args[idx + 1] == "aiperf"

    def test_aiperf_uses_aiperf_section_values(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(
            benchmark_tool="aiperf",
            aiperf=MockerAIPerfArgs(input_tokens=1234, output_tokens=56, request_count=99),
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert args[args.index("--input-tokens") + 1] == "1234"
        assert args[args.index("--output-tokens") + 1] == "56"
        assert args[args.index("--request-count") + 1] == "99"

    def test_genai_perf_uses_genai_perf_section_values(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(
            benchmark_tool="genai_perf",
            genai_perf=MockerGenAIPerfArgs(input_tokens=7777, output_tokens=88, request_count=11),
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert args[args.index("--input-tokens") + 1] == "7777"
        assert args[args.index("--output-tokens") + 1] == "88"
        assert args[args.index("--request-count") + 1] == "11"

    def test_aiperf_extra_fields_forwarded(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(
            benchmark_tool="aiperf",
            aiperf=MockerAIPerfArgs.model_validate({"arrival_pattern": "poisson", "benchmark_duration": "120"}),
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--aiperf-arrival-pattern" in args
        assert args[args.index("--aiperf-arrival-pattern") + 1] == "poisson"
        assert "--aiperf-benchmark-duration" in args
        assert args[args.index("--aiperf-benchmark-duration") + 1] == "120"

    def test_genai_perf_extra_fields_forwarded(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(
            benchmark_tool="genai_perf",
            genai_perf=MockerGenAIPerfArgs.model_validate({"streaming": "true", "endpoint_type": "chat_completions"}),
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--genai-perf-streaming" in args
        assert args[args.index("--genai-perf-streaming") + 1] == "true"
        assert "--genai-perf-endpoint-type" in args
        assert args[args.index("--genai-perf-endpoint-type") + 1] == "chat_completions"

    def test_genai_perf_no_aiperf_extra_flags(self, standalone_system: StandaloneSystem, tmp_path: Path):
        strategy = _make_strategy(standalone_system, tmp_path)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert not any(a.startswith("--aiperf-") for a in args)

    def test_aiperf_no_genai_perf_extra_flags(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(benchmark_tool="aiperf")
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert not any(a.startswith("--genai-perf-") for a in args)

    def test_bool_extra_field_lowercased_aiperf(self, standalone_system: StandaloneSystem, tmp_path: Path):
        """Python bool True must be forwarded as lowercase 'true' so bash [[ '$val' == 'true' ]] works."""
        cmd_args = DynamoMockerCmdArgs(
            benchmark_tool="aiperf",
            aiperf=MockerAIPerfArgs.model_validate({"some_flag": True}),
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--aiperf-some-flag" in args
        assert args[args.index("--aiperf-some-flag") + 1] == "true"

    def test_bool_extra_field_lowercased_genai_perf(self, standalone_system: StandaloneSystem, tmp_path: Path):
        """Python bool True must be forwarded as lowercase 'true' so bash [[ '$val' == 'true' ]] works."""
        cmd_args = DynamoMockerCmdArgs(
            benchmark_tool="genai_perf",
            genai_perf=MockerGenAIPerfArgs.model_validate({"output_tokens_mean_deterministic": True}),
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--genai-perf-output-tokens-mean-deterministic" in args
        assert args[args.index("--genai-perf-output-tokens-mean-deterministic") + 1] == "true"

    def test_aiperf_cmd_forwarded_when_set(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(
            benchmark_tool="aiperf",
            aiperf=MockerAIPerfArgs(cmd="aiperf profile"),
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--aiperf-cmd" in args
        assert args[args.index("--aiperf-cmd") + 1] == "aiperf profile"

    def test_aiperf_cmd_not_emitted_when_empty(self, standalone_system: StandaloneSystem, tmp_path: Path):
        """Empty cmd (default) means use shell default — no --aiperf-cmd emitted."""
        cmd_args = DynamoMockerCmdArgs(benchmark_tool="aiperf")
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--aiperf-cmd" not in args

    def test_aiperf_extra_args_forwarded(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(
            benchmark_tool="aiperf",
            aiperf=MockerAIPerfArgs(extra_args="--streaming --verbose"),
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--aiperf-extra-args" in args
        assert args[args.index("--aiperf-extra-args") + 1] == "--streaming --verbose"

    def test_aiperf_extra_args_not_emitted_when_none(self, standalone_system: StandaloneSystem, tmp_path: Path):
        """extra_args=None (default) must not emit --aiperf-extra-args."""
        cmd_args = DynamoMockerCmdArgs(benchmark_tool="aiperf")
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--aiperf-extra-args" not in args

    def test_genai_perf_cmd_forwarded_when_set(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(
            benchmark_tool="genai_perf",
            genai_perf=MockerGenAIPerfArgs(cmd="genai-perf profile"),
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--genai-perf-cmd" in args
        assert args[args.index("--genai-perf-cmd") + 1] == "genai-perf profile"

    def test_genai_perf_cmd_not_emitted_when_empty(self, standalone_system: StandaloneSystem, tmp_path: Path):
        """Empty cmd (default) means use shell default — no --genai-perf-cmd emitted."""
        strategy = _make_strategy(standalone_system, tmp_path)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--genai-perf-cmd" not in args

    def test_genai_perf_extra_args_forwarded(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(
            benchmark_tool="genai_perf",
            genai_perf=MockerGenAIPerfArgs(extra_args="--streaming --verbose -- -v --async"),
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--genai-perf-extra-args" in args
        assert args[args.index("--genai-perf-extra-args") + 1] == "--streaming --verbose -- -v --async"

    def test_genai_perf_extra_args_not_emitted_when_none(self, standalone_system: StandaloneSystem, tmp_path: Path):
        """extra_args=None (default) must not emit --genai-perf-extra-args."""
        strategy = _make_strategy(standalone_system, tmp_path)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--genai-perf-extra-args" not in args


class TestGenExecCommand:
    """gen_exec_command writes a wrapper script and returns a bash invocation."""

    def test_returns_bash_invocation(self, standalone_system: StandaloneSystem, tmp_path: Path):
        strategy = _make_strategy(standalone_system, tmp_path)
        cmd = strategy.gen_exec_command()
        assert cmd.startswith("bash ")
        assert "dynamo_mocker_run.sh" in cmd

    def test_wrapper_script_created(self, standalone_system: StandaloneSystem, tmp_path: Path):
        strategy = _make_strategy(standalone_system, tmp_path)
        strategy.gen_exec_command()
        wrapper = strategy.test_run.output_path / "dynamo_mocker_run.sh"
        assert wrapper.exists()

    def test_wrapper_invokes_dynamo_mocker_sh(self, standalone_system: StandaloneSystem, tmp_path: Path):
        strategy = _make_strategy(standalone_system, tmp_path)
        strategy.gen_exec_command()
        content = (strategy.test_run.output_path / "dynamo_mocker_run.sh").read_text()
        assert "dynamo_mocker.sh" in content

    def test_wrapper_redirects_stdout_stderr(self, standalone_system: StandaloneSystem, tmp_path: Path):
        strategy = _make_strategy(standalone_system, tmp_path)
        strategy.gen_exec_command()
        content = (strategy.test_run.output_path / "dynamo_mocker_run.sh").read_text()
        assert "stdout.txt" in content
        assert "stderr.txt" in content
        assert ">" in content


class TestComponentExtraArgs:
    """Extra-args passthrough for engine, worker (mocker), and frontend."""

    def test_engine_extra_forwarded(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(engine=MockerEngineArgs.model_validate({"extra_param": "42"}))
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--engine-extra-param" in args
        assert args[args.index("--engine-extra-param") + 1] == "42"

    def test_engine_extra_bool_lowercased(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(engine=MockerEngineArgs.model_validate({"experimental_flag": True}))
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--engine-experimental-flag" in args
        assert args[args.index("--engine-experimental-flag") + 1] == "true"

    def test_engine_no_extra_produces_no_engine_flags(self, standalone_system: StandaloneSystem, tmp_path: Path):
        strategy = _make_strategy(standalone_system, tmp_path)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert not any(a.startswith("--engine-") for a in args)

    def test_worker_extra_uses_mocker_prefix(self, standalone_system: StandaloneSystem, tmp_path: Path):
        """Worker extras become --mocker-*, never --worker-* (avoids --worker-initialized-regex clash)."""
        cmd_args = DynamoMockerCmdArgs(
            worker=MockerWorkerConfig.model_validate({"disaggregation_mode": "none", "extra_topo_flag": "yes"})
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--mocker-extra-topo-flag" in args
        assert args[args.index("--mocker-extra-topo-flag") + 1] == "yes"
        assert not any(a.startswith("--worker-extra") for a in args)

    def test_worker_extra_bool_lowercased(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(
            worker=MockerWorkerConfig.model_validate({"disaggregation_mode": "none", "some_feature": True})
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--mocker-some-feature" in args
        assert args[args.index("--mocker-some-feature") + 1] == "true"

    def test_worker_no_extra_produces_no_mocker_extra_flags(self, standalone_system: StandaloneSystem, tmp_path: Path):
        strategy = _make_strategy(standalone_system, tmp_path)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert not any(a.startswith("--mocker-") for a in args)

    def test_frontend_extra_forwarded(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(
            frontend=MockerFrontendArgs.model_validate(
                {"http_port": 8000, "router_mode": "round_robin", "grpc_port": "9000"}
            )
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--frontend-grpc-port" in args
        assert args[args.index("--frontend-grpc-port") + 1] == "9000"

    def test_frontend_extra_bool_lowercased(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(
            frontend=MockerFrontendArgs.model_validate(
                {"http_port": 8000, "router_mode": "round_robin", "enable_tls": True}
            )
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--frontend-enable-tls" in args
        assert args[args.index("--frontend-enable-tls") + 1] == "true"

    def test_frontend_no_extra_produces_no_frontend_extra_flags(
        self, standalone_system: StandaloneSystem, tmp_path: Path
    ):
        strategy = _make_strategy(standalone_system, tmp_path)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert not any(a.startswith("--frontend-") for a in args)

    def test_all_three_extras_simultaneously(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(
            engine=MockerEngineArgs.model_validate({"eng_opt": "1"}),
            worker=MockerWorkerConfig.model_validate({"disaggregation_mode": "none", "wkr_opt": "2"}),
            frontend=MockerFrontendArgs.model_validate(
                {"http_port": 8000, "router_mode": "round_robin", "frt_opt": "3"}
            ),
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--engine-eng-opt" in args
        assert "--mocker-wkr-opt" in args
        assert "--frontend-frt-opt" in args

    def test_engine_extra_not_in_mocker_prefix(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(engine=MockerEngineArgs.model_validate({"shared_name": "v"}))
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--engine-shared-name" in args
        assert "--mocker-shared-name" not in args


class TestPrefillDecodeExtraArgs:
    """Nested per-worker args and extra_args passthrough for prefill and decode mocker instances."""

    def test_prefill_args_extra_forwarded(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(
            worker=MockerWorkerConfig(
                disaggregation_mode="prefill_decode",
                prefill_worker=MockerWorkerInstance(
                    args=MockerWorkerInstanceArgs.model_validate({"max_num_seqs": "10"})
                ),
            )
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--prefill-args-max-num-seqs" in args
        assert args[args.index("--prefill-args-max-num-seqs") + 1] == "10"

    def test_decode_args_extra_forwarded(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(
            worker=MockerWorkerConfig(
                disaggregation_mode="prefill_decode",
                decode_worker=MockerWorkerInstance(
                    args=MockerWorkerInstanceArgs.model_validate({"max_num_seqs": "20"})
                ),
            )
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--decode-args-max-num-seqs" in args
        assert args[args.index("--decode-args-max-num-seqs") + 1] == "20"

    def test_prefill_and_decode_different_values(self, standalone_system: StandaloneSystem, tmp_path: Path):
        """Prefill and decode can carry the same flag name with different values."""
        cmd_args = DynamoMockerCmdArgs(
            worker=MockerWorkerConfig(
                disaggregation_mode="prefill_decode",
                prefill_worker=MockerWorkerInstance(
                    args=MockerWorkerInstanceArgs.model_validate({"max_num_seqs": "10"})
                ),
                decode_worker=MockerWorkerInstance(
                    args=MockerWorkerInstanceArgs.model_validate({"max_num_seqs": "20"})
                ),
            )
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--prefill-args-max-num-seqs" in args
        assert args[args.index("--prefill-args-max-num-seqs") + 1] == "10"
        assert "--decode-args-max-num-seqs" in args
        assert args[args.index("--decode-args-max-num-seqs") + 1] == "20"

    def test_prefill_args_not_in_decode_flags(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(
            worker=MockerWorkerConfig(
                disaggregation_mode="prefill_decode",
                prefill_worker=MockerWorkerInstance(
                    args=MockerWorkerInstanceArgs.model_validate({"prefill_only_flag": "x"})
                ),
            )
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--prefill-args-prefill-only-flag" in args
        assert "--decode-args-prefill-only-flag" not in args

    def test_decode_args_not_in_prefill_flags(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(
            worker=MockerWorkerConfig(
                disaggregation_mode="prefill_decode",
                decode_worker=MockerWorkerInstance(
                    args=MockerWorkerInstanceArgs.model_validate({"decode_only_flag": "y"})
                ),
            )
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--decode-args-decode-only-flag" in args
        assert "--prefill-args-decode-only-flag" not in args

    def test_prefill_args_bool_lowercased(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(
            worker=MockerWorkerConfig(
                disaggregation_mode="prefill_decode",
                prefill_worker=MockerWorkerInstance(
                    args=MockerWorkerInstanceArgs.model_validate({"enable_chunked_prefill": True})
                ),
            )
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--prefill-args-enable-chunked-prefill" in args
        assert args[args.index("--prefill-args-enable-chunked-prefill") + 1] == "true"

    def test_no_prefill_decode_extras_by_default(self, standalone_system: StandaloneSystem, tmp_path: Path):
        """Combined mode (default) produces no --prefill-* or --decode-* flags."""
        strategy = _make_strategy(standalone_system, tmp_path)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert not any(a.startswith("--prefill-") for a in args)
        assert not any(a.startswith("--decode-") for a in args)

    def test_prefill_initialized_regex_forwarded(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(
            worker=MockerWorkerConfig(
                disaggregation_mode="prefill_decode",
                prefill_worker=MockerWorkerInstance(worker_initialized_regex="prefill ready"),
            )
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--prefill-initialized-regex" in args
        assert args[args.index("--prefill-initialized-regex") + 1] == "prefill ready"

    def test_decode_initialized_regex_forwarded(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(
            worker=MockerWorkerConfig(
                disaggregation_mode="prefill_decode",
                decode_worker=MockerWorkerInstance(worker_initialized_regex="decode ready"),
            )
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--decode-initialized-regex" in args
        assert args[args.index("--decode-initialized-regex") + 1] == "decode ready"

    def test_prefill_extra_args_raw_forwarded(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(
            worker=MockerWorkerConfig(
                disaggregation_mode="prefill_decode",
                prefill_worker=MockerWorkerInstance(extra_args="--no-enable-expert-parallel"),
            )
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--prefill-extra-args" in args
        assert args[args.index("--prefill-extra-args") + 1] == "--no-enable-expert-parallel"

    def test_decode_extra_args_raw_forwarded(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(
            worker=MockerWorkerConfig(
                disaggregation_mode="prefill_decode",
                decode_worker=MockerWorkerInstance(extra_args="--some-decode-flag"),
            )
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--decode-extra-args" in args
        assert args[args.index("--decode-extra-args") + 1] == "--some-decode-flag"

    def test_no_extra_args_if_none(self, standalone_system: StandaloneSystem, tmp_path: Path):
        """When extra_args is None (default), --prefill-extra-args and --decode-extra-args are not emitted."""
        strategy = _make_strategy(standalone_system, tmp_path)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--prefill-extra-args" not in args
        assert "--decode-extra-args" not in args

    def test_prefill_num_nodes_forwarded(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(
            worker=MockerWorkerConfig(
                disaggregation_mode="prefill_decode",
                prefill_worker=MockerWorkerInstance(num_nodes=3),
            )
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--prefill-num-nodes" in args
        assert args[args.index("--prefill-num-nodes") + 1] == "3"

    def test_decode_num_nodes_forwarded(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(
            worker=MockerWorkerConfig(
                disaggregation_mode="prefill_decode",
                decode_worker=MockerWorkerInstance(num_nodes=2),
            )
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--decode-num-nodes" in args
        assert args[args.index("--decode-num-nodes") + 1] == "2"

    def test_prefill_cmd_forwarded_when_set(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(
            worker=MockerWorkerConfig(
                disaggregation_mode="prefill_decode",
                prefill_worker=MockerWorkerInstance(cmd="python3 -m dynamo.mocker"),
            )
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--prefill-cmd" in args
        assert args[args.index("--prefill-cmd") + 1] == "python3 -m dynamo.mocker"

    def test_decode_cmd_forwarded_when_set(self, standalone_system: StandaloneSystem, tmp_path: Path):
        cmd_args = DynamoMockerCmdArgs(
            worker=MockerWorkerConfig(
                disaggregation_mode="prefill_decode",
                decode_worker=MockerWorkerInstance(cmd="python3 -m dynamo.mocker"),
            )
        )
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--decode-cmd" in args
        assert args[args.index("--decode-cmd") + 1] == "python3 -m dynamo.mocker"

    def test_cmd_not_emitted_when_empty(self, standalone_system: StandaloneSystem, tmp_path: Path):
        """Empty cmd (default) means use shell default — no --prefill-cmd/--decode-cmd emitted."""
        cmd_args = DynamoMockerCmdArgs(worker=MockerWorkerConfig(disaggregation_mode="prefill_decode"))
        strategy = _make_strategy(standalone_system, tmp_path, cmd_args)
        args = strategy._build_script_args(strategy.test_run.test.cmd_args, tmp_path / "result", _VENV_PYTHON)
        assert "--prefill-cmd" not in args
        assert "--decode-cmd" not in args
