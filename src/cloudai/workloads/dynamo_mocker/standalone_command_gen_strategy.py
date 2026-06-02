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

import shlex
import stat
from pathlib import Path
from typing import List, Optional, cast

import toml

from cloudai.core import CommandGenStrategy
from cloudai.models.scenario import TestRunDetails

from .dynamo_mocker import DynamoMockerCmdArgs, DynamoMockerTestDefinition

_SCRIPT_DIR = Path(__file__).parent


class DynamoMockerStandaloneCommandGenStrategy(CommandGenStrategy):
    """
    Command generation strategy for DynamoMocker on StandaloneSystem.

    Builds the command that invokes dynamo_mocker.sh with all configured
    parameters and redirects stdout/stderr to the output directory.
    """

    @staticmethod
    def _extra_flags(model_extra: Optional[dict], prefix: str) -> List[str]:
        """
        Convert a model_extra dict to flat --<prefix>-<key> value pairs.

        Python booleans are lowercased (True → "true") so the bash boolean
        check [[ "$val" == "true" ]] correctly converts them to bare flags.
        """
        result: List[str] = []
        for key, val in (model_extra or {}).items():
            str_val = str(val).lower() if isinstance(val, bool) else str(val)
            result += [f"--{prefix}-{key.replace('_', '-')}", str_val]
        return result

    def _worker_flags(self, args: DynamoMockerCmdArgs) -> List[str]:
        """Return disaggregation-mode-specific flags for the worker section."""
        w = args.worker
        if w.disaggregation_mode == "none":
            flags: List[str] = ["--num-workers", str(w.num_workers)]
        else:
            # prefill_decode: per-role instance counts, commands, KV bandwidth, init regex
            flags = [
                "--prefill-num-nodes",
                str(w.prefill_worker.num_nodes),
                "--decode-num-nodes",
                str(w.decode_worker.num_nodes),
                "--kv-transfer-bandwidth",
                str(args.engine.kv_transfer_bandwidth),
                "--prefill-initialized-regex",
                w.prefill_worker.worker_initialized_regex,
                "--decode-initialized-regex",
                w.decode_worker.worker_initialized_regex,
            ]
            if w.prefill_worker.cmd:
                flags += ["--prefill-cmd", w.prefill_worker.cmd]
            if w.decode_worker.cmd:
                flags += ["--decode-cmd", w.decode_worker.cmd]

        # Per-worker instance args and extra_args are emitted regardless of mode.
        flags += self._extra_flags(w.prefill_worker.args.model_extra, "prefill-args")
        if w.prefill_worker.extra_args:
            flags += ["--prefill-extra-args", w.prefill_worker.extra_args]
        flags += self._extra_flags(w.decode_worker.args.model_extra, "decode-args")
        if w.decode_worker.extra_args:
            flags += ["--decode-extra-args", w.decode_worker.extra_args]
        return flags

    def _benchmark_flags(self, args: DynamoMockerCmdArgs) -> List[str]:
        """
        Return benchmark-tool cmd, extra_args, and passthrough flags.

        cmd/extra_args are named fields (not in model_extra) so they are
        emitted explicitly before the model_extra wildcard loop.
        """
        if args.benchmark_tool == "aiperf":
            bench, prefix = args.aiperf, "aiperf"
        else:
            bench, prefix = args.genai_perf, "genai-perf"

        flags: List[str] = []
        if bench.cmd:
            flags += [f"--{prefix}-cmd", bench.cmd]
        if bench.extra_args:
            flags += [f"--{prefix}-extra-args", bench.extra_args]
        flags += self._extra_flags(bench.model_extra, prefix)
        return flags

    def _build_script_args(self, args: DynamoMockerCmdArgs, result_dir: Path, python_exec: Path) -> List[str]:
        """Return the CLI arg list to pass to dynamo_mocker.sh."""
        e = args.engine
        w = args.worker
        f = args.frontend
        bench = args.aiperf if args.benchmark_tool == "aiperf" else args.genai_perf

        cli_args = [
            "--venv-python",
            str(python_exec),
            "--result-dir",
            str(result_dir),
            "--model-path",
            args.model_path,
            "--nats-cmd",
            args.nats_cmd,
            "--speedup-ratio",
            str(e.speedup_ratio),
            "--block-size",
            str(e.block_size),
            "--num-gpu-blocks-override",
            str(e.num_gpu_blocks_override),
            "--enable-prefix-caching",
            str(e.enable_prefix_caching).lower(),
            "--http-port",
            str(f.http_port),
            "--router-mode",
            f.router_mode,
            "--disaggregation-mode",
            w.disaggregation_mode,
            "--benchmark-tool",
            args.benchmark_tool,
            "--input-tokens",
            str(bench.input_tokens),
            "--output-tokens",
            str(bench.output_tokens),
            "--request-count",
            str(bench.request_count),
            "--replay-concurrency",
            str(bench.replay_concurrency),
            "--replay-mode",
            bench.replay_mode,
        ]
        cli_args += self._worker_flags(args)
        # Forward extras: engine → --engine-*, worker → --mocker-*, frontend → --frontend-*
        cli_args += self._extra_flags(e.model_extra, "engine")
        cli_args += self._extra_flags(w.model_extra, "mocker")
        cli_args += self._extra_flags(f.model_extra, "frontend")
        cli_args += self._benchmark_flags(args)
        return cli_args

    def gen_exec_command(self) -> str:
        tdef = cast(DynamoMockerTestDefinition, self.test_run.test)
        args: DynamoMockerCmdArgs = tdef.cmd_args
        output_path = self.test_run.output_path
        output_path.mkdir(parents=True, exist_ok=True)

        if tdef.python_environment.venv_path is None:
            raise RuntimeError("Unexpected state: python environment is not installed yet")
        python_exec = tdef.python_environment.venv_path / "bin" / "python"

        script_path = _SCRIPT_DIR / "dynamo_mocker.sh"
        script_args = self._build_script_args(args, output_path, python_exec)
        wrapper_path = output_path / "dynamo_mocker_run.sh"

        # Wrapper script: redirects stdout/stderr for cloudai log capture
        stdout_file = output_path / "stdout.txt"
        stderr_file = output_path / "stderr.txt"
        quoted_args = " ".join(shlex.quote(a) for a in script_args)
        wrapper_lines = [
            "#!/usr/bin/env bash",
            (
                f"bash {shlex.quote(str(script_path))} {quoted_args}"
                f" > {shlex.quote(str(stdout_file))} 2> {shlex.quote(str(stderr_file))}"
            ),
        ]
        wrapper_path.write_text("\n".join(wrapper_lines), encoding="utf-8")
        wrapper_path.chmod(wrapper_path.stat().st_mode | stat.S_IXUSR)

        self.store_test_run()

        return f'bash "{wrapper_path}"'

    def store_test_run(self) -> None:
        """Persist TestRunDetails to the output directory for inspection."""
        tdef = cast(DynamoMockerTestDefinition, self.test_run.test)
        args: DynamoMockerCmdArgs = tdef.cmd_args
        output_path = self.test_run.output_path
        output_path.mkdir(parents=True, exist_ok=True)

        if tdef.python_environment.venv_path is None:
            raise RuntimeError("Unexpected state: python environment is not installed yet")
        python_exec = tdef.python_environment.venv_path / "bin" / "python"

        script_path = _SCRIPT_DIR / "dynamo_mocker.sh"
        script_args = self._build_script_args(args, output_path, python_exec)
        wrapper_path = output_path / "dynamo_mocker_run.sh"

        test_cmd = "bash " + shlex.quote(str(script_path)) + " " + " ".join(shlex.quote(a) for a in script_args)
        full_cmd = f'bash "{wrapper_path}"'
        dump_path = output_path / self.TEST_RUN_DUMP_FILE_NAME
        trd = TestRunDetails.from_test_run(self.test_run, test_cmd=test_cmd, full_cmd=full_cmd)
        with dump_path.open("w", encoding="utf-8") as f:
            toml.dump(trd.model_dump(), f)
