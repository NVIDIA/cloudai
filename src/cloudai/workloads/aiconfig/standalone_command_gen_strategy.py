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

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

from cloudai.core import CommandGenStrategy

from .aiconfigurator import Agg, AiconfiguratorCmdArgs, AiconfiguratorTestDefinition, Disagg


class AiconfiguratorStandaloneCommandGenStrategy(CommandGenStrategy):
    """Generate a standalone command that invokes the Aiconfigurator predictor and writes JSON output."""

    def store_test_run(self) -> None:
        return

    def gen_exec_command(self) -> str:
        tdef: AiconfiguratorTestDefinition = cast(AiconfiguratorTestDefinition, self.test_run.test)
        args: AiconfiguratorCmdArgs = tdef.cmd_args
        out_dir = self.test_run.output_path

        report_txt = Path(out_dir) / "report.txt"
        stdout_txt = Path(out_dir) / "stdout.txt"
        stderr_txt = Path(out_dir) / "stderr.txt"

        script_path = Path(__file__).parent / "simple_predictor.py"

        cloudai_src_dir = Path(__file__).resolve().parents[3]

        python_exec = "python"

        base_cmd = [
            python_exec,
            str(script_path),
            "--model-name",
            args.model_name,
            "--system",
            args.system,
            "--backend",
            args.backend,
            "--version",
            args.version,
            "--isl",
            str(args.isl),
            "--osl",
            str(args.osl),
        ]

        if args.agg is not None:
            a = cast(Agg, args.agg)
            cmd = [
                *base_cmd,
                "--mode",
                "agg",
                "--batch-size",
                str(a.batch_size),
                "--ctx-tokens",
                str(a.ctx_tokens),
                "--tp",
                str(a.tp),
                "--pp",
                str(a.pp),
                "--dp",
                str(a.dp),
                "--moe-tp",
                str(a.moe_tp),
                "--moe-ep",
                str(a.moe_ep),
            ]
        elif args.disagg is not None:
            d = cast(Disagg, args.disagg)
            cmd = [
                *base_cmd,
                "--mode",
                "disagg",
                "--p-tp",
                str(d.p_tp),
                "--p-pp",
                str(d.p_pp),
                "--p-dp",
                str(d.p_dp),
                "--p-bs",
                str(d.p_bs),
                "--p-workers",
                str(d.p_workers),
                "--d-tp",
                str(d.d_tp),
                "--d-pp",
                str(d.d_pp),
                "--d-dp",
                str(d.d_dp),
                "--d-bs",
                str(d.d_bs),
                "--d-workers",
                str(d.d_workers),
                "--prefill-correction-scale",
                str(d.prefill_correction_scale),
                "--decode-correction-scale",
                str(d.decode_correction_scale),
            ]
        else:
            cmd = [*base_cmd, "--mode", "agg"]

        cmd.extend(["--output", str(report_txt)])

        script_file = Path(out_dir) / "run_simple_predictor.sh"
        script_lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"export PYTHONPATH='{cloudai_src_dir}':\"${{PYTHONPATH:-}}\"",
            f"{' '.join(cmd)} 1> '{stdout_txt}' 2> '{stderr_txt}'",
        ]
        script_file.parent.mkdir(parents=True, exist_ok=True)
        script_file.write_text("\n".join(script_lines) + "\n", encoding="utf-8")
        os.chmod(script_file, 0o755)

        return f"bash {script_file}"
