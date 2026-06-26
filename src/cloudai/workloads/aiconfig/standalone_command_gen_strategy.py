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

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any, cast

import toml

from cloudai.core import CommandGenStrategy
from cloudai.models.scenario import TestRunDetails

from .aiconfigurator import Agg, AiconfiguratorCmdArgs, AiconfiguratorTestDefinition, Disagg


class AiconfiguratorStandaloneCommandGenStrategy(CommandGenStrategy):
    """Generate a standalone command that invokes the Aiconfigurator predictor and writes JSON output."""

    @staticmethod
    def _scalar(value: Any) -> Any:
        """
        Render a parallelism/batch dim as the single scalar the predictor CLI expects.

        These dims are typed ``Union[int, List[int]]`` so a TOML may express a
        sweep. By command-generation time every dim must collapse to one concrete
        value: DSE resolves tunable dims to scalars, while non-tunable single-value
        dims arrive as one-element lists (e.g. ``p_pp = [1]``). ``simple_predictor.py``
        parses these as ``int``, so a raw ``"[1]"`` is rejected. A multi-element list
        here means an unresolved sweep leaked into command-gen, which is a bug.
        """
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError(
                    f"Aiconfigurator command-gen expected a single resolved value, got {value!r}; "
                    "multi-element sweeps must be resolved by the DSE agent before command generation."
                )
            return value[0]
        return value

    def store_test_run(self) -> None:
        test_cmd, full_cmd = ("", "n/a")
        self.test_run.output_path.mkdir(parents=True, exist_ok=True)
        with (self.test_run.output_path / self.TEST_RUN_DUMP_FILE_NAME).open("w", encoding="utf-8") as f:
            trd = TestRunDetails.from_test_run(self.test_run, test_cmd=test_cmd, full_cmd=full_cmd)
            toml.dump(trd.model_dump(), f)

    def gen_exec_command(self) -> str:
        self.store_test_run()

        tdef: AiconfiguratorTestDefinition = cast(AiconfiguratorTestDefinition, self.test_run.test)
        args: AiconfiguratorCmdArgs = tdef.cmd_args
        out_dir = Path(self.test_run.output_path).resolve()

        report_json = Path(out_dir) / "report.json"
        stdout_txt = Path(out_dir) / "stdout.txt"
        stderr_txt = Path(out_dir) / "stderr.txt"

        python_exec = tdef.python_environment.python_path(self.system.install_path)
        predictor_script = Path(__file__).with_name("runtime") / "simple_predictor.py"

        base_cmd = [
            python_exec,
            predictor_script.resolve(),
            "--model-name",
            args.model_name,
            "--system",
            args.system,
            "--backend",
            args.backend,
            "--version",
            args.version,
            "--isl",
            str(self._scalar(args.isl)),
            "--osl",
            str(self._scalar(args.osl)),
        ]

        if args.agg is not None:
            a = cast(Agg, args.agg)
            cmd = [
                *base_cmd,
                "--mode",
                "agg",
                "--batch-size",
                str(self._scalar(a.batch_size)),
                "--ctx-tokens",
                str(self._scalar(a.ctx_tokens)),
                "--tp",
                str(self._scalar(a.tp)),
                "--pp",
                str(self._scalar(a.pp)),
                "--dp",
                str(self._scalar(a.dp)),
                "--moe-tp",
                str(self._scalar(a.moe_tp)),
                "--moe-ep",
                str(self._scalar(a.moe_ep)),
            ]
        elif args.disagg is not None:
            d = cast(Disagg, args.disagg)
            cmd = [
                *base_cmd,
                "--mode",
                "disagg",
                "--p-tp",
                str(self._scalar(d.p_tp)),
                "--p-pp",
                str(self._scalar(d.p_pp)),
                "--p-dp",
                str(self._scalar(d.p_dp)),
                "--p-bs",
                str(self._scalar(d.p_bs)),
                "--p-workers",
                str(self._scalar(d.p_workers)),
                "--d-tp",
                str(self._scalar(d.d_tp)),
                "--d-pp",
                str(self._scalar(d.d_pp)),
                "--d-dp",
                str(self._scalar(d.d_dp)),
                "--d-bs",
                str(self._scalar(d.d_bs)),
                "--d-workers",
                str(self._scalar(d.d_workers)),
                "--prefill-correction-scale",
                str(d.prefill_correction_scale),
                "--decode-correction-scale",
                str(d.decode_correction_scale),
            ]
        else:
            raise ValueError(
                "Either cmd_args.agg or cmd_args.disagg must be specified for the Aiconfigurator workload."
            )

        cmd.extend(["--output", str(report_json)])

        cmd_str = " ".join(shlex.quote(str(x)) for x in cmd)
        full_cmd = f"{cmd_str} 1> {shlex.quote(str(stdout_txt))} 2> {shlex.quote(str(stderr_txt))}"

        script_file = Path(out_dir) / "run_simple_predictor.sh"
        script_file.parent.mkdir(parents=True, exist_ok=True)
        script_file.write_text(f"#!/usr/bin/env bash\nset -euo pipefail\n{full_cmd}\n", encoding="utf-8")
        os.chmod(script_file, 0o755)

        return f"bash {shlex.quote(str(script_file))}"
