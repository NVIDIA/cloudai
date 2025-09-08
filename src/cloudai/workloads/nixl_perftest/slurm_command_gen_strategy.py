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

from pathlib import Path
from typing import cast

from cloudai.core import TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.common.nixl import NIXLCmdGenBase

from .nixl_perftest import NixlPerftestTestDefinition


class NixlPerftestSlurmCommandGenStrategy(NIXLCmdGenBase):
    """Command generation strategy for NixlPerftest tests."""

    def __init__(self, system: SlurmSystem, test_run: TestRun) -> None:
        super().__init__(system, test_run)

    @property
    def matrix_gen_path(self) -> Path:
        return self.test_run.output_path / "matrices"

    @property
    def tdef(self) -> NixlPerftestTestDefinition:
        return cast(NixlPerftestTestDefinition, self.test_run.test.test_definition)

    def image_path(self) -> str | None:
        return str(self.tdef.docker_image.installed_path)

    def _container_mounts(self) -> list[str]:
        return []

    def _gen_srun_command(self) -> str:
        matrix_gen_command: list[str] = self.gen_matrix_gen_srun_command()
        etcd_command: list[str] = self.gen_etcd_srun_command(self.tdef.cmd_args.etcd_path)
        perftest_command: list[str] = self.gen_perftest_srun_command()
        return "\n".join(
            [
                "echo SLURM_JOB_MASTER_NODE=$SLURM_JOB_MASTER_NODE",
                " ".join(matrix_gen_command),
                " ".join(etcd_command),
                " ".join(self.gen_wait_for_etcd_command(self.tdef.cmd_args.wait_etcd_for)),
                " ".join(perftest_command),
            ]
        )

    def gen_matrix_gen_srun_command(self) -> list[str]:
        cmd = [
            *self.gen_srun_prefix(),
            "--ntasks-per-node=1",
            "--ntasks=1",
            "-N1",
            "bash",
            "-c",
            f'"{" ".join(self.gen_matrix_gen_command())}"',
        ]
        return cmd

    @staticmethod
    def prop_to_cli_arg(prop: str) -> str:
        return "--" + prop.replace("_", "-")

    def gen_matrix_gen_command(self) -> list[str]:
        cmd = [
            self.tdef.cmd_args.python_executable,
            self.tdef.cmd_args.matgen_script,
            "generate",
            "--num-user-requests=" + str(self.tdef.cmd_args.num_user_requests),
            "--batch-size=" + str(self.tdef.cmd_args.batch_size),
            "--num-prefill-nodes=" + str(self.tdef.cmd_args.num_prefill_nodes),
            "--num-decode-nodes=" + str(self.tdef.cmd_args.num_decode_nodes),
            "--results-dir=" + str(self.matrix_gen_path.absolute()),
            f"--prefill-tp={self.tdef.cmd_args.prefill_tp}",
            f"--prefill-pp={self.tdef.cmd_args.prefill_pp}",
            f"--prefill-cp={self.tdef.cmd_args.prefill_cp}",
            f"--decode-tp={self.tdef.cmd_args.decode_tp}",
            f"--decode-pp={self.tdef.cmd_args.decode_pp}",
            f"--decode-cp={self.tdef.cmd_args.decode_cp}",
        ]

        if self.tdef.cmd_args.model:
            cmd.append(f"--model={self.tdef.cmd_args.model}")
        else:
            args = ["hidden_size", "num_layers", "num_heads", "num_kv_heads", "dtype_size"]
            for arg in args:
                if getattr(self.tdef.cmd_args, arg) is not None:
                    cmd.append(f"{self.prop_to_cli_arg(arg)}={getattr(self.tdef.cmd_args, arg)}")

        opt_args = ["isl_mean", "isl_scale"]
        for arg in opt_args:
            if getattr(self.tdef.cmd_args, arg) is not None:
                cmd.append(f"{self.prop_to_cli_arg(arg)}={getattr(self.tdef.cmd_args, arg)}")

        if self.tdef.cmd_args.matgen_args:
            if self.system.ntasks_per_node is not None and self.tdef.cmd_args.matgen_args.ppn is None:
                self.tdef.cmd_args.matgen_args.ppn = self.system.ntasks_per_node

            for arg, value in self.tdef.cmd_args.matgen_args.model_dump().items():
                if value is None:
                    continue
                cmd.append(f"{self.prop_to_cli_arg(arg)}={value}")

        (self.matrix_gen_path).mkdir(parents=True, exist_ok=True)

        return cmd

    def gen_perftest_srun_command(self) -> list[str]:
        self.create_env_vars_file()

        cmd = [
            *self.gen_srun_prefix(),
            "--overlap",
            f'bash -c "source {(self.test_run.output_path / "env_vars.sh").absolute()}; ',
            self.tdef.cmd_args.python_executable,
            self.tdef.cmd_args.perftest_script,
            self.tdef.cmd_args.subtest,
            str(self.matrix_gen_path.absolute() / "metadata.yaml"),
            "--json-output-path=" + str(self.test_run.output_path.absolute() / "results.json"),
            '"',
        ]
        return cmd
