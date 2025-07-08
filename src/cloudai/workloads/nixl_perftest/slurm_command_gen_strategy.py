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
from cloudai.systems.slurm import SlurmCommandGenStrategy, SlurmSystem

from .nixl_perftest import NixlPerftestTestDefinition


class NixlPerftestSlurmCommandGenStrategy(SlurmCommandGenStrategy):
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
        return [str(self.test_run.output_path.absolute())]

    def _gen_srun_command(self) -> str:
        with (self.test_run.output_path / "env_vars.sh").open("w") as f:
            for key, value in self.final_env_vars.items():
                f.write(f"export {key}={value}\n")

        matrix_gen_command: list[str] = self.gen_matrix_gen_srun_command()
        etcd_command: list[str] = self.gen_etcd_srun_command()
        perftest_command: list[str] = self.gen_perftest_srun_command()
        return " ".join(matrix_gen_command) + "\n" + " ".join(etcd_command) + "\n" + " ".join(perftest_command)

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
        ]

        if self.tdef.cmd_args.model:
            cmd.append(f"--model={self.tdef.cmd_args.model}")
        else:
            args = ["hidden_size", "num_layers", "num_heads", "num_kv_heads", "dtype_size"]
            for arg in args:
                if getattr(self.tdef.cmd_args, arg) is not None:
                    cmd.append(f"{self.prop_to_cli_arg(arg)}={getattr(self.tdef.cmd_args, arg)}")

        opt_args = [
            "isl_mean",
            "isl_scale",
            "prefill_tp",
            "prefill_pp",
            "prefill_cp",
            "decode_tp",
            "decode_pp",
            "decode_cp",
        ]
        for arg in opt_args:
            if getattr(self.tdef.cmd_args, arg) is not None:
                cmd.append(f"{self.prop_to_cli_arg(arg)}={getattr(self.tdef.cmd_args, arg)}")

        (self.matrix_gen_path).mkdir(parents=True, exist_ok=True)

        return cmd

    def gen_etcd_srun_command(self) -> list[str]:
        etcd_cmd = [
            self.tdef.cmd_args.etcd_path,
            "--listen-client-urls",
            "http://0.0.0.0:2379",
            "--advertise-client-urls",
            "http://$(hostname -I | awk '{print $1}'):2379",
        ]
        cmd = [
            *self.gen_srun_prefix(),
            "--overlap",
            "--ntasks-per-node=1",
            "--ntasks=1",
            "--nodelist=$SLURM_JOB_MASTER_NODE",
            "-N1",
            "bash",
            "-c",
            f'"{" ".join(etcd_cmd)}" &',
        ]
        return cmd

    def gen_perftest_srun_command(self) -> list[str]:
        cmd = [
            *self.gen_srun_prefix(),
            "--overlap",
            "--ntasks-per-node=$SLURM_GPUS_PER_NODE",
            self.tdef.cmd_args.python_executable,
            self.tdef.cmd_args.perftest_script,
            self.tdef.cmd_args.subtest,
            str(self.matrix_gen_path.absolute() / "metadata.yaml"),
            "--json-output-path=" + str(self.test_run.output_path.absolute() / "results.json"),
        ]
        return cmd
