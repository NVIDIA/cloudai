# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any, Dict, List, Union, cast

from cloudai import TestRun
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy
from cloudai.workloads.nccl_test import NCCLTestDefinition


class NcclTestSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for NCCL tests on Slurm systems."""

    def _container_mounts(self, tr: TestRun) -> list[str]:
        mounts: list[str] = []
        env = tr.test.extra_env_vars | self.system.global_env_vars
        if "NCCL_TOPO_FILE" in env:
            nccl_topo_file = Path(env["NCCL_TOPO_FILE"]).resolve()  # pyright: ignore [reportArgumentType]
            mounts.append(f"{nccl_topo_file}:{nccl_topo_file}")
        return mounts

    def _parse_slurm_args(
        self,
        job_name_prefix: str,
        env_vars: Dict[str, Union[str, List[str]]],
        cmd_args: Dict[str, Union[str, List[str]]],
        tr: TestRun,
    ) -> Dict[str, Any]:
        base_args = super()._parse_slurm_args(job_name_prefix, env_vars, cmd_args, tr)

        tdef: NCCLTestDefinition = cast(NCCLTestDefinition, tr.test.test_definition)
        base_args.update({"image_path": tdef.docker_image.installed_path})

        return base_args

    def generate_test_command(
        self, env_vars: Dict[str, Union[str, List[str]]], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> List[str]:
        srun_command_parts = [f"{cmd_args['subtest_name']}"]
        nccl_test_args = [
            "nthreads",
            "ngpus",
            "minbytes",
            "maxbytes",
            "stepbytes",
            "op",
            "datatype",
            "root",
            "iters",
            "warmup_iters",
            "agg_iters",
            "average",
            "parallel_init",
            "check",
            "blocking",
            "cudagraph",
        ]
        for arg in nccl_test_args:
            if arg in cmd_args:
                srun_command_parts.append(f"--{arg} {cmd_args[arg]}")

        if tr.test.extra_cmd_args:
            srun_command_parts.append(tr.test.extra_cmd_args)

        return srun_command_parts

    def gen_srun_success_check(self, tr: TestRun) -> str:
        output_file = Path(tr.output_path) / "stdout.txt"
        return f'grep -q "Avg bus bandwidth" {output_file} && echo 1 || echo 0'
