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

from typing import cast

from cloudai.systems.slurm import SlurmCommandGenStrategy

from .nixl_kvbench import NIXLKVBenchTestDefinition


class NIXLKVBenchSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for NIXLKVBench tests."""

    def _container_mounts(self) -> list[str]:
        return []

    @property
    def tdef(self) -> NIXLKVBenchTestDefinition:
        return cast(NIXLKVBenchTestDefinition, self.test_run.test.test_definition)

    @property
    def final_env_vars(self) -> dict[str, str | list[str]]:
        env_vars = super().final_env_vars
        env_vars["NIXL_ETCD_NAMESPACE"] = "/nixl/kvbench/$(uuidgen)"
        env_vars["NIXL_ETCD_ENDPOINTS"] = '"$SLURM_JOB_MASTER_NODE:2379"'
        return env_vars

    @final_env_vars.setter
    def final_env_vars(self, value: dict[str, str | list[str]]) -> None:
        super().final_env_vars = value

    def image_path(self) -> str | None:
        return str(self.tdef.docker_image.installed_path)

    def _gen_srun_command(self) -> str:
        etcd_command: list[str] = self.gen_etcd_srun_command()
        kvbench_command: list[str] = self.gen_kvbench_srun_command()

        final_cmd: list[str] = []
        if self.tdef.cmd_args.with_etcd:
            final_cmd.append(" ".join(etcd_command))
            final_cmd.append(" ".join(self.gen_wait_for_etcd_command()))

        final_cmd.append(" ".join(kvbench_command))
        return "\n".join(final_cmd)

    def gen_kvbench_command(self) -> list[str]:
        command: list[str] = [
            f"{self.tdef.cmd_args.python_executable}",
            f"{self.tdef.cmd_args.kvbench_script}",
            self.tdef.cmd_args.command,
        ]
        for k, v in self.test_run.test.test_definition.cmd_args_dict.items():
            command.append(f"--{k} {v}")

        if self.tdef.cmd_args.with_etcd:
            command.append("--etcd-endpoints http://$NIXL_ETCD_ENDPOINTS:2379")

        return command

    def gen_kvbench_srun_command(self) -> list[str]:
        with (self.test_run.output_path / "env_vars.sh").open("w") as f:
            for key, value in self.final_env_vars.items():
                if key in {"NIXL_ETCD_ENDPOINTS", "NIXL_ETCD_NAMESPACE"}:
                    continue
                f.write(f"export {key}={value}\n")

        cmd = [
            *self.gen_srun_prefix(),
            "--overlap",
            f'bash -c "source {(self.test_run.output_path / "env_vars.sh").absolute()}; ',
            *self.gen_kvbench_command(),
            '"',
        ]
        return cmd

    def gen_etcd_srun_command(self) -> list[str]:
        etcd_cmd = [
            self.tdef.cmd_args.etcd_path,
            "--listen-client-urls=http://0.0.0.0:2379",
            "--advertise-client-urls=http://$SLURM_JOB_MASTER_NODE:2379",
            "--listen-peer-urls=http://0.0.0.0:2380",
            "--initial-advertise-peer-urls=http://$SLURM_JOB_MASTER_NODE:2380",
            '--initial-cluster="default=http://$SLURM_JOB_MASTER_NODE:2380"',
            "--initial-cluster-state=new",
        ]
        cmd = [
            *self.gen_srun_prefix(),
            f"--output={self.test_run.output_path.absolute() / 'etcd.log'}",
            "--overlap",
            "--ntasks-per-node=1",
            "--ntasks=1",
            "--nodelist=$SLURM_JOB_MASTER_NODE",
            "-N1",
            *etcd_cmd,
            " &",
        ]
        return cmd

    def gen_wait_for_etcd_command(self) -> list[str]:
        cmd = [
            "timeout",
            str(self.tdef.cmd_args.wait_etcd_for),
            "bash",
            "-c",
            '"until curl -s $NIXL_ETCD_ENDPOINTS/health > /dev/null 2>&1; do sleep 1; done" || {\n',
            f'  echo "ETCD ($NIXL_ETCD_ENDPOINTS) was unreachable after {self.tdef.cmd_args.wait_etcd_for} seconds";\n',
            "  exit 1\n",
            "}",
        ]
        return cmd
