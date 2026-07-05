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
from __future__ import annotations

from cloudai.systems.slurm import SlurmCommandGenStrategy


class EtcdCmdGenMixin(SlurmCommandGenStrategy):
    """
    Per-test etcd lifecycle shared by workloads whose rendezvous needs an etcd server (NIXL, MoE Hybrid-EP).

    It provides the building blocks for a self-contained lifecycle inside a single test's command block:
    start etcd on the master node in the background, wait until it is healthy, (run the test), then kill it
    and wait for it to exit. Because the lifecycle lives in the test block rather than a job-level prologue,
    it behaves identically in multi-sbatch (one job per test) and single-sbatch (all tests in one job): every
    test case owns its etcd start/wait/stop.

    Relies on SLURM_JOB_MASTER_NODE (exported by the runner in both modes) and NIXL_ETCD_ENDPOINTS (exported
    from the strategy's final_env_vars).
    """

    def gen_etcd_srun_command(self, etcd_path: str = "etcd") -> list[str]:
        etcd_cmd = [
            etcd_path,
            "--listen-client-urls=http://0.0.0.0:2379",
            "--advertise-client-urls=http://$SLURM_JOB_MASTER_NODE:2379",
            "--listen-peer-urls=http://0.0.0.0:2380",
            "--initial-advertise-peer-urls=http://$SLURM_JOB_MASTER_NODE:2380",
            '--initial-cluster="default=http://$SLURM_JOB_MASTER_NODE:2380"',
            "--initial-cluster-state=new",
        ]
        return [
            *self.gen_srun_prefix(with_num_nodes=False),
            f"--output={self.test_run.output_path.absolute() / 'etcd.log'}",
            "--overlap",
            "--ntasks-per-node=1",
            "--ntasks=1",
            "--nodelist=$SLURM_JOB_MASTER_NODE",
            "-N1",
            *etcd_cmd,
            " &",
        ]

    def gen_wait_for_etcd_command(self, timeout: int = 60) -> list[str]:
        return [
            "timeout",
            str(timeout),
            "bash",
            "-c",
            '"until curl -s $NIXL_ETCD_ENDPOINTS/health > /dev/null 2>&1; do sleep 1; done" || {\n',
            f'  echo "ETCD ($NIXL_ETCD_ENDPOINTS) was unreachable after {timeout} seconds";\n',
            "  exit 1\n",
            "}",
        ]

    def gen_kill_and_wait_cmd(self, pid_var: str, timeout: int = 60) -> list[str]:
        return [
            f"kill -TERM ${pid_var}\n",
            "timeout",
            str(timeout),
            "bash",
            "-c",
            f'"while kill -0 ${pid_var} 2>/dev/null; do sleep 1; done" || {{\n',
            f'  echo "Failed to kill ETCD (pid=${pid_var}) within {timeout} seconds";\n',
            "  exit 1\n",
            "}",
        ]
