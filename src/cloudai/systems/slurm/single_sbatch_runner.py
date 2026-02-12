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

import copy
import logging
import time
from datetime import timedelta
from pathlib import Path
from typing import Generator, Optional, cast

from cloudai.configurator.cloudai_gym import CloudAIGymEnv
from cloudai.core import JobIdRetrievalError, System, TestRun, TestScenario
from cloudai.util import CommandShell, format_time_limit, parse_time_limit

from .slurm_command_gen_strategy import SlurmCommandGenStrategy
from .slurm_metadata import SlurmJobMetadata, SlurmStepMetadata
from .slurm_runner import SlurmJob, SlurmRunner
from .slurm_system import SlurmSystem


class SingleSbatchRunner(SlurmRunner):
    """Slurm runner that puts all tests from a scenario into a single sbatch job."""

    def __init__(self, mode: str, system: System, test_scenario: TestScenario, output_path: Path) -> None:
        super().__init__(mode, system, test_scenario, output_path)
        self.cmd_shell = CommandShell()
        self.system = cast(SlurmSystem, system)
        self.job_name = "cloudai-single-sbatch"

    def get_sbatch_directives(self) -> list[str]:
        max_nodes, node_list = self.extract_sbatch_nodes_spec()

        batch_script_content: list[str] = [
            f"#SBATCH --nodelist={','.join(node_list)}" if node_list else f"#SBATCH -N {max_nodes}",
            f"#SBATCH --job-name={self.job_name}",
            f"#SBATCH --output={self.scenario_root.absolute() / 'common.out'}",
            f"#SBATCH --error={self.scenario_root.absolute() / 'common.err'}",
            f"#SBATCH --partition={self.system.default_partition}",
        ]
        time_limit = self.build_time_limit()
        if time_limit:
            batch_script_content.append(f"#SBATCH --time={time_limit}")
        if self.system.account:
            batch_script_content.append(f"#SBATCH --account={self.system.account}")
        if self.system.distribution:
            batch_script_content.append(f"#SBATCH --distribution={self.system.distribution}")
        if self.system.gpus_per_node and self.system.supports_gpu_directives:
            batch_script_content.append(f"#SBATCH --gpus-per-node={self.system.gpus_per_node}")
            batch_script_content.append(f"#SBATCH --gres=gpu:{self.system.gpus_per_node}")
        if self.system.ntasks_per_node:
            batch_script_content.append(f"#SBATCH --ntasks-per-node={self.system.ntasks_per_node}")

        for arg in self.system.extra_sbatch_args:
            batch_script_content.append(f"#SBATCH {arg}")

        return batch_script_content

    def build_time_limit(self) -> Optional[str]:
        total_trs, trs_with_time_limit = 0, 0
        total_time_limit: timedelta = timedelta(seconds=0)
        for tr in self.all_trs:
            total_trs += 1
            if tr.time_limit:
                trs_with_time_limit += 1
                total_time_limit += parse_time_limit(tr.time_limit)

        if trs_with_time_limit == 0:
            return None

        if trs_with_time_limit != 0 and trs_with_time_limit != total_trs:
            raise ValueError("All tests must have a time limit or none of them must have a time limit")

        return format_time_limit(total_time_limit)

    def extract_sbatch_nodes_spec(self) -> tuple[int, list[str]]:
        max_nodes = 1
        all_node_lists: list[str] = []
        for tr in self.all_trs:
            max_nodes = max(max_nodes, tr.nnodes)
            all_node_lists.extend(tr.nodes)

        _, node_list = self.system.get_nodes_by_spec(max_nodes, all_node_lists)
        if node_list:
            if max_nodes <= len(node_list):
                max_nodes = len(node_list)
            else:
                raise ValueError(
                    f"Number of nodes in the nodes list ({len(node_list)}) does not cover the max number "
                    f"of nodes ({max_nodes})"
                )

        return max_nodes, node_list

    def aux_commands(self) -> list[str]:
        tr = copy.deepcopy(next(self.all_trs))
        tr.output_path = self.scenario_root
        max_nodes, _ = self.extract_sbatch_nodes_spec()
        tr.num_nodes = max_nodes
        cmd_gen = cast(SlurmCommandGenStrategy, self.get_cmd_gen_strategy(self.system, tr))
        return [cmd_gen._metadata_cmd(), cmd_gen._ranks_mapping_cmd()]

    def get_single_tr_block(self, tr: TestRun) -> str:
        cmd_gen = cast(SlurmCommandGenStrategy, self.get_cmd_gen_strategy(self.system, tr))
        srun_cmd = cmd_gen.gen_srun_command()
        nnodes, node_list = self.system.get_nodes_by_spec(tr.nnodes, tr.nodes)
        node_arg = f"--nodelist={','.join(node_list)}" if node_list else f"-N{nnodes}"
        extra_args = (
            f"{node_arg} --output={tr.output_path.absolute()}/stdout.txt --error={tr.output_path.absolute()}/stderr.txt"
        )
        srun_cmd = srun_cmd.replace("srun", f"srun {extra_args}")

        return srun_cmd

    def unroll_dse(self, tr: TestRun) -> Generator[TestRun, None, None]:
        for idx, combination in enumerate(tr.all_combinations):
            next_tr = tr.apply_params_set(combination)
            next_tr.step = idx + 1
            next_tr.output_path = self.get_job_output_path(next_tr)

            if next_tr.test.constraint_check(next_tr):
                yield next_tr

    def get_global_env_vars(self) -> str:
        vars: list[str] = ["export SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)"]
        tr = self.test_scenario.test_runs[0]
        cmd_gen = cast(SlurmCommandGenStrategy, self.get_cmd_gen_strategy(self.system, tr))
        for key, value in cmd_gen.final_env_vars.items():
            vars.append(f"export {key}={value}")
        return "\n".join(vars)

    def gen_sbatch_content(self) -> str:
        content: list[str] = ["#!/bin/bash", *self.get_sbatch_directives(), ""]
        content.extend(self.aux_commands())
        content.append("")

        content.append(self.get_global_env_vars())
        content.append("")

        tr = self.test_scenario.test_runs[0]
        if tr.pre_test:
            content.append(self.add_pre_tests(tr.pre_test, tr))

        for tr in self.all_trs:
            self.on_job_submit(tr)
            content.append(self.get_single_tr_block(tr))
            content.append("")

        return "\n".join(content)

    def add_pre_tests(self, pre_tc: TestScenario, base_tr: TestRun) -> str:
        content = []
        cmd_gen = cast(SlurmCommandGenStrategy, self.get_cmd_gen_strategy(self.system, base_tr))
        content.append(cmd_gen.gen_pre_test(pre_tc, self.scenario_root))
        content.append("if [ $PRE_TEST_SUCCESS -ne 1 ]; then")
        content.append("    exit 1")
        content.append("fi")
        content.append("")
        return "\n".join(content)

    @property
    def all_trs(self) -> Generator[TestRun, None, None]:
        for tr in self.test_scenario.test_runs:
            if tr.is_dse_job:
                for _tr in self.unroll_dse(tr):
                    yield _tr
            else:
                tr.output_path = self.get_job_output_path(tr)
                yield tr

    def run(self):
        if self.shutting_down:
            return

        self.scenario_root.mkdir(parents=True, exist_ok=True)
        tr = self.test_scenario.test_runs[0]
        job = self._submit_test(tr)

        is_completed = False
        while not is_completed:
            if self.shutting_down:
                break
            is_completed = True if self.mode == "dry-run" else self.system.is_job_completed(job)
            time.sleep(self.system.monitor_interval)

        self.handle_dse()

        self.on_job_completion(job)

    def handle_dse(self):
        for tr in self.test_scenario.test_runs:
            if not tr.is_dse_job:
                continue

            for idx, combination in enumerate(tr.all_combinations, start=1):
                next_tr = tr.apply_params_set(combination)
                next_tr.step = idx
                next_tr.output_path = self.get_job_output_path(next_tr)

                gym = CloudAIGymEnv(next_tr, self)
                observation = gym.get_observation({})
                reward = gym.compute_reward(observation)
                gym.write_trajectory(idx, combination, reward, observation)

    def _submit_test(self, tr: TestRun) -> SlurmJob:
        with open(self.scenario_root / "cloudai_sbatch_script.sh", "w") as f:
            f.write(self.gen_sbatch_content())

        job_id = 0
        if self.mode == "run":
            exec_cmd = f"sbatch {self.scenario_root / 'cloudai_sbatch_script.sh'}"
            stdout, stderr = self.cmd_shell.execute(exec_cmd).communicate()
            job_id = self.get_job_id(stdout, stderr)
            if job_id is None:
                raise JobIdRetrievalError(
                    test_name=tr.name,
                    command=exec_cmd,
                    stdout=stdout,
                    stderr=stderr,
                    message="Failed to retrieve job ID.",
                )
        logging.info(f"Submitted slurm job: {job_id}")
        return SlurmJob(tr, id=job_id)

    def _get_job_metadata(
        self, job: SlurmJob, steps_metadata: list[SlurmStepMetadata]
    ) -> tuple[Path, SlurmJobMetadata]:
        return self.scenario_root / "slurm-job.toml", SlurmJobMetadata(
            job_id=int(job.id),
            name=steps_metadata[0].name,
            state=steps_metadata[0].state,
            exit_code=steps_metadata[0].exit_code,
            start_time=steps_metadata[0].start_time,
            end_time=steps_metadata[0].end_time,
            elapsed_time_sec=steps_metadata[0].elapsed_time_sec,
            job_steps=steps_metadata[1:],
            srun_cmd="n/a for single sbatch run",
            test_cmd="n/a for single sbatch run",
            is_single_sbatch=True,
            job_root=self.scenario_root.absolute(),
        )
