# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import PurePosixPath
from typing import Any, List, cast

from cloudai.systems.slurm import SlurmCommandGenStrategy

from .deepep import DeepEPCmdArgs, DeepEPTestDefinition

_LEGACY_SUBTESTS = {"test_internode", "test_intranode", "test_low_latency"}

_CLI_FIELDS_BY_SUBTEST = {
    "test_internode": (
        "num_processes",
        "num_tokens",
        "hidden",
        "num_topk_groups",
        "num_topk",
        "pressure_test_mode",
        "num_experts",
        "test_ll_compatibility",
    ),
    "test_intranode": (
        "num_processes",
        "num_tokens",
        "hidden",
        "num_topk",
        "num_experts",
        "allow_mnnvl",
    ),
    "test_low_latency": (
        "num_processes",
        "num_tokens",
        "hidden",
        "num_topk",
        "num_experts",
        "allow_mnnvl",
        "disable_nvlink",
        "use_logfmt",
        "pressure_test",
        "shrink_test",
    ),
    "test_ep": (
        "num_processes",
        "num_sms",
        "num_qps",
        "num_allocated_qps",
        "num_gpu_timeout_secs",
        "num_cpu_timeout_secs",
        "sl_idx",
        "num_tokens",
        "hidden",
        "num_topk",
        "num_experts",
        "do_cpu_sync",
        "allow_hybrid_mode",
        "allow_multiple_reduction",
        "prefer_overlap_with_compute",
        "deterministic",
        "seed",
        "skip_check",
        "skip_perf_test",
        "do_pressure_test",
        "reuse_elastic_buffer",
        "test_first_only",
        "unbalanced_ratio",
        "precise_unbalanced_ratio",
        "masked_ratio",
        "dump_profile_traces",
        "ignore_local_traffic",
    ),
}


def _flag_name(field_name: str) -> str:
    return f"--{field_name.replace('_', '-')}"


class DeepEPSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for official DeepEP v1/v2 tests."""

    @property
    def tdef(self) -> DeepEPTestDefinition:
        return cast(DeepEPTestDefinition, self.test_run.test)

    def _append_head_node_detection(self, batch_script_content: List[str]) -> None:
        batch_script_content.extend(
            [
                "",
                "nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )",
                "nodes_array=($nodes)",
                "head_node=${nodes_array[0]}",
                'head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)',
                "",
                "echo Nodes: $SLURM_JOB_NODELIST",
                "echo Num Nodes: ${#nodes[@]}",
                "echo Head Node IP: $head_node_ip",
                "",
                "export MASTER_ADDR=$head_node_ip",
                "export MASTER_PORT=29500",
                "",
            ]
        )

    def _append_sbatch_directives(self, batch_script_content: List[str]) -> None:
        num_nodes, node_list = self.get_cached_nodes_spec()

        self._add_reservation(batch_script_content)
        batch_script_content.append(f"#SBATCH --output={self.test_run.output_path.absolute() / 'stdout.txt'}")
        batch_script_content.append(f"#SBATCH --error={self.test_run.output_path.absolute() / 'stderr.txt'}")
        batch_script_content.append(f"#SBATCH --partition={self.system.default_partition}")
        if self.system.account:
            batch_script_content.append(f"#SBATCH --account={self.system.account}")
        if node_list:
            batch_script_content.append(f"#SBATCH --nodelist={','.join(node_list)}")
        batch_script_content.append(f"#SBATCH -N {num_nodes}")
        batch_script_content.append(f"#SBATCH --gpus-per-node={self.system.gpus_per_node}")
        batch_script_content.append(f"#SBATCH --gres=gpu:{self.system.gpus_per_node}")
        batch_script_content.append("#SBATCH --ntasks-per-node=1")
        if self.test_run.time_limit:
            batch_script_content.append(f"#SBATCH --time={self.test_run.time_limit}")
        batch_script_content.append(
            "\nexport SLURM_JOB_MASTER_NODE=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)"
        )
        self._append_head_node_detection(batch_script_content)

    def _gen_srun_command(self) -> str:
        srun_command_parts = self.gen_srun_prefix(use_pretest_extras=True, with_num_nodes=False)
        num_nodes, _ = self.get_cached_nodes_spec()
        srun_command_parts.extend([f"--nodes={num_nodes}", f"--ntasks={num_nodes}", "--ntasks-per-node=1"])

        nsys_command_parts = self.gen_nsys_command()
        test_command_parts = self.generate_test_command()

        with (self.test_run.output_path / "env_vars.sh").open("w") as f:
            for key, value in self.final_env_vars.items():
                f.write(f'export {key}="{value}"\n')

        full_test_cmd = (
            f'bash -c "source {(self.test_run.output_path / "env_vars.sh").absolute()}; '
            + " ".join(nsys_command_parts + test_command_parts)
            + '"'
        )

        return " ".join(srun_command_parts) + " " + full_test_cmd

    def _container_mounts(self) -> list[str]:
        return []

    def image_path(self) -> str | None:
        return str(self.tdef.docker_image.installed_path)

    def _script_path(self, cmd_args: DeepEPCmdArgs) -> str:
        deep_ep_root = PurePosixPath(cmd_args.deep_ep_root)
        if cmd_args.subtest_name in _LEGACY_SUBTESTS:
            if cmd_args.legacy_tests_root:
                tests_root = PurePosixPath(cmd_args.legacy_tests_root)
            else:
                tests_root = deep_ep_root / "tests" / "legacy"
        elif cmd_args.elastic_tests_root:
            tests_root = PurePosixPath(cmd_args.elastic_tests_root)
        else:
            tests_root = deep_ep_root / "tests" / "elastic"

        return str(tests_root / f"{cmd_args.subtest_name}.py")

    def _append_cli_field(self, parts: list[str], field_name: str, value: Any) -> None:
        if value is None or value == "":
            return

        flag = _flag_name(field_name)
        if isinstance(value, bool):
            if value:
                parts.append(flag)
            return

        parts.extend([flag, str(value)])

    def generate_test_command(self) -> List[str]:
        cmd_args = self.tdef.cmd_args
        num_nodes, _ = self.get_cached_nodes_spec()
        parts: list[str] = [
            "torchrun",
            f"--nnodes={num_nodes}",
            "--nproc_per_node=1",
            "--rdzv_id=\\${SLURM_JOB_ID:-0}",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=\\${MASTER_ADDR}:\\${MASTER_PORT}",
            self._script_path(cmd_args),
        ]

        for field_name in _CLI_FIELDS_BY_SUBTEST[cmd_args.subtest_name]:
            self._append_cli_field(parts, field_name, getattr(cmd_args, field_name))

        if self.test_run.test.extra_cmd_args:
            parts.append(self.test_run.test.extra_args_str)

        return parts

    def gen_srun_success_check(self) -> str:
        output_file = self.test_run.output_path / "stdout.txt"
        return f'grep -Eq "\\[testing\\]|dispatch|combine|passed|tuning|Best" {output_file} && echo 1 || echo 0'
