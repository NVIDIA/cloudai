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

from pathlib import Path, PurePosixPath
from typing import ClassVar, List, cast

from cloudai.systems.slurm import SlurmCommandGenStrategy

from .moe_benchmark import MoEBenchmarkCmdArgs, MoEBenchmarkTestDefinition


class MoEBenchmarkSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for the custom MoE benchmark on Slurm systems."""

    # Per-backend env overrides. The backends are chained in ONE srun and share its
    # env, but each EP lib needs different NCCL/UCX knobs, so we emit these as inline
    # `VAR=val` assignments before that backend's `python` — they apply to ONLY that
    # process. v1/v2/uccl_ep need nothing extra (they use the shared env). Mirrors
    # csrc/gaia-run-allbackends.sh in the dp-benchmark repo.
    #   * nixl_ep: GID=auto + GIN(type 3) + NCCL net-plugin off (so deep_ep's
    #     duplicate-NCCL check doesn't trip when it loads the hpcx plugin).
    #   * nccl_ep: same + EP_SKIP_DEEPEP_PREIMPORT=1 (don't import deep_ep) and
    #     prepend the ISOLATED v0.1.0 NCCL prefix baked into the unified image, so
    #     libnccl_ep/nccl.ep resolve it without colliding with deepep's NCCL.
    _BACKEND_ENV: ClassVar[dict[str, list[str]]] = {
        "nixl_ep": [
            "NCCL_IB_GID_INDEX=auto",
            "NCCL_NET_PLUGIN=none",
            "NCCL_IB_QPS_PER_CONNECTION=4",
            "NCCL_IB_SPLIT_DATA_ON_QPS=0",
            "NCCL_IBEXT_DISABLE=0",
            "OMPI_MCA_btl=tcp,self",
            "EP_SUPPRESS_NCCL_CHECK=1",
            "NCCL_GIN_TYPE=3",
        ],
        "nccl_ep": [
            "NCCL_IB_GID_INDEX=auto",
            "NCCL_NET_PLUGIN=none",
            "NCCL_IB_QPS_PER_CONNECTION=4",
            "NCCL_IB_SPLIT_DATA_ON_QPS=0",
            "NCCL_IBEXT_DISABLE=0",
            "OMPI_MCA_btl=tcp,self",
            "NCCL_GIN_TYPE=3",
            "EP_SKIP_DEEPEP_PREIMPORT=1",
            "LD_LIBRARY_PATH=/opt/nccl-ep-build/lib:$LD_LIBRARY_PATH",
        ],
        # Hybrid-EP: NIXL transport (etcd rendezvous started in the sbatch script,
        # NIXL_ETCD_ENDPOINTS exported globally there). Its kernels are JIT-compiled
        # at runtime, so nvcc needs the UCX(device API)+NIXL includes on CPATH.
        "deepep_hybrid": [
            "NCCL_IB_GID_INDEX=auto",
            "NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API=64",
            "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=64",
            "NUM_OF_TOKENS_PER_CHUNK_PREPROCESSING_API=64",
            "CPATH=/usr/local/ucx-nixl/include:/usr/local/nixl/include:/usr/local/nixl/include/gpu/ucx:$CPATH",
            "CPLUS_INCLUDE_PATH=/usr/local/ucx-nixl/include:/usr/local/nixl/include:/usr/local/nixl/include/gpu/ucx:$CPATH",
        ],
    }

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
        # Hybrid-EP's NIXL path needs an etcd server reachable by all ranks. If a
        # hybrid backend is in the run, start etcd on the head node (background, in
        # the unified container) and export NIXL_ETCD_ENDPOINTS for every step
        # (only hybrid reads it). Slurm tears down this step when the job ends.
        tdef: MoEBenchmarkTestDefinition = cast(MoEBenchmarkTestDefinition, self.test_run.test)
        versions = tdef.cmd_args.deepep_versions or []
        if any(v in ("deepep_hybrid", "hybrid", "hybrid_ep") for v in versions):
            image = self.image_path()
            batch_script_content.extend(
                [
                    "# Hybrid-EP NIXL rendezvous: etcd on the head node (background).",
                    f'srun --overlap --nodes=1 --ntasks=1 -w "$head_node" --container-image={image} '
                    'bash -c "etcd --log-level error --listen-client-urls http://0.0.0.0:2379 '
                    '--advertise-client-urls http://$head_node_ip:2379 --data-dir /tmp/etcd-cloudai-$$" &',
                    "export NIXL_ETCD_ENDPOINTS=http://$head_node_ip:2379",
                    "sleep 8",
                    "",
                ]
            )

    def _append_sbatch_directives(self, batch_script_content: List[str]) -> None:
        super()._append_sbatch_directives(batch_script_content)
        self._append_head_node_detection(batch_script_content)

    def _container_mounts(self) -> List[str]:
        """Return container mounts specific to the MoE benchmark."""
        tdef: MoEBenchmarkTestDefinition = cast(MoEBenchmarkTestDefinition, self.test_run.test)
        cmd_args: MoEBenchmarkCmdArgs = tdef.cmd_args

        config_file_path = self.test_run.output_path / "config.yaml"
        self._generate_config_yaml(config_file_path)

        return [
            f"{config_file_path.absolute()}:{cmd_args.config_file_path}",
            f"{self.test_run.output_path.absolute()}:{cmd_args.results_dir}",
        ]

    def image_path(self) -> str | None:
        tdef: MoEBenchmarkTestDefinition = cast(MoEBenchmarkTestDefinition, self.test_run.test)
        return str(tdef.docker_image.installed_path)

    def generate_test_command(self) -> List[str]:
        tdef: MoEBenchmarkTestDefinition = cast(MoEBenchmarkTestDefinition, self.test_run.test)
        cmd_args: MoEBenchmarkCmdArgs = tdef.cmd_args

        benchmark_script = str(PurePosixPath(cmd_args.benchmark_root) / "benchmark.py")

        # standard AND low_latency chain one process per backend (each backend is the
        # 3rd CLI arg -> benchmark.py runs only that one; backends can't coexist in a
        # process). Other modes fall back to a single call.
        if cmd_args.mode not in ("standard", "low_latency"):
            return ["python", benchmark_script, cmd_args.config_file_path]

        versions = cmd_args.deepep_versions or ["legacy"]
        parts: List[str] = []
        for version in versions:
            if parts:
                parts.append("&&")
            # Inline per-backend env (applies to this backend's process only).
            parts.extend(self._BACKEND_ENV.get(version, []))
            parts.extend(["python", benchmark_script, cmd_args.config_file_path, version])
        return parts

    def _generate_config_yaml(self, config_path: Path) -> None:
        tdef: MoEBenchmarkTestDefinition = cast(MoEBenchmarkTestDefinition, self.test_run.test)

        config_lines = ["# MoE Benchmark Configuration", "# Generated by CloudAI", ""]
        config_lines.append(f'benchmark_type: "{tdef.cmd_args.mode}"')
        for key, value in tdef.cmd_args_dict.items():
            if isinstance(value, bool):
                config_lines.append(f"{key}: {str(value).lower()}")
            elif isinstance(value, str):
                config_lines.append(f'{key}: "{value}"')
            else:
                config_lines.append(f"{key}: {value}")

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            f.write("\n".join(config_lines))

    def gen_srun_success_check(self) -> str:
        output_file = self.test_run.output_path / "stdout.txt"
        return (
            'grep -Eq "global_bw|RDMA BW \\(GB/s\\)|NVLink BW \\(GB/s\\)|Bus BW \\(GB/s\\)|Global BW \\(GB/s\\)" '
            f"{output_file} && echo 1 || echo 0"
        )
