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

import logging
from pathlib import Path
from typing import List, cast

from pydantic import BaseModel, TypeAdapter, ValidationError

from cloudai.core import DockerImage, File, GitRepo
from cloudai.systems.slurm import SlurmCommandGenStrategy

from .ai_dynamo import AIDynamoTestDefinition


class AIDynamoSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for AI Dynamo on Slurm systems."""

    def _container_mounts(self) -> list[str]:
        td = cast(AIDynamoTestDefinition, self.test_run.test)

        result = [f"{self.system.hf_home_path.absolute()}:{self.CONTAINER_MOUNT_HF_HOME}"]

        logging.info(f"storage_cache_dir: {td.cmd_args.storage_cache_dir}")
        if td.cmd_args.storage_cache_dir:
            result.append(f"{td.cmd_args.storage_cache_dir}:{td.cmd_args.storage_cache_dir}")

        return result

    def image_path(self) -> str | None:
        tdef: AIDynamoTestDefinition = cast(AIDynamoTestDefinition, self.test_run.test)
        if tdef.docker_image and tdef.docker_image.installed_path:
            return str(tdef.docker_image.installed_path)
        return None

    def _get_toml_args(self, base_model: BaseModel, prefix: str, exclude: List[str] | None = None) -> List[str]:
        args = []
        exclude = exclude or []
        git_repo_adapter = TypeAdapter(GitRepo)
        file_adapter = TypeAdapter(File)
        toml_args = base_model.model_dump(by_alias=True, exclude=set(exclude), exclude_none=True)
        for k, v in toml_args.items():
            if isinstance(v, dict):
                try:
                    repo = git_repo_adapter.validate_python(v)
                    if repo.installed_path:
                        args.extend([f'{prefix}{k} "{self.CONTAINER_MOUNT_INSTALL}/{repo.repo_name}"'])
                    continue
                except ValidationError:
                    pass
                try:
                    file_obj = file_adapter.validate_python(v)
                    if file_obj.installed_path:
                        args.extend([f'{prefix}{k} "{self.CONTAINER_MOUNT_INSTALL}/{file_obj.src.name}"'])
                    continue
                except ValidationError:
                    pass
            str_v = str(v)
            if str_v.startswith("{") and str_v.endswith("}"):
                args.append(f"{prefix}{k} '{str_v}'")
            else:
                args.append(f'{prefix}{k} "{v}"')

        return args

    def _get_nested_toml_args(self, base_model: BaseModel, prefix: str) -> List[str]:
        result = self._get_toml_args(base_model, prefix, exclude=["args"])

        if (nested_args := getattr(base_model, "args", None)) is not None:
            result.extend(self._get_toml_args(nested_args, prefix + "args-"))

        return result

    def _gen_script_args(self, td: AIDynamoTestDefinition, wait_for_external_workload: bool = False) -> List[str]:
        assert td.repo.installed_path
        args = [
            "--user $USER",
            f"--install-dir {self.CONTAINER_MOUNT_INSTALL}",
            f"--results-dir {self.CONTAINER_MOUNT_OUTPUT}",
            f"--dynamo-repo {self.CONTAINER_MOUNT_INSTALL}/{td.repo.repo_name}",
            f"--hf-home {self.CONTAINER_MOUNT_HF_HOME}",
            f"--workloads {td.cmd_args.workloads}",
            f"--failure-marker {self.CONTAINER_MOUNT_OUTPUT}/{td.failure_marker}",
            f"--success-marker {self.CONTAINER_MOUNT_OUTPUT}/{td.success_marker}",
        ]

        if wait_for_external_workload:
            args.append("--wait-for-external-workload true")

        if td.cmd_args.storage_cache_dir:
            args.append(f"--storage-cache-dir {td.cmd_args.storage_cache_dir}")

        args.extend(
            self._get_toml_args(
                td.cmd_args.dynamo,
                "--dynamo-",
                exclude=[
                    "prefill_worker",
                    "decode_worker",
                ],
            )
        )

        if td.cmd_args.dynamo.prefill_worker:
            args.extend(self._get_nested_toml_args(td.cmd_args.dynamo.prefill_worker, "--prefill-"))
        args.extend(self._get_nested_toml_args(td.cmd_args.dynamo.decode_worker, "--decode-"))

        args.extend(self._get_nested_toml_args(td.cmd_args.lmcache, "--lmcache-"))
        args.extend(self._get_nested_toml_args(td.cmd_args.genai_perf, "--genai_perf-"))

        return args

    def _client_image_path(self, td: AIDynamoTestDefinition) -> str | None:
        client_image_url = td.cmd_args.genai_perf.client_docker_image_url
        if not client_image_url:
            return None
        client_image = DockerImage(url=client_image_url)
        if client_image.installed_path:
            return str(client_image.installed_path)
        return None

    def _gen_srun_command(self) -> str:
        td = cast(AIDynamoTestDefinition, self.test_run.test)
        num_nodes, node_list = self.get_cached_nodes_spec()

        out_dir = str(self.test_run.output_path.absolute())

        srun_cmd = self.gen_srun_prefix()
        srun_cmd.extend(
            [
                f"--nodes={num_nodes}",
                *([] if not node_list else [f"--nodelist={','.join(node_list)}"]),
                f"--ntasks={num_nodes}",
                "--ntasks-per-node=1",
                f"--output={out_dir}/node-%n-stdout.txt",
                f"--error={out_dir}/node-%n-stderr.txt",
                "bash",
                f"{self.CONTAINER_MOUNT_INSTALL}/{td.script.src.name}",
            ]
        )
        srun_cmd.extend(self._gen_script_args(td))
        return " \\\n  ".join(srun_cmd) + "\n"

    def _gen_service_srun_command(self) -> str:
        td = cast(AIDynamoTestDefinition, self.test_run.test)
        num_nodes, node_list = self.get_cached_nodes_spec()
        out_dir = str(self.test_run.output_path.absolute())

        srun_cmd = self.gen_srun_prefix()
        srun_cmd.extend(
            [
                f"--nodes={num_nodes}",
                *([] if not node_list else [f"--nodelist={','.join(node_list)}"]),
                f"--ntasks={num_nodes}",
                "--ntasks-per-node=1",
                f"--output={out_dir}/node-%n-stdout.txt",
                f"--error={out_dir}/node-%n-stderr.txt",
                "bash",
                f"{self.CONTAINER_MOUNT_INSTALL}/{td.script.src.name}",
            ]
        )
        srun_cmd.extend(self._gen_script_args(td, wait_for_external_workload=True))
        return " ".join(srun_cmd)

    def _client_container_mounts(self, td: AIDynamoTestDefinition) -> list[str]:
        mounts = [
            f"{self.test_run.output_path.absolute()}:{self.CONTAINER_MOUNT_OUTPUT}",
            f"{self.system.install_path.absolute()}:{self.CONTAINER_MOUNT_INSTALL}",
            f"{self.test_run.output_path.absolute()}",
            *td.extra_container_mounts,
            f"{self.system.hf_home_path.absolute()}:{self.CONTAINER_MOUNT_HF_HOME}",
        ]

        if td.cmd_args.storage_cache_dir:
            mounts.append(f"{td.cmd_args.storage_cache_dir}:{td.cmd_args.storage_cache_dir}")

        return mounts

    def _gen_client_srun_prefix(self, image_path: str) -> list[str]:
        srun_cmd = ["srun", "--export=ALL", f"--mpi={self.system.mpi}"]
        mounts = self._client_container_mounts(cast(AIDynamoTestDefinition, self.test_run.test))
        srun_cmd.append(f"--container-image={image_path}")
        srun_cmd.append(f"--container-mounts={','.join(mounts)}")
        if not self.system.container_mount_home:
            srun_cmd.append("--no-container-mount-home")
        if self.system.extra_srun_args:
            srun_cmd.append(self.system.extra_srun_args)
        if self.test_run.extra_srun_args:
            srun_cmd.append(self.test_run.extra_srun_args)
        return srun_cmd

    def _gen_external_benchmark_command(self, image_path: str, frontend_node: str) -> str:
        td = cast(AIDynamoTestDefinition, self.test_run.test)
        out_dir = str(self.test_run.output_path.absolute())

        srun_cmd = self._gen_client_srun_prefix(image_path)
        srun_cmd.extend(
            [
                "--nodes=1",
                f"--nodelist={frontend_node}",
                "--ntasks=1",
                "--ntasks-per-node=1",
                f"--output={out_dir}/genai-perf-stdout.txt",
                f"--error={out_dir}/genai-perf-stderr.txt",
                "bash",
                f"{self.CONTAINER_MOUNT_INSTALL}/{td.cmd_args.genai_perf.script.src.name}",
                f"--result-dir {self.CONTAINER_MOUNT_OUTPUT}",
                f'--model "{td.cmd_args.dynamo.model}"',
                f'--url "http://{frontend_node}"',
                f'--port "{td.cmd_args.dynamo.port}"',
                f'--endpoint "{td.cmd_args.dynamo.endpoint}"',
                f'--gpus-per-node "{self.system.gpus_per_node or 1}"',
                f'--report-name "{td.cmd_args.genai_perf.report_name}"',
                f'--cmd "{td.cmd_args.genai_perf.cmd}"',
            ]
        )
        if td.cmd_args.genai_perf.extra_args:
            extra_args = td.cmd_args.genai_perf.extra_args
            if isinstance(extra_args, list):
                extra_args = " ".join(str(arg) for arg in extra_args)
            srun_cmd.append(f'--extra-args "{extra_args}"')

        srun_cmd.append("--")
        srun_cmd.extend(self._get_toml_args(td.cmd_args.genai_perf.args, "--"))

        return " ".join(srun_cmd)

    def gen_exec_command(self) -> str:
        td = cast(AIDynamoTestDefinition, self.test_run.test)
        client_image_path = self._client_image_path(td)
        if not client_image_path:
            return super().gen_exec_command()

        service_cmd = self._gen_service_srun_command()
        benchmark_cmd = self._gen_external_benchmark_command(client_image_path, "$FRONTEND_NODE")
        success_marker = f"{self.test_run.output_path.absolute()}/{td.success_marker}"

        full_command = "\n".join(
            [
                'FRONTEND_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)',
                'if [ -z "$FRONTEND_NODE" ]; then',
                '  echo "Failed to resolve frontend node from SLURM_JOB_NODELIST" >&2',
                "  exit 1",
                "fi",
                f"{service_cmd} &",
                "SERVICE_PID=$!",
                "BENCH_RC=0",
                "for _ in $(seq 1 120); do",
                f'  if curl -sf "http://$FRONTEND_NODE:{td.cmd_args.dynamo.port}/v1/models" >/dev/null 2>&1; then',
                "    break",
                "  fi",
                "  sleep 5",
                "done",
                f'if ! curl -sf "http://$FRONTEND_NODE:{td.cmd_args.dynamo.port}/v1/models" >/dev/null 2>&1; then',
                "  BENCH_RC=1",
                "else",
                f"  {benchmark_cmd} || BENCH_RC=$?",
                "fi",
                f'touch "{success_marker}"',
                "wait $SERVICE_PID || true",
                "exit $BENCH_RC",
            ]
        )

        return self._write_sbatch_script(full_command)

    def _validate_worker_nodes(
        self, node_list: list[str], worker_nodes: str | None, num_nodes: int, worker_type: str
    ) -> None:
        """Validate node list for a specific worker type."""
        if not worker_nodes:
            return

        worker_node_list = worker_nodes.split(",")
        if len(worker_node_list) != num_nodes:
            raise ValueError(
                f"Number of {worker_type} nodes ({len(worker_node_list)}) does not match num_nodes ({num_nodes})"
            )
        if not all(node in node_list for node in worker_node_list):
            raise ValueError(f"Some {worker_type} nodes are not in the allocated node list")

    def _validate_node_overlap(self, prefill_nodes: str, decode_nodes: str) -> None:
        """Validate that there is no overlap between prefill and decode nodes."""
        prefill_set = set(prefill_nodes.split(","))
        decode_set = set(decode_nodes.split(","))
        if prefill_set & decode_set:
            raise ValueError("Overlap found between prefill and decode node lists")

    def get_cached_nodes_spec(self) -> tuple[int, list[str]]:
        cache_key = ":".join(
            [
                self.test_run.name,
                str(self.test_run.current_iteration),
                str(self.test_run.step),
                str(self.test_run.num_nodes),
                ",".join(self.test_run.nodes),
            ]
        )

        if cache_key in self._node_spec_cache:
            return self._node_spec_cache[cache_key]

        td = cast(AIDynamoTestDefinition, self.test_run.test)
        prefill_n, prefill_nodes = 0, ""
        if td.cmd_args.dynamo.prefill_worker:
            prefill_n = cast(int, td.cmd_args.dynamo.prefill_worker.num_nodes)
            prefill_nodes = td.cmd_args.dynamo.prefill_worker.nodes
        decode_n = td.cmd_args.dynamo.decode_worker.num_nodes
        decode_nodes = td.cmd_args.dynamo.decode_worker.nodes

        assert isinstance(prefill_n, int), "prefill_worker.num_nodes must be an integer"
        assert isinstance(decode_n, int), "decode_worker.num_nodes must be an integer"

        if prefill_nodes and decode_nodes:
            self.test_run.nodes = prefill_nodes.split(",") + decode_nodes.split(",") + self.test_run.nodes
            self.test_run.num_nodes = len(self.test_run.nodes)
            prefill_n = len(prefill_nodes.split(","))
            decode_n = len(decode_nodes.split(","))
        else:
            self.test_run.num_nodes = prefill_n + decode_n

        total_nodes = prefill_n + decode_n

        logging.info("Setting num_nodes from %d to %d", self.test_run.num_nodes, total_nodes)

        self.test_run.num_nodes = total_nodes

        requested_nodes, node_list = self.system.get_nodes_by_spec(self.test_run.nnodes, self.test_run.nodes)

        if prefill_nodes or decode_nodes:
            self._validate_worker_nodes(node_list, prefill_nodes, prefill_n, "prefill")
            self._validate_worker_nodes(node_list, decode_nodes, decode_n, "decode")
            if prefill_nodes and decode_nodes:
                self._validate_node_overlap(prefill_nodes, decode_nodes)

        if total_nodes > requested_nodes:
            raise ValueError(
                f"Not enough nodes requested: need {total_nodes} total nodes "
                f"({prefill_n} prefill + {decode_n} decode), "
                f"but only got {requested_nodes}"
            )

        result = (total_nodes, node_list)
        self._node_spec_cache[cache_key] = result
        return result

    def gen_dynamo_cmd(self, module: str, config: Path) -> str:
        """
        Generate the dynamo command for serving a module with a config.

        Args:
            module: The module to serve.
            config: The path to the config file.

        Returns:
            The dynamo command string.
        """
        return f"dynamo serve {module} -f {config}"
