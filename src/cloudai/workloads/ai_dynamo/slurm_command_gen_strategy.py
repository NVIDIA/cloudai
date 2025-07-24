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
from typing import List, cast
from textwrap import indent

import yaml

from cloudai.systems.slurm import SlurmCommandGenStrategy

from .ai_dynamo import AIDynamoTestDefinition, WorkerBaseArgs, PrefillWorkerArgs, DecodeWorkerArgs


class AIDynamoSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for AI Dynamo on Slurm systems."""

    def _container_mounts(self) -> list[str]:
        td = cast(AIDynamoTestDefinition, self.test_run.test.test_definition)
        mounts = [
            f"{td.huggingface_home_host_path}:{td.cmd_args.huggingface_home_container_path}",
        ]
        script_host = (self.test_run.output_path / "run.sh").resolve()
        script_container = "/opt/run.sh"
        yaml_path = (self.test_run.output_path / "dynamo_config.yaml").resolve()
        self._generate_wrapper_script(script_host, td, yaml_path)
        mounts.append(f"{script_host}:{script_container}")

        return mounts

    def image_path(self) -> str | None:
        tdef: AIDynamoTestDefinition = cast(AIDynamoTestDefinition, self.test_run.test.test_definition)
        return str(tdef.docker_image.installed_path)

    def _generate_wrapper_script(self, script_path: Path, td: AIDynamoTestDefinition, yaml_path: Path) -> None:
        lines = ["#!/bin/bash", ""]
        lines += self._common_header(td)
        lines += self._role_dispatch(td, yaml_path)
        self._write_script(script_path, lines)

    def _bg(self, cmd: str, stdout_tag: str, stderr_tag: str) -> str:
        cmd: List[str] = [
            f"{cmd}",
            f"> /cloudai_run_results/{stdout_tag}.txt",
            f"2> /cloudai_run_results/{stderr_tag}.txt &",
        ]
        return " \\\n  ".join(cmd)

    def _wait_for_frontend_marker(self) -> str:
        return (
            'echo "Waiting for frontend completion marker...";\n'
            'while [ ! -f "$DONE_MARKER" ]; do sleep 10; done;\n'
            'echo "Done marker found. Exiting prefill node.";\n'
        )

    def _etcd_cmd(self, etcd_cmd: str, etcd_port: int) -> str:
        return (
            f"{etcd_cmd} \\\n"
            f"  --listen-client-urls http://0.0.0.0:{etcd_port} \\\n"
            f"  --advertise-client-urls http://0.0.0.0:{etcd_port}"
        )

    def _nats_cmd(self, nats_cmd: str, nats_port: int) -> str:
        return f"{nats_cmd} -p {nats_port}"

    def _build_genai_perf_command(self, td: AIDynamoTestDefinition) -> str:
        args = td.cmd_args
        cmd: List[str] = [
            "genai-perf profile",
            f"-m {td.cmd_args.dynamo.model}",
            "--url ${DYNAMO_URL}",
            f"--endpoint-type {args.genai_perf.endpoint_type}",
            f"--output-tokens-mean {str(args.genai_perf.output_tokens_mean)}",
            f"--output-tokens-stddev {str(args.genai_perf.output_tokens_stddev)}",
            f"--random-seed {str(args.genai_perf.random_seed)}",
            f"--request-count {str(args.genai_perf.request_count)}",
            f"--synthetic-input-tokens-mean {str(args.genai_perf.synthetic_input_tokens_mean)}",
            f"--synthetic-input-tokens-stddev {str(args.genai_perf.synthetic_input_tokens_stddev)}",
            f"--warmup-request-count {str(args.genai_perf.warmup_request_count)}",
            "--profile-export-file profile.json",
            "--artifact-dir /cloudai_run_results/",
        ]
        if args.genai_perf.endpoint:
            cmd.append(f"--endpoint {args.genai_perf.endpoint}")
        if args.genai_perf.streaming:
            cmd.append("--streaming")
        if args.genai_perf.extra_inputs:
            cmd += [args.genai_perf.extra_inputs]
        if args.genai_perf.input_file:
            cmd.append(f"--input-file {args.genai_perf.input_file}")
        if args.genai_perf.concurrency:
            cmd.append(f"--concurrency {args.genai_perf.concurrency}")
        if args.genai_perf.request_rate:
            cmd.append(f"--request-rate {args.genai_perf.request_rate}")
        cmd += [ "-- -v --async" ]

        return " \\\n  ".join(cmd)

    def _common_header(self, td: AIDynamoTestDefinition) -> List[str]:
        return [
            "echo 'Launching node setup cmd'",
            self._node_setup_cmd(td),
            "echo 'Done executing node setup cmd'",
            f"export HF_HOME={td.cmd_args.huggingface_home_container_path}",
            "export DYNAMO_FRONTEND=$SLURM_JOB_MASTER_NODE",
            f'export NATS_SERVER="nats://${{DYNAMO_FRONTEND}}:{td.cmd_args.dynamo.nats_port}"',
            f'export ETCD_ENDPOINTS="http://${{DYNAMO_FRONTEND}}:{td.cmd_args.dynamo.etcd_port}"',
            f"export DYNAMO_URL=http://${{DYNAMO_FRONTEND}}:{td.cmd_args.dynamo.port}",
            f"export DYNAMO_HEALTH_URL=${{DYNAMO_URL}}/health",
            "export DONE_MARKER=/cloudai_run_results/frontend_done.marker",
            f"cd {td.cmd_args.dynamo.workspace_path}",
            "",
            'function get_role() {',
            '  ROLE="frontend"',
            '  if [[ -n "$FRONT_NODE" ]] && [[ "$FRONT_NODE" == "$SLURM_NODEID" ]]; then',
            '    return "frontend"',
            '  elif [[ -n "$PREFILL_NODES" ]] && [[ "$PREFILL_NODES" == *"$SLURM_NODEID"* ]]; then',
            '    return "prefill"',
            '  elif [[ -n "$DECODE_NODES" ]] && [[ "$DECODE_NODES" == *"$SLURM_NODEID"* ]]; then',
            '    return "decode"',
            '  fi',
            '',
            '  if [ "$SLURM_NODEID" -ge 1 ] && [ "$SLURM_NODEID" -le 1 ]; then',
            '    ROLE="prefill"',
            '  elif [ "$SLURM_NODEID" -ge $(( 1 + 1 )) ] && [ "$SLURM_NODEID" -le $(( 1 + 1 )) ]; then',
            '    ROLE="decode"',
            '  fi',
            '  return $ROLE',
            '}',
            '',
            "function launch_etcd() {",
            indent(self._bg(self._etcd_cmd(td.cmd_args.dynamo.etcd_cmd, td.cmd_args.dynamo.etcd_port), "etcd_stdout", "etcd_stderr"), '  '),
            "}",
            "",
            "function launch_nats() {",
            indent(self._bg(self._nats_cmd(td.cmd_args.dynamo.nats_cmd, td.cmd_args.dynamo.nats_port), "nats_stdout", "nats_stderr"), '  '),
            "}",
            "",
            "function wait_for_etcd() {",
            '  echo "Waiting for etcd to be ready...";\n'
            '  while [ "`curl -ks ${ETCD_ENDPOINTS}/readyz`" != "ok" ]; do sleep 10; done;\n'
            '  echo "etcd is ready";\n'
            "}",
            "",
            "function launch_ingress() {",
            indent(self._bg(td.cmd_args.dynamo.ingress_cmd, 'ingress_stdout', 'ingress_stderr'), '  '),
            "}",
            "",
            'function wait_for_dynamo_frontend() {',
            '  echo "Waiting for dynamo frontend to be ready..."',
            '  while [ "`curl -I -w \"%{http_code}\" -o /dev/null -sk ${DYNAMO_HEALTH_URL}`" != "200" ]; do sleep 10; done',
            '  echo "Dynamo frontend is ready"',
            '}',
            "",
            "function wait_for_frontend_marker() {",
            '  echo "Waiting for frontend completion marker...";\n'
            '  while [ ! -f "$DONE_MARKER" ]; do sleep 10; done;\n'
            '  echo "Done marker found. Exiting prefill node.";\n'
            "}",
            "",
            "function launch_genai_perf() {",
            indent(self._build_genai_perf_command(td), '  '),
            "}",
            "",
        ]

    def _indent(self, text: List[str], indent_level: int = 0) -> List[str]:
        return [indent(c, '  ' * indent_level) for c in text]

    def _role_dispatch(self, td: AIDynamoTestDefinition, yaml_config_path: Path) -> List[str]:
        prefill_n = td.cmd_args.dynamo.prefill_worker.num_nodes
        decode_n = td.cmd_args.dynamo.decode_worker.num_nodes

        assert isinstance(prefill_n, int), "prefill_worker.num_nodes must be an integer"
        assert isinstance(decode_n, int), "decode_worker.num_nodes must be an integer"

        dispatch = [
            'ROLE=$(get_role)',
            'echo "Node ID: $SLURM_NODEID, Role: $ROLE"',
            '',
            'if [ "$ROLE" == "frontend" ]; then',
        ]
        dispatch += self._indent(self._frontend_block(td), 1)
        dispatch += ['elif [ "$ROLE" == "prefill" ]; then']
        dispatch += self._indent(self._prefill_block(td), 1)
        dispatch += ['elif [ "$ROLE" == "decode" ]; then']
        dispatch += self._indent(self._decode_block(td), 1)
        dispatch += [
            'else',
            '  echo "Unknown role! Exiting."',
            '  exit 1',
            'fi',
        ]
        return dispatch

    def _node_setup_cmd(self, td: AIDynamoTestDefinition) -> str:
        return td.cmd_args.node_setup_cmd

    def _frontend_block(self, td: AIDynamoTestDefinition) -> List[str]:
        return [
            'launch_etcd',
            'launch_nats',
            'wait_for_etcd',
            'launch_ingress',
            '',
            self._bg(
                self._dynamo_cmd(td, td.cmd_args.dynamo.decode_worker),
                'decode_stdout_node${SLURM_NODEID}',
                'decode_stderr_node${SLURM_NODEID}',
            ),
            '',
            'wait_for_dynamo_frontend',
            '',
            f'for i in {{1..{td.cmd_args.genai_perf.iterations}}}; do',
            '  echo "Starting genai-perf run $i"',
            '  sleep 300',
            '  echo "done sleeping genai-perf run $i"',
            '  launch_genai_perf',
            'done',
            '',
            'echo "genai-perf runs finished"',
            'touch "$DONE_MARKER"',
            'exit 0',
        ]

    def _prefill_block(self, td: AIDynamoTestDefinition, indent_level: int = 0) -> List[str]:
        cmd = [
            "wait_for_etcd",
            "",
            self._bg(
                self._dynamo_cmd(td, td.cmd_args.dynamo.prefill_worker),
                "prefill_stdout_node${SLURM_NODEID}",
                "prefill_stderr_node${SLURM_NODEID}",
            ),
            "",
            "wait_for_frontend_marker",
            "exit 0",
        ]
        return [indent(c, '  ' * indent_level) for c in cmd]

    def _decode_block(self, td: AIDynamoTestDefinition) -> List[str]:
        return [
            "wait_for_etcd",
            "",
            self._bg(
                self._dynamo_cmd(td, td.cmd_args.dynamo.decode_worker),
                "decode_stdout_node${SLURM_NODEID}",
                "decode_stderr_node${SLURM_NODEID}",
            ),
            "wait_for_frontend_marker",
            "",
            "exit 0",
        ]

    def _dynamo_cmd(self, td: AIDynamoTestDefinition, worker: WorkerBaseArgs) -> str:
        cmd: List[str] = [
            f"{worker.cmd}",
            f"--model {td.cmd_args.dynamo.model}",
            f"--tensor-parallel-size {worker.tensor_parallel_size}",
            f"--pipeline-parallel-size {worker.pipeline_parallel_size}",
            f"--data-parallel-size {worker.data_parallel_size}",
            f"--gpu-memory-utilization {worker.gpu_memory_utilization}",
            f"{'--enforce-eager' if worker.enforce_eager else ''}",
        ]

        if worker.enable_expert_parallel:
            cmd.append("--enable-expert-parallel")
        if worker.extra_args:
            cmd.append(worker.extra_args)
        if td.cmd_args.extra_args:
            cmd.append(td.cmd_args.extra_args)
        return " \\\n  ".join(cmd)

    def _write_script(self, script_path: Path, lines: List[str]) -> None:
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("\n".join(lines), encoding="utf-8")
        script_path.chmod(0o755)

    def _gen_srun_command(self) -> str:
        num_nodes, _ = self.get_cached_nodes_spec()
        srun_prefix = self.gen_srun_prefix()
        srun_prefix.extend(
            [
                f"--nodes={num_nodes}",
                f"--ntasks={num_nodes}",
                "--ntasks-per-node=1",
            ]
        )
        return " \\\n  ".join([*srun_prefix, "/opt/run.sh"])

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

        td = cast(AIDynamoTestDefinition, self.test_run.test.test_definition)
        prefill_n = td.cmd_args.dynamo.prefill_worker.num_nodes
        decode_n = td.cmd_args.dynamo.decode_worker.num_nodes

        assert isinstance(prefill_n, int), "prefill_worker.num_nodes must be an integer"
        assert isinstance(decode_n, int), "decode_worker.num_nodes must be an integer"

        total_nodes = prefill_n + decode_n

        requested_nodes, node_list = self.system.get_nodes_by_spec(self.test_run.nnodes, self.test_run.nodes)
        if total_nodes > requested_nodes:
            raise ValueError(
                f"Not enough nodes requested: need {total_nodes} total nodes "
                f"({prefill_n} prefill + {decode_n} decode), "
                f"but only got {requested_nodes}"
            )

        result = (total_nodes, node_list)
        self._node_spec_cache[cache_key] = result
        return result
