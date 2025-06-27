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
from typing import Dict, List, Optional, Union, cast

import yaml

from cloudai.models.workload import TestRun
from cloudai.systems.slurm import SlurmCommandGenStrategy

from .ai_dynamo import AIDynamoTestDefinition


class AIDynamoSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for AI Dynamo on Slurm systems."""

    def _container_mounts(self, tr: TestRun) -> list[str]:
        td = cast(AIDynamoTestDefinition, tr.test.test_definition)
        mounts = [
            f"{td.hugging_face_home_path}:{td.hugging_face_home_path}",
        ]
        script_host = (tr.output_path / "run.sh").resolve()
        script_container = "/opt/run.sh"
        yaml_path = (tr.output_path / "dynamo_config.yaml").resolve()
        self._generate_wrapper_script(script_host, td, yaml_path)
        mounts.append(f"{script_host}:{script_container}")

        self._generate_yaml_config(td, yaml_path)
        mounts.append(f"{yaml_path}:{yaml_path}")

        return mounts

    def _generate_yaml_config(self, td: AIDynamoTestDefinition, yaml_path: Path) -> Path:
        config = {
            "Frontend": td.cmd_args.dynamo.frontend.model_dump(
                by_alias=True, exclude={"port_etcd", "port_nats"}, exclude_none=True
            ),
            "Processor": td.cmd_args.dynamo.processor.model_dump(by_alias=True, exclude_none=True),
            "Router": td.cmd_args.dynamo.router.model_dump(by_alias=True, exclude_none=True),
            "VllmWorker": td.cmd_args.dynamo.vllm_worker.model_dump(
                by_alias=True, exclude={"num_nodes"}, exclude_none=True
            ),
            "PrefillWorker": td.cmd_args.dynamo.prefill_worker.model_dump(
                by_alias=True, exclude={"num_nodes"}, exclude_none=True
            ),
        }
        model_name = td.cmd_args.served_model_name
        config["Frontend"]["served_model_name"] = model_name
        for section_name in ["Processor", "Router", "PrefillWorker", "VllmWorker"]:
            config[section_name]["model"] = model_name
        with open(yaml_path, "w") as yaml_file:
            yaml.dump(config, yaml_file)
        return yaml_path

    def image_path(self, tr: TestRun) -> str | None:
        tdef: AIDynamoTestDefinition = cast(AIDynamoTestDefinition, tr.test.test_definition)
        return str(tdef.docker_image.installed_path)

    def _generate_wrapper_script(self, script_path: Path, td: AIDynamoTestDefinition, yaml_path: Path) -> None:
        hf_home = td.hugging_face_home_path
        port_nats = td.cmd_args.dynamo.frontend.port_nats
        port_etcd = td.cmd_args.dynamo.frontend.port_etcd
        lines = ["#!/bin/bash", ""]
        lines += self._common_header(hf_home, port_nats, port_etcd)
        lines += self._role_dispatch(td, yaml_path)
        self._write_script(script_path, lines)

    def _common_header(self, hf_home: Path, port_nats: int, port_etcd: int) -> List[str]:
        return [
            f"export HF_HOME={hf_home}",
            "export DYNAMO_FRONTEND=$SLURM_JOB_MASTER_NODE",
            f'export NATS_SERVER="nats://${{DYNAMO_FRONTEND}}:{port_nats}"',
            f'export ETCD_ENDPOINTS="http://${{DYNAMO_FRONTEND}}:{port_etcd}"',
            "cd /workspace/examples/llm/",
            "CURRENT_HOST=$(hostname)",
            "export DONE_MARKER=/cloudai_run_results/frontend_done.marker",
            "",
        ]

    def _role_dispatch(self, td: AIDynamoTestDefinition, yaml_config_path: Path) -> List[str]:
        prefill_n = td.cmd_args.dynamo.prefill_worker.num_nodes
        decode_n = td.cmd_args.dynamo.vllm_worker.num_nodes

        assert isinstance(prefill_n, int), "prefill_worker.num_nodes must be an integer"
        assert isinstance(decode_n, int), "vllm_worker.num_nodes must be an integer"

        dispatch = [
            'ROLE="frontend"',
            f'if [ "$SLURM_NODEID" -ge 1 ] && [ "$SLURM_NODEID" -le {prefill_n} ]; then',
            '  ROLE="prefill"',
            f'elif [ "$SLURM_NODEID" -ge $(( {prefill_n} + 1 )) ] '
            f'&& [ "$SLURM_NODEID" -le $(( {prefill_n} + {decode_n} )) ]; then',
            '  ROLE="decode"',
            "fi",
            'echo "Node ID: $SLURM_NODEID, Role: $ROLE"',
            "",
            'if [ "$ROLE" == "frontend" ]; then',
        ]
        dispatch += self._frontend_block(td, yaml_config_path)
        dispatch += ['elif [ "$ROLE" == "prefill" ]; then']
        dispatch += self._prefill_block(td, yaml_config_path)
        dispatch += ['elif [ "$ROLE" == "decode" ]; then']
        dispatch += self._decode_block(td, yaml_config_path)
        dispatch += [
            "else",
            "  echo 'Unknown role! Exiting.'",
            "  exit 1",
            "fi",
        ]
        return dispatch

    def _frontend_block(self, td: AIDynamoTestDefinition, yaml_config_path: Path) -> List[str]:
        cmd = self._build_genai_perf_command(td)
        return [
            self._bg(self._etcd_cmd(td.cmd_args.dynamo.frontend.port_etcd), "etcd_stdout", "etcd_stderr"),
            self._bg(self._nats_cmd(), "nats_stdout", "nats_stderr"),
            self._bg(
                self._dynamo_cmd("graphs.agg_router:Frontend", yaml_config_path),
                "frontend_stdout",
                "frontend_stderr",
            ),
            f"sleep {td.cmd_args.sleep_seconds}",
            "echo 'Starting second genai-perf run'",
            cmd,
            "echo 'genai-perf finished. Writing done marker'",
            'touch "$DONE_MARKER"',
            "exit 0",
        ]

    def _etcd_cmd(self, port_etcd: int) -> str:
        return (
            f"etcd --listen-client-urls http://0.0.0.0:{port_etcd} "
            f"--advertise-client-urls http://0.0.0.0:{port_etcd} "
            "--log-level debug"
        )

    def _nats_cmd(self) -> str:
        return "nats-server -js"

    def _prefill_block(self, td: AIDynamoTestDefinition, yaml_config_path: Path) -> List[str]:
        return [
            self._bg(
                self._dynamo_cmd("components.prefill_worker:PrefillWorker", yaml_config_path, service_name=None),
                "prefill_stdout_node${SLURM_NODEID}",
                "prefill_stderr_node${SLURM_NODEID}",
            ),
            "echo 'Waiting for frontend completion marker...'",
            'while [ ! -f "$DONE_MARKER" ]; do sleep 10; done',
            "echo 'Done marker found. Exiting prefill node.'",
            "exit 0",
        ]

    def _decode_block(self, td: AIDynamoTestDefinition, yaml_config_path: Path) -> List[str]:
        return [
            self._bg(
                self._dynamo_cmd("components.worker:VllmWorker", yaml_config_path, service_name="VllmWorker"),
                "decode_stdout_node${SLURM_NODEID}",
                "decode_stderr_node${SLURM_NODEID}",
            ),
            "echo 'Waiting for frontend completion marker...'",
            'while [ ! -f "$DONE_MARKER" ]; do sleep 10; done',
            "echo 'Done marker found. Exiting decode node.'",
            "exit 0",
        ]

    def _dynamo_cmd(self, module: str, config: Path, service_name: Optional[str] = None) -> str:
        svc = f"--service-name {service_name} " if service_name else ""
        return f"dynamo serve {module} -f {config} {svc}"

    def _bg(self, cmd: str, stdout_tag: str, stderr_tag: str) -> str:
        return f"{cmd} > /cloudai_run_results/{stdout_tag}.txt 2> /cloudai_run_results/{stderr_tag}.txt &"

    def _write_script(self, script_path: Path, lines: List[str]) -> None:
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("\n".join(lines), encoding="utf-8")
        script_path.chmod(0o755)

    def _build_genai_perf_command(self, td: AIDynamoTestDefinition) -> str:
        args = td.cmd_args
        cmd: List[str] = [
            "genai-perf",
            "profile",
            "-m",
            td.cmd_args.served_model_name,
            "--url",
            f"${{CURRENT_HOST}}:{args.genai_perf.port}",
            "--endpoint-type",
            args.genai_perf.endpoint_type,
        ]
        if args.genai_perf.endpoint:
            cmd.append(f"--endpoint {args.genai_perf.endpoint}")
        if args.genai_perf.streaming:
            cmd.append("--streaming")
        if args.genai_perf.extra_inputs:
            cmd += [args.genai_perf.extra_inputs]
        if args.genai_perf.input_file:
            cmd.append(f"--input-file {args.genai_perf.input_file}")
        cmd += [
            "--output-tokens-mean",
            str(args.genai_perf.output_tokens_mean),
            "--osl",
            str(args.genai_perf.osl),
            "--output-tokens-stddev",
            str(args.genai_perf.output_tokens_stddev),
            "--random-seed",
            str(args.genai_perf.random_seed),
            "--request-count",
            str(args.genai_perf.request_count),
            "--synthetic-input-tokens-mean",
            str(args.genai_perf.synthetic_input_tokens_mean),
            "--isl",
            str(args.genai_perf.isl),
            "--synthetic-input-tokens-stddev",
            str(args.genai_perf.synthetic_input_tokens_stddev),
            "--warmup-request-count",
            str(args.genai_perf.warmup_request_count),
        ]
        if args.genai_perf.concurrency:
            cmd.append(f"--concurrency {args.genai_perf.concurrency}")
        cmd += [
            "--profile-export-file",
            "profile.json",
            "--artifact-dir",
            "/cloudai_run_results/",
            "--",
            "-v",
            "--async",
        ]
        if args.genai_perf.request_rate:
            cmd.append(f"--request-rate {args.genai_perf.request_rate}")
        return " ".join(cmd)

    def _gen_srun_command(
        self,
        env_vars: Dict[str, Union[str, List[str]]],
        cmd_args: Dict[str, Union[str, List[str]]],
        tr: TestRun,
    ) -> str:
        num_nodes, _ = self.get_cached_nodes_spec(tr)
        srun_prefix = self.gen_srun_prefix(tr)
        srun_prefix.extend(
            [
                f"--nodes={num_nodes}",
                f"--ntasks={num_nodes}",
                "--ntasks-per-node=1",
            ]
        )
        return " ".join([*srun_prefix, "/opt/run.sh"])

    def get_cached_nodes_spec(self, tr: TestRun) -> tuple[int, list[str]]:
        cache_key = f"{tr.current_iteration}:{tr.step}:{tr.num_nodes}:{','.join(tr.nodes)}"

        if cache_key in self._node_spec_cache:
            return self._node_spec_cache[cache_key]

        td = cast(AIDynamoTestDefinition, tr.test.test_definition)
        prefill_n = td.cmd_args.dynamo.prefill_worker.num_nodes
        decode_n = td.cmd_args.dynamo.vllm_worker.num_nodes

        assert isinstance(prefill_n, int), "prefill_worker.num_nodes must be an integer"
        assert isinstance(decode_n, int), "vllm_worker.num_nodes must be an integer"

        total_nodes = 1 + prefill_n + decode_n

        requested_nodes, node_list = self.system.get_nodes_by_spec(tr.nnodes, tr.nodes)
        if total_nodes > requested_nodes:
            raise ValueError(
                f"Not enough nodes requested: need {total_nodes} total nodes "
                f"(1 frontend + {prefill_n} prefill + {decode_n} decode), "
                f"but only got {requested_nodes}"
            )

        result = (total_nodes, node_list)
        self._node_spec_cache[cache_key] = result
        return result
