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

from cloudai import TestRun
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy

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
        self._generate_wrapper_script(script_host, td)
        mounts.append(f"{script_host}:{script_container}")
        return mounts

    def image_path(self, tr: TestRun) -> str | None:
        tdef: AIDynamoTestDefinition = cast(AIDynamoTestDefinition, tr.test.test_definition)
        return str(tdef.docker_image.installed_path)

    def _generate_wrapper_script(self, script_path: Path, td: AIDynamoTestDefinition) -> None:
        lines = ["#!/bin/bash", ""]
        lines += self._common_header(td)
        lines += self._role_dispatch(td)
        self._write_script(script_path, lines)

    def _common_header(self, td: AIDynamoTestDefinition) -> List[str]:
        return [
            f"export HF_HOME={td.hugging_face_home_path}",
            "export DYNAMO_FRONTEND=$SLURM_JOB_MASTER_NODE",
            f'export NATS_SERVER="nats://${{DYNAMO_FRONTEND}}:{td.cmd_args.port_nats}"',
            f'export ETCD_ENDPOINTS="http://${{DYNAMO_FRONTEND}}:{td.cmd_args.port_etcd}"',
            "cd /workspace/examples/llm/",
            "CURRENT_HOST=$(hostname)",
            "export DONE_MARKER=/cloudai_run_results/frontend_done.marker",
            "",
        ]

    def _role_dispatch(self, td: AIDynamoTestDefinition) -> List[str]:
        prefill_n = td.cmd_args.num_prefill_nodes
        decode_n = td.cmd_args.num_decode_nodes

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
        dispatch += self._frontend_block(td)
        dispatch += ['elif [ "$ROLE" == "prefill" ]; then']
        dispatch += self._prefill_block(td)
        dispatch += ['elif [ "$ROLE" == "decode" ]; then']
        dispatch += self._decode_block(td)
        dispatch += [
            "else",
            "  echo 'Unknown role! Exiting.'",
            "  exit 1",
            "fi",
        ]
        return dispatch

    def _frontend_block(self, td: AIDynamoTestDefinition) -> List[str]:
        cmd = self._build_genai_perf_command(td)
        return [
            self._bg(self._etcd_cmd(td), "etcd_stdout", "etcd_stderr"),
            self._bg(self._nats_cmd(), "nats_stdout", "nats_stderr"),
            self._bg(
                self._dynamo_cmd("graphs.agg_router:Frontend", td.cmd_args.config_path),
                "frontend_stdout",
                "frontend_stderr",
            ),
            f"sleep {td.cmd_args.sleep_seconds}",
            "echo 'Starting initial genai-perf'",
            f"{cmd} > /cloudai_run_results/perf_initial_stdout.txt 2> /cloudai_run_results/perf_initial_stderr.txt &",
            "sleep 60",
            "echo 'Killing initial genai-perf run'",
            "pkill -f genai-perf || true",
            "echo 'Starting second genai-perf run'",
            cmd,
            "echo 'genai-perf finished. Writing done marker'",
            'touch "$DONE_MARKER"',
            "exit 0",
        ]

    def _prefill_block(self, td: AIDynamoTestDefinition) -> List[str]:
        return [
            self._bg(
                self._dynamo_cmd("components.prefill_worker:PrefillWorker", td.cmd_args.config_path, service_name=None),
                "prefill_stdout_node${SLURM_NODEID}",
                "prefill_stderr_node${SLURM_NODEID}",
            ),
            "echo 'Waiting for frontend completion marker...'",
            'while [ ! -f "$DONE_MARKER" ]; do sleep 10; done',
            "echo 'Done marker found. Exiting prefill node.'",
            "exit 0",
        ]

    def _decode_block(self, td: AIDynamoTestDefinition) -> List[str]:
        return [
            self._bg(
                self._dynamo_cmd("components.worker:VllmWorker", td.cmd_args.config_path, service_name="VllmWorker"),
                "decode_stdout_node${SLURM_NODEID}",
                "decode_stderr_node${SLURM_NODEID}",
            ),
            "echo 'Waiting for frontend completion marker...'",
            'while [ ! -f "$DONE_MARKER" ]; do sleep 10; done',
            "echo 'Done marker found. Exiting decode node.'",
            "exit 0",
        ]

    def _etcd_cmd(self, td: AIDynamoTestDefinition) -> str:
        return (
            f"etcd --listen-client-urls http://0.0.0.0:{td.cmd_args.port_etcd} "
            f"--advertise-client-urls http://0.0.0.0:{td.cmd_args.port_etcd}"
        )

    def _nats_cmd(self) -> str:
        return "nats-server -js"

    def _dynamo_cmd(self, module: str, config: str, service_name: Optional[str] = None) -> str:
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
            args.served_model_name,
            "--url",
            f"${{CURRENT_HOST}}:{args.port}",
            "--endpoint-type",
            args.endpoint_type,
            "--service-kind",
            args.service_kind,
            "--endpoint",
            args.endpoint,
        ]
        if args.streaming:
            cmd.append("--streaming")
        cmd += [
            "--warmup-request-count",
            str(args.warmup_request_count),
            "--random-seed",
            str(args.random_seed),
            "--synthetic-input-tokens-mean",
            str(args.synthetic_input_tokens_mean),
            "--synthetic-input-tokens-stddev",
            str(args.synthetic_input_tokens_stddev),
            "--output-tokens-mean",
            str(args.output_tokens_mean),
            "--output-tokens-stddev",
            str(args.output_tokens_stddev),
        ]
        if args.extra_inputs:
            cmd += [args.extra_inputs]
        cmd += [
            "--profile-export-file",
            "profile.json",
            "--artifact-dir",
            "/cloudai_run_results/",
            "--concurrency",
            str(args.concurrency),
            "--request-count",
            str(args.request_count),
            "--",
            "-v",
            "--async",
        ]
        return " ".join(cmd)

    def _gen_srun_command(
        self,
        env_vars: Dict[str, Union[str, List[str]]],
        cmd_args: Dict[str, Union[str, List[str]]],
        tr: TestRun,
    ) -> str:
        num_nodes, _ = self.system.get_nodes_by_spec(tr.num_nodes, tr.nodes)
        td = cast(AIDynamoTestDefinition, tr.test.test_definition)
        expected = 1 + td.cmd_args.num_prefill_nodes + td.cmd_args.num_decode_nodes

        if num_nodes != expected:
            raise ValueError(
                f"Invalid node count: expected {expected} total nodes "
                f"(1 frontend + {td.cmd_args.num_prefill_nodes} prefill + "
                f"{td.cmd_args.num_decode_nodes} decode), but got {num_nodes}"
            )

        srun_prefix = self.gen_srun_prefix(tr)
        srun_prefix.extend(
            [
                f"--nodes={num_nodes}",
                f"--ntasks={num_nodes}",
                "--ntasks-per-node=1",
            ]
        )
        return " ".join([*srun_prefix, "/opt/run.sh"])
