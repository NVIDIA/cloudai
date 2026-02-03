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

from typing import cast

from cloudai.systems.slurm import SlurmCommandGenStrategy

from .vllm import VllmCmdArgs, VllmTestDefinition


class VllmSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for vLLM on Slurm systems."""

    VLLM_RUN_SCRIPT_NAME = "vllm_run.sh"

    def _container_mounts(self) -> list[str]:
        return []

    def get_vllm_serve_command(self) -> list[str]:
        tdef: VllmTestDefinition = cast(VllmTestDefinition, self.test_run.test)
        tdef_cmd_args: VllmCmdArgs = tdef.cmd_args
        return ["vllm", "serve", tdef_cmd_args.model, "--port", str(tdef_cmd_args.port)]

    def generate_serve_run_and_wait_block(self) -> str:
        """Generate bash block to run vLLM serve and wait for health check."""
        tdef: VllmTestDefinition = cast(VllmTestDefinition, self.test_run.test)
        cmd_args: VllmCmdArgs = tdef.cmd_args
        serve_cmd = " ".join(self.get_vllm_serve_command())

        return f"""\
{serve_cmd} &
VLLM_PID=$!

TIMEOUT={cmd_args.vllm_serve_wait_seconds}
SLEEP_INTERVAL=5
HOST=0.0.0.0
PORT={cmd_args.port}

end_time=$(($(date +%s) + TIMEOUT))
while [ "$(date +%s)" -lt "$end_time" ]; do
    if curl -sf "http://${{HOST}}:${{PORT}}/health" > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM server process died unexpectedly!"
        exit 1
    fi
    sleep "$SLEEP_INTERVAL"
done

if ! curl -sf "http://${{HOST}}:${{PORT}}/health" > /dev/null 2>&1; then
    echo "Timeout waiting for vLLM to start"
    exit 1
fi"""

    def _gen_srun_command(self) -> str:
        script_path = self.test_run.output_path / self.VLLM_RUN_SCRIPT_NAME
        script_path.write_text(self.generate_serve_run_and_wait_block())

        srun_parts = [
            *self.gen_srun_prefix(),
            "--ntasks-per-node=1",
            "--ntasks=1",
            "bash",
            "-c",
            f'"{script_path.absolute()}"',
        ]
        return " ".join(srun_parts)
