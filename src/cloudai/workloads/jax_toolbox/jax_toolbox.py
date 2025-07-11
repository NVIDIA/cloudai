# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import re
from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, field_serializer

from cloudai.core import JobStatusResult, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition


class JaxFdl(BaseModel):
    """JAX FDL configuration."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_policy: Union[str, list[str]] = "save_nothing"
    dcn_mesh_shape: Union[str, list[str]] = "'[1, 1, 1]'"
    fprop_dtype: Union[str, list[str]] = "bfloat16"
    ici_mesh_shape: Union[str, list[str]] = "'[1, 8, 1]'"
    max_steps: Union[int, list[int]] = 20
    num_gpus: Union[int, list[int]] = 64
    num_microbatches: Union[int, list[int]] = 1
    num_stages: Union[int, list[int]] = 1
    percore_batch_size: Union[float, list[float]] = 4.0
    use_fp8: Union[int, list[int]] = 1
    use_repeated_layer: Union[bool, list[bool]] = False

    @field_serializer("fprop_dtype")
    def fprop_dtype_serializer(self, value: str) -> str:
        if value.startswith('\\"') and value.endswith('\\"'):
            return value
        return f'\\"{value}\\"'

    @field_serializer("checkpoint_policy")
    def checkpoint_policy_serializer(self, value: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(value, list):
            return [self._serialize_single_policy(v) for v in value]
        return self._serialize_single_policy(value)

    def _serialize_single_policy(self, value: str) -> str:
        if value.startswith('\\"') and value.endswith('\\"'):
            return value
        elif value.startswith('"') and value.endswith('"'):
            return value.replace('"', '\\"')
        return f'\\"{value}\\"'


class JaxToolboxCmdArgs(CmdArgs):
    """JAX Toolbox test command arguments."""

    docker_image_url: str
    load_container: bool = True
    output_path: Optional[str] = None


class JaxToolboxTestDefinition(TestDefinition):
    """Test object for JAX Toolbox."""

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        result = self.check_profile_stderr(tr.output_path)
        if not result.is_successful:
            return result

        error_files = list(tr.output_path.glob("error-*.txt"))
        if not error_files:
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    f"No 'error-*.txt' files found in the output directory, {tr.output_path}. There are two stages in "
                    "the Grok run. The profiling stage passed successfully, but something went wrong in the actual run "
                    "stage. Please ensure the actual run stage completed successfully. "
                    "Run the generated sbatch script manually to debug."
                ),
            )

        return self.check_error_files(error_files, tr.output_path)

    def check_profile_stderr(self, output_path: Path) -> JobStatusResult:
        """
        Check all profile_stderr_*.txt files for known error messages.

        Args:
            profile_stderr_path (Path): Path to the 'profile_stderr.txt' file.
            output_path (Path): Path to the output directory.

        Returns:
            JobStatusResult: The result containing the job status and an optional error message.
        """
        profile_stderr_files = list(output_path.glob("profile_stderr_*.txt"))
        if not profile_stderr_files:
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    f"No profile_stderr_*.txt files found in the specified output directory, {output_path}. "
                    "These files are expected to be created during the profiling stage. "
                    "Please ensure the profiling stage completed successfully. "
                    "Run the generated sbatch script manually to debug."
                ),
            )

        for profile_stderr_path in profile_stderr_files:
            with open(profile_stderr_path, "r") as file:
                content = file.read()

                if "[PAX STATUS]: E2E time: Elapsed time for " in content:
                    result = self.check_common_errors(content, profile_stderr_path, output_path)
                    if result.is_successful:
                        return JobStatusResult(is_successful=True)
                    else:
                        return result

        return JobStatusResult(
            is_successful=False,
            error_message=(
                "The profiling stage completed but did not generate the expected "
                "'[PAX STATUS]: E2E time: Elapsed time for ' keyword in any of the "
                "profile_stderr_*.txt files. There are two stages in the Grok run, "
                "and an error occurred in the profiling stage. While profile_stderr_*.txt "
                "files were created, the expected keyword is missing. You need to run the "
                "sbatch script manually to see what happens."
            ),
        )

    def check_common_errors(self, content: str, file_path: Path, output_path: Path) -> JobStatusResult:
        """
        Check for common errors in the file content.

        Args:
            content (str): The content of the file to check.
            file_path (Path): The path of the file being checked.
            output_path (Path): Path to the output directory.

        Returns:
            JobStatusResult: The result containing the job status and an optional error message.
        """
        if "CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected" in content:
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    "CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected. This may be due to missing "
                    "environment variables, specifically but not limited to CUDA_VISIBLE_DEVICES. "
                    "First, check if GPUs are available on the server. "
                    "Second, if running the job with Slurm, ensure proper resource-related options are set, "
                    "including GPU resource requirements. Lastly, check environment variables. "
                    "If the problem persists, verify commands and environment variables by running a simple GPU-only "
                    "example command."
                ),
            )
        if "Terminating process because the coordinator detected missing heartbeats" in content:
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    "Terminating process because the coordinator detected missing heartbeats. This most likely "
                    f"indicates that another task died. Please review the file at {file_path} and any relevant logs in"
                    f" {output_path}. Ensure the servers allocated for this task can reach each other with their "
                    "hostnames, and they can open any ports and reach others' ports."
                ),
            )
        if "NCCL operation ncclGroupEnd() failed" in content:
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    "NCCL operation ncclGroupEnd() failed: unhandled system error. Please check if the NCCL-test "
                    "passes. Run with NCCL_DEBUG=INFO for more details."
                ),
            )
        if re.search(r"pyxis:\s+mktemp: failed to create directory via template", content):
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    "pyxis: mktemp: failed to create directory via template. This is due to insufficient disk cache "
                    "capacity. This is not a CloudAI issue. When you run JaxToolbox, CloudAI executes srun, which "
                    "includes the container image option. When the container image argument is a remote URL, "
                    "Slurm downloads and caches the Docker image locally. It fails with this error when the system "
                    "does not have enough disk capacity to cache the Docker image."
                ),
            )

        return JobStatusResult(is_successful=True)

    def check_error_files(self, error_files: list[Path], output_path: Path) -> JobStatusResult:
        """
        Check the error-*.txt files for known error messages.

        Args:
            error_files (list[Path]): List of paths to error files.
            output_path (Path): Path to the output directory.

        Returns:
            JobStatusResult: The result containing the job status and an optional error message.
        """
        for error_file in error_files:
            with error_file.open("r") as file:
                content = file.read()
                result = self.check_common_errors(content, error_file, output_path)
                if not result.is_successful:
                    return result
                if "E2E time: Elapsed time for" not in content:
                    return JobStatusResult(
                        is_successful=False,
                        error_message=(
                            f"The file {error_file} does not contain the expected 'E2E time: Elapsed time for' "
                            "keyword at the end. This indicates the actual run did not complete successfully. "
                            "Please debug this manually to ensure the actual run stage completes as expected."
                        ),
                    )

        return JobStatusResult(is_successful=True)


class XLAFlags(BaseModel):
    """XLA flags configuration."""

    model_config = ConfigDict(extra="forbid")

    xla_disable_hlo_passes: Union[str, list[str]] = "rematerialization"
    xla_dump_hlo_pass_re: Union[str, list[str]] = ".*"
    xla_gpu_enable_all_gather_combine_by_dim: Union[bool, list[bool]] = False
    xla_gpu_enable_highest_priority_async_stream: Union[bool, list[bool]] = True
    xla_gpu_enable_latency_hiding_scheduler: Union[bool, list[bool]] = True
    xla_gpu_enable_pipelined_all_gather: Union[bool, list[bool]] = True
    xla_gpu_enable_pipelined_all_reduce: Union[bool, list[bool]] = True
    xla_gpu_enable_pipelined_reduce_scatter: Union[bool, list[bool]] = True
    xla_gpu_enable_reduce_scatter_combine_by_dim: Union[bool, list[bool]] = False
    xla_gpu_enable_triton_gemm: Union[bool, list[bool]] = False
    xla_gpu_enable_triton_softmax_fusion: Union[bool, list[bool]] = False
    xla_gpu_enable_while_loop_double_buffering: Union[bool, list[bool]] = True
    xla_gpu_graph_level: Union[int, list[int]] = 0


class SetupFlags(BaseModel):
    """Setup flags configuration."""

    model_config = ConfigDict(extra="forbid")

    docker_workspace_dir: str = "/opt/paxml/workspace/"
    enable_checkpoint_saving: bool = False
    gpus_per_node: int = 8
    mpi: str = "pmix"
    num_nodes: int = 8
    tfds_data_dir: str = "/opt/dataset"
