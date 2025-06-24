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

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from cloudai.core import BaseJob, File, Installable, System
from cloudai.models.scenario import ReportConfig, parse_reports_spec
from cloudai.util import CommandShell

from .slurm_metadata import SlurmStepMetadata
from .slurm_node import SlurmNode, SlurmNodeState


class DataRepositoryConfig(BaseModel):
    """Configuration for a data repository."""

    endpoint: str
    verify_certs: bool = True


def parse_node_list(node_list: str) -> List[str]:
    """
    Expand a list of node names (with ranges) into a flat list of individual node names, keeping leading zeroes.

    Args:
        node_list (str): A list of node names, possibly including ranges.

    Returns:
        List[str]: A flat list of expanded node names with preserved zeroes.
    """
    node_list = node_list.strip()
    nodes = []
    if not node_list:
        return []

    components = re.split(r",\s*(?![^[]*\])", node_list)
    for component in components:
        if "[" not in component:
            nodes.append(component)
        else:
            header, node_number = component.split("[")
            node_number = node_number.replace("]", "")
            ranges = node_number.split(",")
            for r in ranges:
                if "-" in r:
                    start_node, end_node = r.split("-")
                    number_of_digits = len(end_node)
                    nodes.extend(
                        [f"{header}{str(i).zfill(number_of_digits)}" for i in range(int(start_node), int(end_node) + 1)]
                    )
                else:
                    nodes.append(f"{header}{r}")

    return nodes


class SlurmGroup(BaseModel):
    """Represents a group of nodes within a partition."""

    model_config = ConfigDict(extra="forbid")
    name: str
    nodes: List[str]


class SlurmPartition(BaseModel):
    """Represents a partition within a Slurm system."""

    model_config = ConfigDict(extra="forbid")
    name: str
    groups: List[SlurmGroup] = []
    slurm_nodes: list[SlurmNode] = Field(default_factory=list[SlurmNode], exclude=True)


class SlurmSystem(BaseModel, System):
    """
    Represents a Slurm system.

    Attributes
        output_path (Path): Path to the output directory.
        default_partition (str): The default partition for job submission.
        partitions (Dict[str, List[SlurmNode]]): Mapping of partition names to lists of SlurmNodes.
        account (Optional[str]): Account name for charging resources used by this job.
        distribution (Optional[str]): Specifies alternate distribution methods for remote processes.
        mpi (Optional[str]): Indicates the Process Management Interface (PMI) implementation to be used for
            inter-process communication.
        gpus_per_node (Optional[int]): Specifies the number of GPUs available per node.
        ntasks_per_node (Optional[int]): Specifies the number of tasks that can run concurrently on a single node.
        cache_docker_images_locally (bool): Whether to cache Docker images locally for the Slurm system.
        groups (Dict[str, Dict[str, List[SlurmNode]]]): Nested mapping where the key is the partition name and the
            value is another dictionary with group names as keys and lists of SlurmNodes as values, representing the
            group composition within each partition.
        cmd_shell (CommandShell): An instance of CommandShell for executing system commands.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    install_path: Path
    output_path: Path
    default_partition: str
    partitions: List[SlurmPartition]
    account: Optional[str] = None
    distribution: Optional[str] = None
    mpi: str = "pmix"
    gpus_per_node: Optional[int] = None
    ntasks_per_node: Optional[int] = None
    cache_docker_images_locally: bool = False
    global_env_vars: Dict[str, Any] = {}
    scheduler: str = "slurm"
    monitor_interval: int = 60
    cmd_shell: CommandShell = Field(default=CommandShell(), exclude=True)
    extra_srun_args: Optional[str] = None
    extra_sbatch_args: list[str] = []
    supports_gpu_directives_cache: Optional[bool] = Field(default=None, exclude=True)
    container_mount_home: bool = False

    data_repository: Optional[DataRepositoryConfig] = None
    reports: Optional[dict[str, ReportConfig]] = None

    @field_validator("reports", mode="before")
    @classmethod
    def parse_reports(cls, value: dict[str, Any] | None) -> dict[str, ReportConfig] | None:
        return parse_reports_spec(value)

    @property
    def groups(self) -> Dict[str, Dict[str, List[SlurmNode]]]:
        groups: Dict[str, Dict[str, List[SlurmNode]]] = {}
        for part in self.partitions:
            groups[part.name] = {}
            for group in part.groups:
                node_names = set()
                for group_nodes in group.nodes:
                    node_names.update(set(parse_node_list(group_nodes)))

                groups[part.name][group.name] = []
                for node_name in node_names:
                    node_in_partition = next((node for node in part.slurm_nodes if node.name == node_name), None)
                    if not node_in_partition:
                        logging.error(f"Node '{node_name}' not found in partition '{part.name}'")
                        groups[part.name][group.name].append(
                            SlurmNode(name=node_name, partition=self.name, state=SlurmNodeState.UNKNOWN_STATE)
                        )
                    else:
                        groups[part.name][group.name].append(node_in_partition)

        return groups

    @property
    def supports_gpu_directives(self) -> bool:
        if self.supports_gpu_directives_cache is not None:
            return self.supports_gpu_directives_cache

        stdout, stderr = self.fetch_command_output("scontrol show config")
        if stderr:
            logging.warning(f"Error checking GPU support: {stderr}")
            self.supports_gpu_directives_cache = True
            return True

        for line in stdout.splitlines():
            if "GresTypes" in line and "gpu" in line:
                self.supports_gpu_directives_cache = True
                return True

        self.supports_gpu_directives_cache = False
        return False

    @field_serializer("install_path", "output_path")
    def _path_serializer(self, v: Path) -> str:
        return str(v)

    def update(self) -> None:
        """
        Update the system object for a SLURM system.

        This method updates the system object by querying the current state of each node using the 'sinfo' and 'squeue'
        commands, and correlating this information to determine the state of each node and the user running jobs on
        each node.
        """
        all_nodes = self.nodes_from_sinfo()
        self.update_nodes_state_and_user(all_nodes, insert_new=True)
        self.update_nodes_state_and_user(self.nodes_from_squeue())

    def nodes_from_sinfo(self) -> list[SlurmNode]:
        sinfo_output, _ = self.fetch_command_output("sinfo -o '%P|%t|%u|%N'")
        nodes: list[SlurmNode] = []
        for line in sinfo_output.split("\n"):
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            partition, state, user, nodelist = parts[:4]
            partition = partition.rstrip("*").strip()
            node_names = parse_node_list(nodelist)
            logging.debug(f"{partition=}, {state=}, {nodelist=}, {node_names=}")
            for node_name in node_names:
                nodes.append(
                    SlurmNode(name=node_name, partition=partition, state=self.convert_state_to_enum(state), user=user)
                )
        return nodes

    def nodes_from_squeue(self) -> list[SlurmNode]:
        squeue_output, _ = self.fetch_command_output("squeue --states=running,pending --noheader -o '%P|%T|%N|%u'")
        nodes: list[SlurmNode] = []
        for line in squeue_output.split("\n"):
            parts = line.split("|")
            if len(parts) < 4:
                continue
            partition, _, nodelist, user = parts[:4]
            node_names = parse_node_list(nodelist)
            for node in node_names:
                nodes.append(SlurmNode(name=node, partition=partition, state=SlurmNodeState.ALLOCATED, user=user))
        return nodes

    def update_nodes_state_and_user(self, nodes: list[SlurmNode], insert_new: bool = False) -> None:
        for node in nodes:
            for part in self.partitions:
                if part.name != node.partition:
                    continue

                found = False
                for pnode in part.slurm_nodes:
                    if pnode.name != node.name:
                        continue
                    pnode.state = node.state
                    pnode.user = node.user
                    found = True
                    break

                if not found and insert_new:
                    part.slurm_nodes.append(node)

    def is_job_running(self, job: BaseJob, retry_threshold: int = 3) -> bool:
        """
        Determine if a specified Slurm job is currently running by checking its presence and state in the job queue.

        This method queries the Slurm job accounting using 'sacct' to identify if the job with the specified ID is
        running. It handles transient network or system errors by retrying the query a limited number of times.

        Args:
            job (BaseJob): The job to check.
            retry_threshold (int): Maximum number of retry attempts for the query in case of transient errors.

        Returns:
            bool: True if the job is currently running, False otherwise.

        Raises:
            RuntimeError: If an error occurs that prevents determination of the job's running status, or if the status
                        cannot be determined after the specified number of retries.
        """
        retry_count = 0
        command = f"sacct -j {job.id} --format=State --noheader"

        while retry_count < retry_threshold:
            stdout, stderr = self.cmd_shell.execute(command).communicate()
            logging.debug(f"Job running: {command=} {stdout=} {stderr=}")

            if "Socket timed out" in stderr or "slurm_load_jobs error" in stderr:
                retry_count += 1
                logging.warning(
                    f"An error occurred while querying the job status. Retrying... ({retry_count}/{retry_threshold})."
                )
                continue

            if stderr:
                error_message = f"Error checking job status: {stderr}"
                logging.error(error_message)
                raise RuntimeError(error_message)

            job_states = stdout.strip().split()
            if "RUNNING" in job_states:
                return True

            break

        if retry_count == retry_threshold:
            error_message = f"Failed to confirm job running status after {retry_threshold} attempts."
            logging.error(error_message)
            raise RuntimeError(error_message)

        return False

    def is_job_completed(self, job: BaseJob, retry_threshold: int = 3) -> bool:
        """
        Check if a Slurm job is completed by querying its status.

        Retries the query a specified number of times if certain errors are encountered.

        Args:
            job (BaseJob): The job to check.
            retry_threshold (int): Maximum number of retries for transient errors.

        Returns:
            bool: True if the job is completed, False otherwise.

        Raises:
            RuntimeError: If unable to determine job status after retries, or if a non-retryable error is encountered.
        """
        retry_count = 0
        command = f"sacct -j {job.id} --format=State --noheader"

        while retry_count < retry_threshold:
            stdout, stderr = self.cmd_shell.execute(command).communicate()
            logging.debug(f"Job completed: {command=} {stdout=} {stderr=}")

            if "Socket timed out" in stderr or "slurm_load_jobs error" in stderr:
                retry_count += 1
                logging.warning(f"Retrying job status check (attempt {retry_count}/{retry_threshold})")
                continue

            if stderr:
                error_message = f"Error checking job status: {stderr}"
                logging.error(error_message)
                raise RuntimeError(error_message)

            job_states = stdout.strip().split()
            if "RUNNING" in job_states:
                return False

            if any(state in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "CANCELLED+"] for state in job_states):
                return True

            break

        if retry_count == retry_threshold:
            error_message = f"Failed to confirm job completion status after {retry_threshold} attempts."
            logging.error(error_message)
            raise RuntimeError(error_message)

        return False

    def get_job_status(self, job: BaseJob, retry_threshold: int = 3) -> list[SlurmStepMetadata]:
        retry_count = 0
        command = (
            f"sacct -j {job.id} --format=JobID,JobName,State,ExitCode,Start,End,ElapsedRAW,SubmitLine "
            "--delimiter='|' -p --noheader"
        )

        while retry_count < retry_threshold:
            stdout, stderr = self.cmd_shell.execute(command).communicate()
            logging.debug(f"Job status: {command=} {stdout=} {stderr=}")

            if "Socket timed out" in stderr or "slurm_load_jobs error" in stderr:
                retry_count += 1
                logging.warning(f"Retrying job status check (attempt {retry_count}/{retry_threshold})")
                continue

            if stderr:
                error_message = f"Error checking job status: {stderr}"
                logging.error(error_message)
                raise RuntimeError(error_message)

            return SlurmStepMetadata.from_sacct_output(stdout, delimiter="|")

        return []

    def kill(self, job: BaseJob) -> None:
        """
        Terminate a Slurm job.

        Args:
            job (BaseJob): The job to be terminated.
        """
        assert isinstance(job.id, int)
        self.scancel(job.id)

    @classmethod
    def format_node_list(cls, node_names: List[str]) -> str:
        """
        Format a list of node names into a condensed string representing groups of nodes as ranges.

        Mimicking the compact display found in systems like Slurm's sinfo command output.

        Args:
            node_names: A list of node names, potentially including numerically sequential nodes that can be condensed
                into a range format.

        Returns:
            A string representing the condensed node list, with numerically adjacent nodes shown as ranges.
        """

        def extract_parts(name: str) -> tuple:
            """
            Extract the prefix and numeric part of a node name, along with the length of the numeric part.

            Zero-padding is used.

            Args:
                name: The node name to be parsed.

            Returns:
                A tuple containing the prefix (str), numeric part (int), and
                the length of the numeric part (int).
            """
            match = re.match(r"^(.*?-)(\d+)$", name)
            if not match:
                raise ValueError(f"Cannot extract numeric part from '{name}'")
            prefix, num = match.groups()
            return prefix, int(num), len(num)

        def format_range(lst: List[int], padding: int) -> List[str]:
            """
            Format a list of integers into string ranges, considering zero-padding.

            Args:
                lst: A sorted list of node numbers.
                padding: The number of digits for zero-padding the node numbers.

            Returns:
                A list of formatted string ranges.
            """
            if not lst:
                return []
            lst.sort()
            start = lst[0]
            end = lst[0]
            ranges = []
            for num in lst[1:]:
                if num == end + 1:
                    end = num
                else:
                    range_str = f"{start:0{padding}d}-{end:0{padding}d}" if start != end else f"{start:0{padding}d}"
                    ranges.append(range_str)
                    start = end = num
            range_str = f"{start:0{padding}d}-{end:0{padding}d}" if start != end else f"{start:0{padding}d}"
            ranges.append(range_str)
            return ranges

        nodes_by_prefix = {}
        for name in node_names:
            prefix, num, length = extract_parts(name)
            nodes_by_prefix.setdefault(prefix, {"nums": [], "padding": 0})
            nodes_by_prefix[prefix]["nums"].append(num)
            nodes_by_prefix[prefix]["padding"] = max(nodes_by_prefix[prefix]["padding"], length)

        formatted_ranges = []
        for prefix, details in nodes_by_prefix.items():
            ranges = format_range(details["nums"], details["padding"])
            range_str = f"[{','.join(ranges)}]" if ranges else ""
            formatted_ranges.append(f"{prefix}{range_str}")

        return ", ".join(formatted_ranges)

    def get_available_nodes_from_group(
        self, partition_name: str, group_name: str, number_of_nodes: Union[int, str]
    ) -> List[SlurmNode]:
        """
        Retrieve a specific number of potentially available nodes from a group within a partition.

        Prioritizes nodes by their current state, preferring idle nodes first, then completing nodes, and finally
        allocated nodes, while excluding nodes that are down and allocated nodes to the current user.

        Args:
            partition_name (str): The name of the partition.
            group_name (str): The name of the group.
            number_of_nodes (Union[int,str]): The number of nodes to retrieve.
                Could also be 'all' to retrieve all the nodes from the group.

        Returns:
            List[SlurmNode]: Objects that are potentially available for use.

        Raises:
            ValueError: If the partition or group is not found, or if the requested number of nodes exceeds the
                available nodes.
        """
        self.update()

        self.validate_partition_and_group(partition_name, group_name)

        grouped_nodes = self.group_nodes_by_state(partition_name, group_name)

        try:
            allocated_nodes = self.allocate_nodes(grouped_nodes, number_of_nodes, group_name)

            logging.info(
                f"Allocated nodes from group '{group_name}' in partition '{partition_name}': "
                f"{[node.name for node in allocated_nodes]}"
            )

            return allocated_nodes

        except ValueError as e:
            logging.error(
                f"Error occurred while allocating nodes from group '{group_name}' in partition '{partition_name}': {e}",
                exc_info=True,
            )

            return []

    def validate_partition_and_group(self, partition_name: str, group_name: str) -> None:
        """
        Validate that the partition and group exist.

        Args:
            partition_name (str): The name of the partition.
            group_name (str): The name of the group.

        Raises:
            ValueError: If the partition or group is not found.

        """
        if partition_name not in self.groups:
            raise ValueError(f"Partition '{partition_name}' not found.")
        if group_name not in self.groups[partition_name]:
            raise ValueError(f"Group '{group_name}' not found in partition '{partition_name}'.")

    def group_nodes_by_state(self, partition_name: str, group_name: str) -> Dict[SlurmNodeState, List[SlurmNode]]:
        """
        Group nodes by their states, excluding nodes allocated to the current user.

        Args:
            partition_name (str): The name of the partition.
            group_name (str): The name of the group.
            current_user (str): The username of the current user.

        Returns:
            Dict[SlurmNodeState, List[SlurmNode]]: A dictionary grouping nodes by their state.
        """
        grouped_nodes = {
            SlurmNodeState.IDLE: [],
            SlurmNodeState.COMPLETING: [],
            SlurmNodeState.ALLOCATED: [],
        }

        for node in self.groups[partition_name][group_name]:
            if node.state in grouped_nodes:
                grouped_nodes[node.state].append(node)

        logging.debug(f"Grouped nodes by state: {grouped_nodes}")

        return grouped_nodes

    def allocate_nodes(
        self, grouped_nodes: Dict[SlurmNodeState, List[SlurmNode]], number_of_nodes: Union[int, str], group_name: str
    ) -> List[SlurmNode]:
        """
        Allocate nodes based on the requested number or maximum availability.

        Args:
            grouped_nodes (Dict[SlurmNodeState, List[SlurmNode]]): Nodes grouped by their state.
            number_of_nodes (Union[int, str]): The number of nodes to allocate, or 'max_avail' to allocate
                all available nodes.
            group_name (str): The name of the group.

        Returns:
            List[SlurmNode]: A list of allocated nodes.

        Raises:
            ValueError: If the requested number of nodes exceeds the available nodes.
        """
        allocated_nodes = []

        if isinstance(number_of_nodes, str) and number_of_nodes == "max_avail":
            allocated_nodes.extend(grouped_nodes[SlurmNodeState.IDLE])
            allocated_nodes.extend(grouped_nodes[SlurmNodeState.COMPLETING])

            if len(allocated_nodes) == 0:
                raise ValueError(
                    f"CloudAI is requesting the maximum available nodes from the group '{group_name}', "
                    f"but no nodes are available. Please review the available nodes in the system and ensure "
                    f"there are sufficient resources to meet the requirements of the test scenario. Additionally, "
                    f"verify that the system is capable of hosting the maximum number of nodes specified in the test "
                    "scenario."
                )

        elif isinstance(number_of_nodes, int):
            for state in grouped_nodes:
                while grouped_nodes[state] and len(allocated_nodes) < number_of_nodes:
                    allocated_nodes.append(grouped_nodes[state].pop(0))

            if len(allocated_nodes) < number_of_nodes:
                raise ValueError(
                    f"CloudAI is requesting {number_of_nodes} nodes from the group '{group_name}', but only "
                    f"{len(allocated_nodes)} nodes are available. Please review the available nodes in the system "
                    f"and ensure there are enough resources to meet the requested node count. Additionally, "
                    f"verify that the system can accommodate the number of nodes required by the test scenario."
                )
        else:
            raise ValueError(
                f"The 'number_of_nodes' argument must be either an integer specifying the number of nodes to allocate,"
                f" or 'max_avail' to allocate all available nodes. Received: '{number_of_nodes}'. "
                "Please correct the input."
            )

        return allocated_nodes

    def scancel(self, job_id: int) -> None:
        """
        Terminates a specified Slurm job by sending a cancellation command.

        Args:
            job_id (int): The ID of the job to cancel.
        """
        self.cmd_shell.execute(f"scancel {job_id}")

    def fetch_command_output(self, command: str) -> Tuple[str, str]:
        """
        Execute a system command and return its output.

        Args:
            command (str): The command to execute.

        Returns:
            Tuple[str, str]: The stdout and stderr from the command execution.
        """
        logging.debug(f"Executing command: {command}")
        stdout, stderr = self.cmd_shell.execute(command).communicate()
        if stderr:
            logging.error(f"Error executing command '{command}': {stderr}")
        return stdout, stderr

    def convert_state_to_enum(self, state_str: str) -> SlurmNodeState:
        """
        Convert a Slurm node state string to its corresponding enum member.

        Handles both full state names and abbreviated forms. Special handling for states ending with "*", indicating a
        non-responding node. If the state cannot be matched, UNKNOWN_STATE is returned.

        Args:
            state_str (str): State string from Slurm, could be full name, abbreviated code, or with a "*" suffix.

        Returns:
            SlurmNodeState: Corresponding enum member, or UNKNOWN_STATE for unmatched states, NOT_RESPONDING for "*"
                suffix.

        Raises:
            ValueError: If state_str is not a non-empty string.
        """
        if not isinstance(state_str, str) or not state_str:
            raise ValueError("state_str must be a non-empty string")

        # Mapping of abbreviated states to enum members
        state_abbreviations = {
            "alloc": SlurmNodeState.ALLOCATED,
            "comp": SlurmNodeState.COMPLETING,
            "down": SlurmNodeState.DOWN,
            "drain": SlurmNodeState.DRAINED,
            "drng": SlurmNodeState.DRAINING,
            "fail": SlurmNodeState.FAIL,
            "failg": SlurmNodeState.FAILING,
            "futr": SlurmNodeState.FUTURE,
            "idle": SlurmNodeState.IDLE,
            "maint": SlurmNodeState.MAINTENANCE,
            "mix": SlurmNodeState.MIXED_ALLOCATION,
            "npc": SlurmNodeState.USING_NETWORK_PERFORMANCE_COUNTERS,
            "plnd": SlurmNodeState.PLANNED_STATE,
            "pow_dn": SlurmNodeState.PENDING_POWER_DOWN_STATE,
            "pow_up": SlurmNodeState.POWERING_UP_STATE,
            "resv": SlurmNodeState.RESERVED,
            "unk": SlurmNodeState.UNKNOWN_STATE,
        }

        core_state = state_str.split()[0].upper().rstrip("*+~#!%$@^-")

        if state_str.endswith("*"):
            return SlurmNodeState.NOT_RESPONDING

        try:
            return SlurmNodeState(core_state)
        except ValueError:
            abbrev = core_state.lower()
            if abbrev in state_abbreviations:
                return state_abbreviations[abbrev]
            else:
                logging.warning(f"Unknown state: {core_state}")
                return SlurmNodeState.UNKNOWN_STATE

    def parse_nodes(self, nodes: List[str]) -> List[str]:
        """
        Parse a list of node specifications into individual node names.

        Supports explicit node names and specifications in "partition:group:num_nodes" format, and also handles ranges
        in node names. This allows for dynamic node allocation based on system state and compact node list
        specifications.

        Args:
            nodes (List[str]): A list containing node names or specifications. Specifications should follow
                "partition:group:num_nodes", where "partition" is the partition name, "group" is a group within that
                partition, and "num_nodes" is the number of nodes requested. Node ranges should be specified with
                square brackets and dashes, e.g., "node[01-03]" for "node01", "node02", "node03".

        Returns:
            List[str]: A list of node names. For specifications, it includes names of allocated nodes based on the
                specification, without duplicates. Node ranges are expanded into individual node names.

        Raises:
            ValueError: If a specification is malformed, a specified node is not found, or a node range cannot be
                parsed. This ensures users are aware of incorrect inputs.
        """
        parsed_nodes = []
        for node_spec in nodes:
            if ":" in node_spec:
                parts = node_spec.split(":")
                if len(parts) != 3:
                    raise ValueError("Format should be partition:group:num_nodes")
                partition_name, group_name, num_nodes_spec = parts
                num_nodes = int(num_nodes_spec) if num_nodes_spec != "max_avail" else num_nodes_spec
                group_nodes = self.get_available_nodes_from_group(partition_name, group_name, num_nodes)
                parsed_nodes += [node.name for node in group_nodes]
            else:
                expanded_nodes = parse_node_list(node_spec)
                parsed_nodes += expanded_nodes

        # Remove duplicates while preserving order
        parsed_nodes = list(dict.fromkeys(parsed_nodes))
        return parsed_nodes

    def get_nodes_by_spec(self, num_nodes: int, nodes: list[str]) -> Tuple[int, list[str]]:
        """
        Retrieve a list of node names based on specifications.

        When nodes is empty, returns `(num_nodes, [])`, otherwise parses the node specifications and returns the number
        of nodes and a list of node names.

        Args:
            num_nodes (int): The number of nodes, can't be `0`.
            nodes (list[str]): A list of node names specifications, slurm format or `PARTITION:GROUP:NUM_NODES`.

        Returns:
            Tuple[int, list[str]]: The number of nodes and a list of node names.
        """
        num_nodes, node_list = num_nodes, []
        parsed_nodes = self.parse_nodes(nodes)
        if parsed_nodes:
            num_nodes = len(parsed_nodes)
            node_list = parsed_nodes
        return num_nodes, node_list

    def system_installables(self) -> list[Installable]:
        return [File(Path(__file__).parent.absolute() / "slurm-metadata.sh")]
