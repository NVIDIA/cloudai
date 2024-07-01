#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import getpass
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from cloudai import System
from cloudai.util import CommandShell

from .slurm_node import SlurmNode, SlurmNodeState


class SlurmSystem(System):
    """
    Represents a Slurm system, encapsulating the system's configuration.

    Attributes
        name (str): The name of the Slurm system.
        install_path (str): Installation path of CloudAI software.
        output_path (str): Directory path for output files.
        default_partition (str): The default partition for job submission.
        partitions (Dict[str, List[SlurmNode]]): Mapping of partition names to lists of SlurmNodes.
        account (Optional[str]): Account name for charging resources used by this job.
        distribution (Optional[str]): Specifies alternate distribution methods for remote processes.
        gpus_per_node (Optional[int]): Specifies the number of GPUs available per node.
        ntasks_per_node (Optional[int]): Specifies the number of tasks that can run concurrently on a single node.
        cache_docker_images_locally (bool): Whether to cache Docker images locally for the Slurm system.
        groups (Dict[str, Dict[str, List[SlurmNode]]]): Nested mapping where the key is the partition name and the
            value is another dictionary with group names as keys and lists of SlurmNodes as values, representing the
            group composition within each partition.
        global_env_vars (Optional[Dict[str, Any]]): Dictionary containing additional configuration settings for the
            system.
        cmd_shell (CommandShell): An instance of CommandShell for executing system commands.
    """

    def update(self) -> None:
        """
        Update the system object for a SLURM system.

        This method updates the system object by querying the current state of each node using the 'sinfo' and 'squeue'
        commands, and correlating this information to determine the state of each node and the user running jobs on
        each node.
        """
        self.update_node_states()

    @classmethod
    def parse_node_list(cls, node_list: str) -> List[str]:
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
                            [
                                f"{header}{str(i).zfill(number_of_digits)}"
                                for i in range(int(start_node), int(end_node) + 1)
                            ]
                        )
                    else:
                        nodes.append(f"{header}{r}")

        return nodes

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

    def __init__(
        self,
        name: str,
        install_path: str,
        output_path: str,
        default_partition: str,
        partitions: Dict[str, List[SlurmNode]],
        account: Optional[str] = None,
        distribution: Optional[str] = None,
        gpus_per_node: Optional[int] = None,
        ntasks_per_node: Optional[int] = None,
        cache_docker_images_locally: bool = False,
        groups: Optional[Dict[str, Dict[str, List[SlurmNode]]]] = None,
        global_env_vars: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a SlurmSystem instance.

        Args:
            name (str): Name of the Slurm system.
            install_path (str): The installation path of CloudAI.
            output_path (str): Path to the output directory.
            default_partition (str): Default partition.
            partitions (Dict[str, List[SlurmNode]]): Partitions in the system.
            account (Optional[str]): Account name for charging resources used by this job.
            distribution (Optional[str]): Specifies alternate distribution methods for remote processes.
            gpus_per_node (Optional[int]): Specifies the number of GPUs available per node.
            ntasks_per_node (Optional[int]): Specifies the number of tasks that can run concurrently on a single node.
            cache_docker_images_locally (bool): Whether to cache Docker images locally for the Slurm system.
            groups (Optional[Dict[str, Dict[str, List[SlurmNode]]]]): Nested mapping of group names to lists of
                SlurmNodes within partitions, defining the group composition within each partition. Defaults to an
                empty dictionary if not provided.
            global_env_vars (Optional[Dict[str, Any]]): Dictionary containing additional configuration settings for
                the system.
        """
        super().__init__(name, "slurm", output_path)
        self.install_path = install_path
        self.default_partition = default_partition
        self.partitions = partitions
        self.account = account
        self.distribution = distribution
        self.gpus_per_node = gpus_per_node
        self.ntasks_per_node = ntasks_per_node
        self.cache_docker_images_locally = cache_docker_images_locally
        self.groups = groups if groups is not None else {}
        self.global_env_vars = global_env_vars if global_env_vars is not None else {}
        self.cmd_shell = CommandShell()
        logging.debug(f"{self.__class__.__name__} initialized")

    def __repr__(self) -> str:
        """
        Provide a structured string representation of the system.

        Including the system name, scheduler type, and a simplified view similar to the `sinfo` command output,
        focusing on the partition, state, and nodelist.
        """
        header = f"System Name: {self.name}\nScheduler Type: {self.scheduler}"
        parts = [header, "\tPARTITION  STATE    NODELIST"]
        for partition_name, nodes in self.partitions.items():
            state_count = {}
            for node in nodes:
                state_count.setdefault(node.state, []).append(node.name)
            for state, names in state_count.items():
                node_list_str = self.format_node_list(names)
                parts.append(f"\t{partition_name:<10} {state.name:<7} {node_list_str}")
        return "\n".join(parts)

    def get_partition_names(self) -> List[str]:
        """Return a list of all partition names."""
        return list(self.partitions.keys())

    def get_partition_nodes(self, partition_name: str) -> List[SlurmNode]:
        """
        Return a list of SlurmNode objects in the specified partition.

        Args:
            partition_name (str): The name of the partition.

        Returns:
            List[SlurmNode]: Nodes belonging to the specified partition.

        Raises:
            ValueError: If the partition does not exist.
        """
        if partition_name not in self.partitions:
            raise ValueError(f"Partition '{partition_name}' not found.")
        return self.partitions[partition_name]

    def get_partition_node_names(self, partition_name: str) -> List[str]:
        """
        Return the names of all nodes within a specified partition.

        Args:
            partition_name (str): The name of the partition.

        Returns:
            List[str]: Names of nodes within the specified partition.
        """
        return [node.name for node in self.get_partition_nodes(partition_name)]

    def get_group_names(self, partition_name: str) -> List[str]:
        """
        Retrieve names of all groups within a specified partition.

        Args:
            partition_name (str): The partition to query.

        Returns:
            List[str]: A list of group names within the specified partition.

        Raises:
            ValueError: If the partition is not found.
        """
        if partition_name not in self.groups:
            raise ValueError(f"Partition '{partition_name}' not found.")
        return list(self.groups[partition_name].keys())

    def get_group_nodes(self, partition_name: str, group_name: str) -> List[SlurmNode]:
        """
        Return a list of SlurmNode objects in the specified group within a partition.

        Args:
            partition_name (str): The name of the partition.
            group_name (str): The name of the group.

        Returns:
            List[SlurmNode]: Nodes belonging to the specified group within the partition.

        Raises:
            ValueError: If the partition or group does not exist.
        """
        if partition_name not in self.groups:
            raise ValueError(f"Partition '{partition_name}' not found.")
        if group_name not in self.groups[partition_name]:
            raise ValueError(f"Group '{group_name}' not found in partition '{partition_name}'.")
        return self.groups[partition_name][group_name]

    def get_group_node_names(self, partition_name: str, group_name: str) -> List[str]:
        """
        Return the names of all nodes within a specified group and partition.

        Args:
            partition_name (str): The name of the partition.
            group_name (str): The name of the group.

        Returns:
            List[str]: Names of nodes within the specified group and partition.

        Raises:
            ValueError: If the partition or group does not exist.
        """
        return [node.name for node in self.get_group_nodes(partition_name, group_name)]

    def get_available_nodes_from_group(
        self, partition_name: str, group_name: str, number_of_nodes: int
    ) -> List[SlurmNode]:
        """
        Retrieve a specific number of potentially available nodes from a group within a partition.

        Prioritizes nodes by their current state, preferring idle nodes first, then completing nodes, and finally
        allocated nodes, while excluding nodes that are down and allocated nodes to the current user.

        Args:
            partition_name (str): The name of the partition.
            group_name (str): The name of the group.
            number_of_nodes (int): The number of nodes to retrieve.

        Returns:
            List[SlurmNode]: Objects that are potentially available for use.

        Raises:
            ValueError: If the partition or group is not found, or if the requested number of nodes exceeds the
                available nodes.
        """
        if partition_name not in self.groups:
            raise ValueError(f"Partition '{partition_name}' not found.")
        if group_name not in self.groups[partition_name]:
            raise ValueError(f"Group '{group_name}' not found in partition '{partition_name}'.")

        current_user = getpass.getuser()
        self.update_node_states()

        # Group nodes by their states
        grouped_nodes = {
            SlurmNodeState.IDLE: [],
            SlurmNodeState.COMPLETING: [],
            SlurmNodeState.ALLOCATED: [],
        }

        for node in self.groups[partition_name][group_name]:
            if node.state in grouped_nodes:
                # Exclude nodes allocated to the current user
                if node.state == SlurmNodeState.ALLOCATED and node.user == current_user:
                    continue
                if node.state in grouped_nodes:
                    grouped_nodes[node.state].append(node)

        # Allocate nodes based on priority: idle, then completing, then allocated
        allocated_nodes = []
        for state in [
            SlurmNodeState.IDLE,
            SlurmNodeState.COMPLETING,
            SlurmNodeState.ALLOCATED,
        ]:
            while grouped_nodes[state] and len(allocated_nodes) < number_of_nodes:
                allocated_nodes.append(grouped_nodes[state].pop(0))

        if len(allocated_nodes) < number_of_nodes:
            raise ValueError(
                "Requested number of nodes ({}) exceeds the number of " "available nodes in group '{}'.".format(
                    number_of_nodes, group_name
                )
            )

        # Log allocation details
        logging.info(
            "Allocated nodes from group '{}' in partition '{}': {}".format(
                group_name,
                partition_name,
                [node.name for node in allocated_nodes],
            )
        )

        return allocated_nodes

    def is_node_in_system(self, node_name: str) -> bool:
        """
        Check if a given node is part of the Slurm system.

        Args:
            node_name (str): The name of the node to check.

        Returns:
            True if the node is part of the system, otherwise False.
        """
        return any(any(node.name == node_name for node in nodes) for nodes in self.partitions.values())

    def is_job_running(self, job_id: int, retry_threshold: int = 3) -> bool:
        """
        Determine if a specified Slurm job is currently running by checking its presence and state in the job queue.

        This method queries the Slurm job queue using 'squeue' to identify if the
        job with the specified ID is running. It handles transient network or
        system errors by retrying the query a limited number of times.

        Args:
            job_id (int): The ID of the job to check.
            retry_threshold (int): The maximum number of retry attempts for the
                                   query in case of transient errors.

        Returns:
            bool: True if the job is currently running (i.e., listed in the job
                  queue with a running state), False otherwise.

        Raises:
            RuntimeError: If an error occurs that prevents determination of the
                          job's running status, or if the status cannot be
                          determined after the specified number of retries.
        """
        retry_count = 0
        command = f"squeue -j {job_id} --noheader --format=%T"

        while retry_count < retry_threshold:
            logging.debug(f"Executing command to check job status: {command}")
            stdout, stderr = self.cmd_shell.execute(command).communicate()

            if "Socket timed out" in stderr or "slurm_load_jobs error" in stderr:
                retry_count += 1
                logging.warning(f"Transient error encountered. Retrying... " f"({retry_count}/{retry_threshold})")
                continue

            if stderr:
                raise RuntimeError(f"Error checking job status: {stderr}")

            job_state = stdout.strip()
            # If the job is listed with a "RUNNING" state, it's considered active
            if job_state == "RUNNING":
                return True

            # No need for further retries if we got a clear answer
            break

        if retry_count == retry_threshold:
            raise RuntimeError("Failed to confirm job running status after " f"{retry_threshold} attempts.")

        # Job is not active if not "RUNNING" or not found
        return False

    def is_job_completed(self, job_id: int, retry_threshold: int = 3) -> bool:
        """
        Check if a Slurm job is completed by querying its status.

        Retries the query a specified number of times if certain errors are encountered.

        Args:
            job_id (int): The ID of the job to check.
            retry_threshold (int): Maximum number of retries for transient errors.

        Returns:
            bool: True if the job is completed, False otherwise.

        Raises:
            RuntimeError: If unable to determine job status after retries, or if a non-retryable error is encountered.
        """
        retry_count = 0
        while retry_count < retry_threshold:
            command = f"squeue -j {job_id}"
            logging.debug(f"Checking job status with command: {command}")
            stdout, stderr = self.cmd_shell.execute(command).communicate()
            if "Socket timed out" in stderr:
                retry_count += 1
                logging.warning(f"Retrying job status check (attempt {retry_count}/" f"{retry_threshold})")
                continue

            if stderr:
                raise RuntimeError(f"Error checking job status: {stderr}")

            return str(job_id) not in stdout

        raise RuntimeError("Failed to confirm job completion status after " f"{retry_threshold} attempts.")

    def scancel(self, job_id: int) -> None:
        """
        Terminates a specified Slurm job by sending a cancellation command.

        Args:
            job_id (int): The ID of the job to cancel.
        """
        self.cmd_shell.execute(f"scancel {job_id}")

    def update_node_states(self) -> None:
        """
        Update the states of nodes in the Slurm system.

        By querying the current state of each node using the 'sinfo' command, and correlates this with 'squeue' to
        determine which user is running jobs on each node. This method parses the output of these commands, identifies
        the state of nodes and the users, and updates the corresponding SlurmNode instances in the system.
        """
        squeue_output = self.get_squeue()
        sinfo_output = self.get_sinfo()
        node_user_map = self.parse_squeue_output(squeue_output)
        self.parse_sinfo_output(sinfo_output, node_user_map)

    def get_squeue(self) -> str:
        """
        Fetch the output from the 'squeue' command.

        Returns
            str: The stdout from the 'squeue' command execution.
        """
        squeue_output, _ = self.fetch_command_output("squeue -o '%N|%u' --noheader")
        return squeue_output

    def get_sinfo(self) -> str:
        """
        Fetch the output from the 'sinfo' command.

        Returns
            str: The stdout from the 'sinfo' command execution.
        """
        sinfo_output, _ = self.fetch_command_output("sinfo")
        return sinfo_output

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

    def parse_squeue_output(self, squeue_output: str) -> Dict[str, str]:
        """
        Parse the output from the 'squeue' command to map nodes to users.

        The expected format of squeue_output is lines of 'node_spec|user', where node_spec can include comma-separated
        node names or ranges.

        Args:
            squeue_output (str): The raw output from the squeue command.

        Returns:
            Dict[str, str]: A dictionary mapping node names to usernames.
        """
        node_user_map = {}
        for line in squeue_output.split("\n"):
            if line.strip():
                # Split the line into node list and user, handling only the first '|'
                parts = line.split("|")
                if len(parts) < 2:
                    continue  # Skip malformed lines

                node_list_part, user = parts[0], "|".join(parts[1:])
                # Handle cases where multiple node groups or ranges are specified
                for node in self.parse_node_list(node_list_part):
                    node_user_map[node] = user.strip()

        return node_user_map

    def parse_sinfo_output(self, sinfo_output: str, node_user_map: Dict[str, str]) -> None:
        """
        Parse the output from the 'sinfo' command to update node states.

        Args:
            sinfo_output (str): The output from the sinfo command.
            node_user_map (dict): A dictionary mapping node names to users.
        """
        for line in sinfo_output.split("\n")[1:]:  # Skip the header line
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            partition, _, _, _, state, nodelist = parts[:6]
            partition = partition.rstrip("*")
            node_names = self.parse_node_list(nodelist)

            # Convert state to enum, handling states with suffixes
            state_enum = self.convert_state_to_enum(state)

            for node_name in node_names:
                # Find the partition and node to update the state
                for part_name, nodes in self.partitions.items():
                    if part_name != partition:
                        continue
                    for node in nodes:
                        if node.name == node_name:
                            node.state = state_enum
                            node.user = node_user_map.get(node_name, "N/A")
                            break

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
                partition_name, group_name, num_nodes_str = parts
                num_nodes = int(num_nodes_str)
                group_nodes = self.get_available_nodes_from_group(partition_name, group_name, num_nodes)
                parsed_nodes += [node.name for node in group_nodes]
            else:
                # Handle both individual node names and ranges
                if self.is_node_in_system(node_spec) or "[" in node_spec:
                    expanded_nodes = self.parse_node_list(node_spec)
                    parsed_nodes += expanded_nodes
                else:
                    raise ValueError(f"Node '{node_spec}' not found.")

        # Remove duplicates while preserving order
        parsed_nodes = list(dict.fromkeys(parsed_nodes))
        return parsed_nodes
