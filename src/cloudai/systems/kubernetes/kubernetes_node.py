# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from enum import Enum


class KubernetesNodeState(Enum):
    """
    Enumeration of possible states for a Kubernetes compute node, as defined by the Kubernetes workload manager.

    Each state represents the current status or condition of a node within a Kubernetes cluster, impacting its availability
    for job allocation and execution.

    - NOT_RESPONDING: Node is not responding.
    - POWERED_OFF: Node is powered off.
    - BEING_POWERED_UP_OR_CONFIGURED: Node is in the process of being powered up or configured.
    - PENDING_POWER_DOWN: Node is pending power down.
    - BEING_POWERED_DOWN: Node is being powered down.
    - IN_MAINTENANCE_RESERVATION: Node is in a maintenance reservation.
    - PENDING_REBOOT: Node is pending reboot.
    - REBOOT_ISSUED: Node reboot has been issued.
    - PLANNED_FOR_HIGHER_PRIORITY_JOB: Node is planned for a higher priority job.
    - ALLOCATED: Node is allocated to one or more jobs.
    - ALLOCATED_COMPLETING: Node is allocated to jobs that are completing.
    - COMPLETING: Node is completing all jobs.
    - DOWN: Node is down and unavailable for use.
    - DRAINED: Node is drained and unavailable for use.
    - DRAINING: Node is draining; it is allocated a job but will not be allocated additional jobs.
    - FAIL: Node is expected to fail soon.
    - FAILING: Node is failing.
    - FUTURE: Node is not fully configured but expected to be available in the future.
    - IDLE: Node is idle and available for use.
    - INVALID_REGISTRATION: Node did not register correctly.
    - MAINTENANCE: Node is in maintenance.
    - REBOOT_ISSUED_STATE: A reboot request has been issued for the node.
    - REBOOT_REQUESTED: A reboot request has been made for the node.
    - MIXED_ALLOCATION: Node has mixed allocation.
    - USING_NETWORK_PERFORMANCE_COUNTERS: Node is using network performance counters.
    - PLANNED_STATE: Node is planned for future allocation.
    - PENDING_POWER_DOWN_STATE: Node is pending power down.
    - POWERED_DOWN_STATE: Node is powered down.
    - POWERING_DOWN_STATE: Node is powering down.
    - POWERING_UP_STATE: Node is powering up.
    - RESERVED: Node is reserved.
    - UNKNOWN_STATE: Node state is unknown.

    For more details, visit https://kubernetes.schedmd.com/sinfo.html.
    """

    NOT_RESPONDING = "*"
    POWERED_OFF = "~"
    BEING_POWERED_UP_OR_CONFIGURED = "#"
    PENDING_POWER_DOWN = "!"
    BEING_POWERED_DOWN = "%"
    IN_MAINTENANCE_RESERVATION = "$"
    PENDING_REBOOT = "@"
    REBOOT_ISSUED = "^"
    PLANNED_FOR_HIGHER_PRIORITY_JOB = "-"
    ALLOCATED = "ALLOCATED"
    ALLOCATED_COMPLETING = "ALLOCATED+"
    COMPLETING = "COMPLETING"
    DOWN = "DOWN"
    DRAINED = "DRAINED"
    DRAINING = "DRAINING"
    FAIL = "FAIL"
    FAILING = "FAILING"
    FUTURE = "FUTURE"
    IDLE = "IDLE"
    INVALID_REGISTRATION = "INVAL"
    MAINTENANCE = "MAINT"
    REBOOT_ISSUED_STATE = "REBOOT_ISSUED"
    REBOOT_REQUESTED = "REBOOT_REQUESTED"
    MIXED_ALLOCATION = "MIXED"
    USING_NETWORK_PERFORMANCE_COUNTERS = "PERFCTRS"
    PLANNED_STATE = "PLANNED"
    PENDING_POWER_DOWN_STATE = "POWER_DOWN"
    POWERED_DOWN_STATE = "POWERED_DOWN"
    POWERING_DOWN_STATE = "POWERING_DOWN"
    POWERING_UP_STATE = "POWERING_UP"
    RESERVED = "RESERVED"
    UNKNOWN_STATE = "UNKNOWN"


class KubernetesNode:
    """
    Represents a Kubernetes compute node with detailed state and partition info.

    Attributes
        name (str): The name of the node.
        partition (str): The partition to which the node belongs.
        state (KubernetesNodeState): The current state of the node.
        user (str): The name of the user currently using the node. Defaults to N/A if the node is not being used.
    """

    def __init__(
        self,
        name: str,
        partition: str,
        state: KubernetesNodeState,
        user: str = "N/A",
    ) -> None:
        self.name = name
        self.partition = partition
        self.state = state
        self.user = user

    def allocatable(self, free_only: bool = True) -> bool:
        """
        Determine if the node is allocatable based on its state.

        Args:
            free_only (bool): If True, considers only the IDLE state as allocatable. Otherwise, considers various
                states based on Kubernetes's allocation logic, including states where a node may be partially used or in a
                transitional state that does not preclude future allocations.

        Returns:
            bool: True if the node is allocatable, False otherwise.
        """
        if free_only:
            return self.state == KubernetesNodeState.IDLE
        else:
            return self.state in [
                KubernetesNodeState.ALLOCATED,
                KubernetesNodeState.ALLOCATED_COMPLETING,
                KubernetesNodeState.COMPLETING,
                KubernetesNodeState.IDLE,
                KubernetesNodeState.MAINTENANCE,
                KubernetesNodeState.MIXED_ALLOCATION,
                KubernetesNodeState.PLANNED_STATE,
                KubernetesNodeState.POWERING_UP_STATE,
                KubernetesNodeState.RESERVED,
            ]

    def __repr__(self) -> str:
        """
        Provide a structured string representation of the Kubernetes node, including its name, state, and partition.

        Returns
            str: A string representation of the Kubernetes node.
        """
        return (
            f"KubernetesNode(name={self.name}, "
            f"partition={self.partition}, "
            f"state={self.state.name}, "
            f"user={self.user})"
        )
