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

from enum import Enum

from pydantic import BaseModel, ConfigDict


class LSFNodeState(Enum):
    """
    Enumeration of possible states for an LSF compute node, as defined by the LSF workload manager.

    Each state represents the current status or condition of a node within an LSF cluster, impacting its availability
    for job allocation and execution.
    """

    OK = "OK"  # Node is available and functioning normally.
    CLOSED = "CLOSED"  # Node is closed and not available for job allocation.
    UNREACHABLE = "UNREACHABLE"  # Node is unreachable.
    BUSY = "BUSY"  # Node is busy and cannot accept new jobs.
    ERROR = "ERROR"  # Node is in an error state.
    UNKNOWN = "UNKNOWN"  # Node state is unknown.
    DRAINED = "DRAINED"  # Node is drained and unavailable for use.
    DRAINING = "DRAINING"  # Node is in the process of being drained.
    RESERVED = "RESERVED"  # Node is reserved for specific jobs or users.
    POWERING_UP = "POWERING_UP"  # Node is in the process of powering up.
    POWERING_DOWN = "POWERING_DOWN"  # Node is in the process of powering down.
    MAINTENANCE = "MAINTENANCE"  # Node is under maintenance.


class LSFNode(BaseModel):
    """
    Represents an LSF compute node with detailed state and partition info.

    Attributes:
        name (str): The name of the node.
        queue (str): The queue to which the node belongs.
        state (LsfNodeState): The current state of the node.
        user (str): The name of the user currently using the node. Defaults to N/A if the node is not being used.
    """

    model_config = ConfigDict(extra="forbid")
    name: str
    queue: str
    state: LSFNodeState
    user: str = "N/A"

    def __repr__(self) -> str:
        """
        Provide a structured string representation of the LSF node, including its name, state, and queue.

        Returns:
            str: A string representation of the LSF node.
        """
        return f"LsfNode(name={self.name}, queue={self.queue}, state={self.state.name}, user={self.user})"
