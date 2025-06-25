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

from unittest.mock import Mock, patch

import pytest

from cloudai.systems.slurm import (
    SlurmGroup,
    SlurmJob,
    SlurmNode,
    SlurmNodeState,
    SlurmPartition,
    SlurmSystem,
    parse_node_list,
)


class TestGroupAllocation:
    def prepare(
        self, slurm_system: SlurmSystem, taken_node_names: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> tuple[SlurmSystem, list[SlurmNode], list[SlurmNode]]:
        slurm_system.partitions = [
            SlurmPartition(name="main", groups=[SlurmGroup(name="group1", nodes=["node0[1-5]"])])
        ]
        all_nodes = [
            SlurmNode(name=name, partition="main", state=SlurmNodeState.IDLE)
            for name in parse_node_list(slurm_system.partitions[0].groups[0].nodes[0])
        ]
        taken_nodes = [
            SlurmNode(name=node.name, partition="main", state=SlurmNodeState.ALLOCATED)
            for node in all_nodes
            if node.name in taken_node_names
        ]

        mod_path = "cloudai.systems.slurm.slurm_system.SlurmSystem"
        monkeypatch.setattr(f"{mod_path}.nodes_from_sinfo", lambda *args, **kwargs: all_nodes)
        monkeypatch.setattr(f"{mod_path}.nodes_from_squeue", lambda *args, **kwargs: taken_nodes)
        return slurm_system, all_nodes, taken_nodes

    def test_all_nodes_in_group_are_idle(self, slurm_system: SlurmSystem, monkeypatch: pytest.MonkeyPatch):
        system, *_ = self.prepare(slurm_system, [], monkeypatch)
        nnodes, nodes_list = system.get_nodes_by_spec(1, ["main:group1:5"])
        assert nodes_list == parse_node_list(slurm_system.partitions[0].groups[0].nodes[0])
        assert nnodes == len(nodes_list)

    def test_enough_free_nodes_for_allocation(self, slurm_system: SlurmSystem, monkeypatch: pytest.MonkeyPatch):
        system, all_nodes, taken_nodes = self.prepare(slurm_system, ["node01", "node02"], monkeypatch)
        nnodes, nodes_list = system.get_nodes_by_spec(1, ["main:group1:3"])
        assert nnodes == 3
        assert nodes_list == sorted([n.name for n in set(all_nodes) - set(taken_nodes)])

    def test_not_enough_nodes_for_allocation(self, slurm_system: SlurmSystem, monkeypatch: pytest.MonkeyPatch):
        """In this scenario we still return required number of nodes to put job into the queue"""
        system, all_nodes, _ = self.prepare(slurm_system, ["node01", "node02"], monkeypatch)
        nnodes, nodes_list = system.get_nodes_by_spec(1, ["main:group1:5"])
        assert nnodes == 5
        assert nodes_list == sorted([n.name for n in all_nodes])

    def test_two_cases_one_group(self, slurm_system: SlurmSystem, monkeypatch: pytest.MonkeyPatch):
        # system has 5 nodes in the group
        system, *_ = self.prepare(slurm_system, [], monkeypatch)

        # first case asks for 2 nodes
        nnodes, nodes_list1 = system.get_nodes_by_spec(1, ["main:group1:2"])
        assert nnodes == 2

        # second case asks for another 2 nodes
        nnodes, nodes_list2 = system.get_nodes_by_spec(1, ["main:group1:2"])
        assert nnodes == 2

        assert nodes_list1 != nodes_list2, "Same nodes were allocated for two different requests"

    def test_completion_clears_group_allocation_state(self, slurm_system: SlurmSystem, monkeypatch: pytest.MonkeyPatch):
        system, all_nodes, taken_nodes = self.prepare(slurm_system, ["node01", "node02"], monkeypatch)
        system.group_allocated.clear()
        _, nodes_list = system.get_nodes_by_spec(1, ["main:group1:3"])
        assert system.group_allocated == set(all_nodes) - set(taken_nodes)
        assert all(node.state == SlurmNodeState.ALLOCATED for node in system.group_allocated)

        with patch(
            "cloudai.systems.slurm.slurm_system.SlurmSystem.fetch_command_output",
            return_value=(f"{','.join(nodes_list)}|", ""),
        ):
            system.complete_job(SlurmJob(id=1, test_run=Mock()))

        assert len(system.group_allocated) == 0

    def test_group_allocation_is_preserved_on_updated(self, slurm_system: SlurmSystem, monkeypatch: pytest.MonkeyPatch):
        system, all_nodes, _ = self.prepare(slurm_system, [], monkeypatch)
        system.group_allocated.clear()
        _ = system.get_nodes_by_spec(1, ["main:group1:5"])
        assert system.group_allocated == set(all_nodes)
        assert all(node.state == SlurmNodeState.ALLOCATED for node in system.group_allocated)

        # Simulate scenario when sinfo still reports group allocated nodes as idle
        with patch(
            "cloudai.systems.slurm.slurm_system.SlurmSystem.nodes_from_sinfo",
            return_value=[
                SlurmNode(name=node.name, partition=node.partition, state=SlurmNodeState.IDLE) for node in all_nodes
            ],
        ):
            system.update()
        assert all(node.state == SlurmNodeState.ALLOCATED for node in system.group_allocated)
