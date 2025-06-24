import pytest

from cloudai.systems.slurm import SlurmGroup, SlurmNode, SlurmNodeState, SlurmPartition, SlurmSystem, parse_node_list


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

    @pytest.mark.xfail(reason="This is a bug in the code, RM4471870")
    def test_two_cases_one_group(self, slurm_system: SlurmSystem, monkeypatch: pytest.MonkeyPatch):
        # system has 5 nodes in the group
        system, *_ = self.prepare(slurm_system, [], monkeypatch)

        # first case asks for 2 nodes
        nnodes, nodes_list1 = system.get_nodes_by_spec(1, ["main:group1:2"])
        assert nnodes == 2

        # second case asks for another 2 nodes
        nnodes, nodes_list2 = system.get_nodes_by_spec(1, ["main:group1:2"])
        assert nnodes == 2

        assert nodes_list1 != nodes_list2, "Same nodes we allocated for two different requests"
