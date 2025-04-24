from typing import Literal, Optional

from cloudai import CmdArgs, Installable, TestDefinition


class HealthyNodesCmdArgs(CmdArgs):
    """Command line arguments for the Healthy Nodes test."""

    min_healthy_nodes_percentage: int = 90
    round_strategy: Optional[Literal["power_of_2", "even"]] = "even"


class HealthyNodesTestDefinition(TestDefinition):
    """Test definition for the Healthy Nodes test."""

    cmd_args: HealthyNodesCmdArgs

    @property
    def installables(self) -> list[Installable]:
        return []
