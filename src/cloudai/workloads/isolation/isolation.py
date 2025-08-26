from typing import Literal

from cloudai.core import CmdArgs, TestDefinition
from cloudai.workloads.nccl_test import NCCLTestDefinition
from cloudai.workloads.nemo_run import NeMoRunTestDefinition


class IsolationCmdArgs(CmdArgs):
    """Command line arguments for the Isolation workload."""

    nodes_split_rule: Literal["even"]
    main_job: NeMoRunTestDefinition
    noise_job: NCCLTestDefinition


class IsolationTestDefinition(TestDefinition):
    """Test definition for the Isolation workload."""

    cmd_args: IsolationCmdArgs
