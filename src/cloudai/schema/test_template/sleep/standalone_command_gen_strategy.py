# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Dict, List

from cloudai.schema.core.strategy import CommandGenStrategy, StrategyRegistry
from cloudai.schema.system import SlurmSystem, StandaloneSystem

from .template import Sleep


@StrategyRegistry.strategy(CommandGenStrategy, [StandaloneSystem, SlurmSystem], [Sleep])
class SleepStandaloneCommandGenStrategy(CommandGenStrategy):
    """
    Command generation strategy for the Sleep test on standalone systems.

    This strategy generates a command to execute a sleep operation with
    specified duration on standalone systems.
    """

    def gen_exec_command(
        self,
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        extra_env_vars: Dict[str, str],
        extra_cmd_args: str,
        output_path: str,
        nodes: List[str],
    ) -> str:
        """
        Generate the execution command for a sleep test.

        Args:
            env_vars (Dict[str, str]): Environment variables for the test.
            cmd_args (Dict[str, str]): Command-line arguments for the test.
            extra_env_vars (Dict[str, str]): Extra environment variables.
            extra_cmd_args (str): Extra command-line arguments.
            output_path (str): Path to the output directory.
            nodes (List[str]): A list of nodes where the test will be executed.
                This can be an empty list if node information is not applicable.

        Returns:
            str: The generated execution command.
        """
        if not nodes:
            nodes = []
        self.final_cmd_args = self._override_cmd_args(self.default_cmd_args, cmd_args)
        sec = self.final_cmd_args["seconds"]
        return f"sleep {sec}"
