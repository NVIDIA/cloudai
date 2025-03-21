from typing import Dict, List, Union

from cloudai import TestRun
from cloudai.systems.lsf.strategy import LSFCommandGenStrategy


class SleepLSFCommandGenStrategy(LSFCommandGenStrategy):
    """Command generation strategy for Sleep on LSF systems."""

    def _container_mounts(self, tr: TestRun) -> list[str]:
        return []

    def generate_test_command(
        self, env_vars: Dict[str, Union[str, List[str]]], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> List[str]:
        return [f'sleep {cmd_args["seconds"]}']

    def gen_srun_command(self, tr: TestRun) -> str:
        """
        Generate the LSF bsub command for a test based on the given parameters.

        Args:
            tr (TestRun): Contains the test and its run-specific configurations.

        Returns:
            str: The generated LSF bsub command.
        """
        job_name = f"sleep_{tr.name}"
        time_limit = tr.time_limit or "1:00"  # Default time limit if not specified
        num_nodes = tr.num_nodes or 1  # Default to 1 node if not specified

        bsub_command = [
            "bsub",
            f"-J {job_name}",
            f"-W {time_limit}",
            f"-n {num_nodes}",
            f'sleep {tr.test.cmd_args["seconds"]}',
        ]

        return " ".join(bsub_command)
