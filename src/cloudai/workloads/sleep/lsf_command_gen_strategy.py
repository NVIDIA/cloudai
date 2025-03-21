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
