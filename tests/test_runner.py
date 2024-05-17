from unittest.mock import patch

from cloudai.runner.core import BaseRunner, Runner


def test_register_multiple_runners():
    """
    Test registering multiple different runners for the same type to ensure that
    only the last registered runner is kept.
    """
    with patch.dict("cloudai.runner.Runner._runners", clear=True):

        @Runner.register("slurm")
        class FirstSlurmRunner(BaseRunner):
            pass

        @Runner.register("slurm")
        class SecondSlurmRunner(BaseRunner):
            pass

        assert Runner._runners["slurm"] is SecondSlurmRunner
