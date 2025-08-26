from copy import deepcopy
from typing import List, cast

from cloudai.core import Registry, TestDefinition
from cloudai.systems.slurm.slurm_command_gen_strategy import SlurmCommandGenStrategy
from cloudai.workloads.isolation.isolation import IsolationTestDefinition


class IsolationSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for Isolation workload on Slurm systems."""

    def _container_mounts(self) -> List[str]:
        return []

    def cmd_gen_strategy(self, tdef: TestDefinition) -> SlurmCommandGenStrategy:
        strategy_cls = Registry().get_command_gen_strategy(type(self.system), type(tdef))
        tr = deepcopy(self.test_run)
        tr.test.test_definition = tdef
        strategy = cast(SlurmCommandGenStrategy, strategy_cls(self.system, tr))
        return strategy

    def _gen_srun_command(self) -> str:
        srun_parts = self.gen_srun_prefix(use_pretest_extras=True)
        isolation = cast(IsolationTestDefinition, self.test_run.test.test_definition)
        noise_job_cmd = self.cmd_gen_strategy(isolation.cmd_args.noise_job).generate_test_command()
        main_job_cmd = self.cmd_gen_strategy(isolation.cmd_args.main_job).generate_test_command()

        with (self.test_run.output_path / "env_vars.sh").open("w") as f:
            for key, value in self.final_env_vars.items():
                f.write(f'export {key}="{value}"\n')

        noise_srun_cmd = (
            " ".join([*srun_parts, "--overlap"])
            + f' bash -c "source {(self.test_run.output_path / "env_vars.sh").absolute()}; '
            + " ".join(noise_job_cmd)
            + '"'
        )
        main_srun_cmd = (
            " ".join([*srun_parts, "--overlap"])
            + f' bash -c "source {(self.test_run.output_path / "env_vars.sh").absolute()}; '
            + " ".join(main_job_cmd)
            + '"'
        )

        return noise_srun_cmd + "\n" + main_srun_cmd
