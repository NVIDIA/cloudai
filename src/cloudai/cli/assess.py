import argparse
import logging
import re
import subprocess
from pathlib import Path

import click

from cloudai.cli.handlers import handle_non_dse_job, prepare_installation, register_signal_handlers
from cloudai.core import Runner, Test, TestRun, TestScenario, TestTemplate
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.workloads.nccl_test.nccl import NCCLCmdArgs, NCCLTestDefinition


@click.group()
def assess():
    """Assessment commands."""
    pass


def build_slurm_system() -> SlurmSystem:
    """Build a Slurm system for assessment."""
    result = subprocess.run(["scontrol", "show", "partition"], text=True, capture_output=True)
    if result.returncode != 0:
        logging.error(f"Failed to run 'scontrol show partition': {result.stderr}")
        exit(1)

    partitions = re.split(r"\n\s*\n", result.stdout.strip())
    default_partition: str | None = None

    for part in partitions:
        match = re.search(r"PartitionName=(\S+)", part)
        if match:
            name = match.group(1)
            if "Default=YES" in part:
                default_partition = name

    if not default_partition:
        logging.error("No default partition found in Slurm configuration.")
        exit(1)

    return SlurmSystem(
        name="slurm",
        install_path=Path.cwd() / "_install",
        output_path=Path.cwd() / "_output",
        default_partition=default_partition,
        partitions=[],
    )


def get_test_runs(slurm_system: SlurmSystem) -> list[TestRun]:
    nccl = NCCLTestDefinition(
        name="nccl",
        description="NCCL test",
        test_template_name="NcclTest",
        cmd_args=NCCLCmdArgs(
            docker_image_url="nvcr.io#nvidia/pytorch:25.06-py3",
            ngpus=1,
            minbytes="128",
            maxbytes="4G",
            iters=100,
            warmup_iters=50,
            stepfactor=2,
        ),
    )
    test = Test(test_definition=nccl, test_template=TestTemplate(system=slurm_system))
    tr = TestRun(name="NCCL benchmark", test=test, num_nodes=2, nodes=[])
    return [tr]


@assess.command()
def run():
    """Run assessment."""
    system = build_slurm_system()
    system.update()

    partition = None
    for p in system.partitions:
        if p.name == system.default_partition:
            partition = p
            break
    if not partition:
        logging.error(f"Default partition '{system.default_partition}' not found in system partitions.")
        exit(1)

    if len(partition.slurm_nodes) < 2:
        logging.error(f"Partition '{partition.name}' has less than 2 nodes, cannot run assessment.")
        exit(1)

    scenario = TestScenario(name="Assessment", test_runs=get_test_runs(system))

    installables, installer = prepare_installation(system, [tr.test for tr in scenario.test_runs], scenario)
    result = installer.install(installables)
    if not result.success:
        logging.error("Installation failed, cannot proceed with assessment.")
        exit(1)

    runner = Runner("run", system, scenario)
    register_signal_handlers(runner.cancel_on_signal)
    handle_non_dse_job(runner, argparse.Namespace(mode="run"))
