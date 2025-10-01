import argparse
import getpass
import logging
import re
import subprocess
from pathlib import Path

import click
from rich.console import Console
from rich.style import Style

from cloudai.cli.handlers import handle_non_dse_job, prepare_installation, register_signal_handlers
from cloudai.core import Runner, Test, TestRun, TestScenario, TestTemplate
from cloudai.systems.slurm.slurm_system import SlurmPartition, SlurmSystem
from cloudai.workloads.nccl_test.nccl import NCCLCmdArgs, NCCLTestDefinition

error_style = Style(color="red", bold=True)
console = Console()


@click.group()
def assess():
    """Assessment commands."""
    pass


def build_slurm_system() -> SlurmSystem:
    """Build a Slurm system for assessment."""
    result = subprocess.run(["scontrol", "show", "partition"], text=True, capture_output=True)
    if result.returncode != 0:
        console.log(f"Failed to run 'scontrol show partition': {result.stderr}", style=error_style)
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
        console.log("No default partition found in Slurm configuration.", style=error_style)
        exit(1)

    cmd = ["sacctmgr", "-nP", "show", "assoc", "where", f"user={getpass.getuser()}", "format=account"]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        console.log(f"Failed to run '{' '.join(cmd)}': {result.stderr}", style=error_style)
        exit(1)
    account = result.stdout.splitlines()[0].strip()
    if account:
        console.log(f"Using Slurm account: [bold cyan]{account}[/]")
    else:
        console.log(f"No Slurm account found for the current user: {result.stdout}", style=error_style)

    cmd = ["sinfo", "-p", default_partition, "-o", "%G", "--noheader"]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        console.log(f"Failed to run '{' '.join(cmd)}': {result.stderr}", style=error_style)
        exit(1)

    gpus_per_node: int | None = None
    m = re.search(r"gpu:(\d+)", result.stdout)
    if not m:
        console.log(f"[warning]No GPUs info in GRES for [bold green]{default_partition}[/] partition[/warning]")
    else:
        gpus_per_node = int(m.group(1))

    system = SlurmSystem(
        name="slurm",
        install_path=Path.cwd() / "_install",
        output_path=Path.cwd() / "_output",
        default_partition=default_partition,
        account=account,
        gpus_per_node=gpus_per_node,
        partitions=[SlurmPartition(name=default_partition, slurm_nodes=[])],
    )
    console.log(f"Using Slurm system with default partition [bold green]{system.default_partition}[/]")
    system.update()

    partition = None
    for p in system.partitions:
        if p.name == system.default_partition:
            partition = p
            break
    if not partition:
        console.log(
            f"Default partition '{system.default_partition}' not found in system partitions.", style=error_style
        )
        exit(1)

    console.log(
        f"Default partition [bold green]{partition.name}[/] has [bold yellow]{len(partition.slurm_nodes)}[/] nodes"
    )
    return system


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
    tr = TestRun(name="NCCL-benchmark", test=test, num_nodes=2, nodes=[])
    return [tr]


@assess.command()
def run():
    """Run assessment."""
    with console.status("[bold green]Building Slurm system..."):
        system = build_slurm_system()

    trs = get_test_runs(system)
    if not trs:
        logging.error("No test runs available for assessment.")
        exit(1)
    scenario = TestScenario(name="Assessment", test_runs=trs)
    console.log(f"Prepared scenario '{scenario.name}' with {len(scenario.test_runs)} test runs.")

    with console.status("Preparing installation of required components..."):
        installables, installer = prepare_installation(system, [tr.test for tr in scenario.test_runs], scenario)
        result = installer.install(installables)
        if not result.success:
            logging.error("Installation failed, cannot proceed with assessment", style=error_style)
            exit(1)

    console.log("Starting assessment run...")
    runner = Runner("run", system, scenario)
    register_signal_handlers(runner.cancel_on_signal)
    # handle_non_dse_job(runner, argparse.Namespace(mode="run"))
