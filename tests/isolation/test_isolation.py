from pathlib import Path
from typing import cast

import toml

from cloudai.core import Test, TestTemplate
from cloudai.models.scenario import TestScenarioModel
from cloudai.systems.slurm import SlurmSystem
from cloudai.test_parser import TestParser
from cloudai.test_scenario_parser import TestScenarioParser
from cloudai.workloads.isolation import IsolationCmdArgs, IsolationTestDefinition
from cloudai.workloads.nccl_test import NCCLCmdArgs, NCCLTestDefinition
from cloudai.workloads.nemo_run import NeMoRunCmdArgs, NeMoRunTestDefinition


def test_create_via_objects() -> None:
    IsolationTestDefinition(
        name="isolation",
        description="Isolation test",
        test_template_name="Isolation",
        cmd_args=IsolationCmdArgs(
            nodes_split_rule="even",
            main_job=NeMoRunTestDefinition(
                name="main_job",
                description="Main job",
                test_template_name="NeMoRun",
                cmd_args=NeMoRunCmdArgs(docker_image_url="", task="", recipe_name=""),
            ),
            noise_job=NCCLTestDefinition(
                name="noise_job",
                description="Noise job",
                test_template_name="NcclTest",
                cmd_args=NCCLCmdArgs(docker_image_url=""),
            ),
        ),
    )


def test_create_tdef_via_toml(slurm_system: SlurmSystem) -> None:
    tp = TestParser([], slurm_system)
    test = tp._parse_data(
        toml.loads("""
name = "n"
description = "d"
test_template_name = "Isolation"

[cmd_args]
nodes_split_rule = "even"
[cmd_args.main_job]
name = "main_job"
description = "Main job"
test_template_name = "NeMoRun"
[cmd_args.main_job.cmd_args]
docker_image_url = "nvcr.io#nvidia/pytorch:25.06-py3"
task = "train"
recipe_name = "nemo_recipe"

[cmd_args.noise_job]
name = "noise_job"
description = "Noise job"
test_template_name = "NcclTest"
[cmd_args.noise_job.cmd_args]
docker_image_url = "nvcr.io#nvidia/pytorch:25.06-py3"
stepfactor = 2
""")
    )

    assert isinstance(test.test_definition, IsolationTestDefinition)


def test_create_tdef_via_toml_in_scenario() -> None:
    TestScenarioModel.model_validate(
        toml.loads("""
name = "scenario"

[[Tests]]
id = "1"
num_nodes = 2

name = "n"
description = "d"
test_template_name = "Isolation"

[Tests.cmd_args]
nodes_split_rule = "even"
[Tests.cmd_args.main_job]
name = "main_job"
description = "Main job"
test_template_name = "NeMoRun"
[Tests.cmd_args.main_job.cmd_args]
docker_image_url = "nvcr.io#nvidia/pytorch:25.06-py3"
task = "train"
recipe_name = "nemo_recipe"

[Tests.cmd_args.noise_job]
name = "noise_job"
description = "Noise job"
test_template_name = "NcclTest"
[Tests.cmd_args.noise_job.cmd_args]
docker_image_url = "nvcr.io#nvidia/pytorch:25.06-py3"
stepfactor = 2
    """)
    )


def test_test_in_scenario(slurm_system: SlurmSystem) -> None:
    tsp = TestScenarioParser(Path(""), slurm_system, {}, {})
    tsp.test_mapping["isolation"] = Test(
        test_definition=IsolationTestDefinition(
            name="isolation",
            description="Isolation test",
            test_template_name="Isolation",
            cmd_args=IsolationCmdArgs(
                nodes_split_rule="even",
                main_job=NeMoRunTestDefinition(
                    name="main_job",
                    description="Main job",
                    test_template_name="NeMoRun",
                    cmd_args=NeMoRunCmdArgs(docker_image_url="", task="", recipe_name=""),
                ),
                noise_job=NCCLTestDefinition(
                    name="noise_job",
                    description="Noise job",
                    test_template_name="NcclTest",
                    cmd_args=NCCLCmdArgs(docker_image_url=""),
                ),
            ),
        ),
        test_template=TestTemplate(system=slurm_system),
    )
    model = TestScenarioModel.model_validate(
        toml.loads(
            """
name = "scenario"

[[Tests]]
id = "1"
num_nodes = 2
test_name = "isolation"

[Tests.cmd_args.main_job.cmd_args]
opt = "v1"

[Tests.cmd_args.noise_job.cmd_args]
stepfactor = 3
            """
        )
    )
    _, tdef = tsp._prepare_tdef(model.tests[0])
    assert isinstance(tdef, IsolationTestDefinition)
    isolation = cast(IsolationTestDefinition, tdef)
    assert isolation.cmd_args.main_job.cmd_args_dict["opt"] == "v1"
    assert isolation.cmd_args.noise_job.cmd_args.stepfactor == 3
