# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import copy
from pathlib import Path

import pytest

from cloudai.configurator import Trajectory
from cloudai.core import GitRepo, RewardOverrides, TestRun
from cloudai.workloads.megatron_bridge import MegatronBridgeCmdArgs, MegatronBridgeTestDefinition


@pytest.mark.parametrize("credential", ["hf_token", "wandb_key"])
def test_trajectory_projection_respects_workload_secret_redaction(
    tmp_path: Path, slurm_system, credential: str
) -> None:
    secret = "synthetic-recording-secret"
    tdef = MegatronBridgeTestDefinition(
        name="bridge",
        description="bridge",
        test_template_name="MegatronBridge",
        cmd_args=MegatronBridgeCmdArgs.model_validate(
            {
                "hf_token": "synthetic-independent-hf-control",
                credential: secret,
                "model_family_name": "qwen3",
                "model_recipe_name": "30b_a3b",
                "tuple_alias": (secret,),
                "path_alias": Path("/tmp") / secret,
            }
        ),
        extra_env_vars={"AUTH": secret, "ORDINARY": "001"},
        extra_cmd_args={"--credential": secret},
        git_repos=[GitRepo(url="https://github.com/NVIDIA-NeMo/Megatron-Bridge.git", commit="r0.2.0")],
    )
    ordinary = tdef.model_dump()

    serialization_context = {
        "purpose": "trajectory",
        "record": {
            "run_details": {"nodes": [secret], "output_path": f"/tmp/{secret}"},
            "measurement": {"runner_error": f"failed {secret}"},
        },
    }
    projected = tdef.model_dump(mode="json", context=serialization_context)
    assert secret not in repr(serialization_context["record"])

    assert secret not in repr(projected)
    assert projected["extra_env_vars"]["ORDINARY"] == "001"
    assert tdef.model_dump() == ordinary
    assert ordinary["cmd_args"][credential] == secret
    assert ordinary["cmd_args"]["tuple_alias"] == (secret,)
    assert ordinary["cmd_args"]["path_alias"] == Path("/tmp") / secret
    tr = TestRun(
        name=f"bridge-{secret}",
        test=tdef,
        num_nodes=1,
        nodes=[secret],
        exclude_nodes=[f"{secret}-excluded"],
        step=1,
        output_path=tmp_path / secret,
        time_limit=f"{secret}-time",
        extra_srun_args=f"--comment={secret}",
    )
    slurm_system.global_env_vars["WORKLOAD_ALIAS"] = secret
    ordinary_system = slurm_system.model_dump()
    trajectory = Trajectory(iteration_dir=tmp_path)
    trajectory.record(
        test_run=tr,
        action_space={},
        action={},
        reward=1.0,
        observation=[],
        system=slurm_system,
        execution_error=f"failed {secret}",
        rewards=RewardOverrides(),
    )

    for record in (trajectory.records()[0], Trajectory.load(tmp_path).records()[0]):
        assert secret not in repr(dict(record))
        assert secret not in repr(record.metadata)
        assert secret not in repr(record.get("env_params", decode_numbers=True))
    assert secret not in trajectory.output_path.read_text()
    assert secret not in trajectory.metadata_path.read_text()
    assert tdef.model_dump() == ordinary
    assert slurm_system.model_dump() == ordinary_system
    assert tr.nodes == [secret]
    first_id = trajectory.records()[0]["configuration_id"]
    import json

    replacement = "another-synthetic-secret"
    alternate_test = MegatronBridgeTestDefinition.model_validate_json(
        json.dumps(tdef.model_dump(mode="json")).replace(secret, replacement)
    )
    alternate_system = type(slurm_system).model_validate_json(
        json.dumps(slurm_system.model_dump(mode="json")).replace(secret, replacement)
    )
    alternate = TestRun(
        name=f"bridge-{replacement}",
        test=alternate_test,
        num_nodes=1,
        nodes=[replacement],
        exclude_nodes=[f"{replacement}-excluded"],
        step=2,
        output_path=tmp_path / replacement,
        time_limit=f"{replacement}-time",
        extra_srun_args=f"--comment={replacement}",
    )
    alternate_record = trajectory.record(
        test_run=alternate,
        system=alternate_system,
        action_space={},
        action={},
        reward=1.0,
        observation=[],
        execution_error=f"failed {replacement}",
        rewards=RewardOverrides(),
    )
    assert alternate_record["configuration_id"] == first_id
    assert replacement not in trajectory.metadata_path.read_text()

    collision = tdef.model_copy(deep=True)
    collision.extra_env_vars = {secret: "left", projected["cmd_args"][credential]: "right"}
    collision_ordinary = collision.model_dump()
    collision_run = copy.deepcopy(tr)
    collision_run.test = collision
    collision_writer = Trajectory(iteration_dir=tmp_path / "collision")
    before_collision = {path.name: path.read_bytes() for path in (tmp_path / "collision").iterdir()}
    with pytest.raises(ValueError, match=r"collision|duplicate"):
        collision_writer.record(
            test_run=collision_run,
            system=slurm_system,
            rewards=RewardOverrides(),
            action_space={},
            action={},
            reward=1.0,
            observation=[],
        )
    assert {path.name: path.read_bytes() for path in (tmp_path / "collision").iterdir()} == before_collision
    with pytest.raises(ValueError, match=r"collision|duplicate"):
        collision.model_dump(mode="json", context={"purpose": "trajectory"})
    assert collision.model_dump() == collision_ordinary


@pytest.mark.parametrize(
    ("hf_token", "wandb_key"),
    [
        ("synthetic-hf-long", "synthetic-wandb"),
        ("prefix-shared-suffix", "shared"),
        ("REDACTED", "synthetic-wandb"),
        ("synthetic-hf-control", "<REDACTED>-synthetic-wandb"),
    ],
)
def test_trajectory_projection_handles_overlapping_credentials_and_is_idempotent(
    tmp_path: Path, slurm_system, hf_token: str, wandb_key: str
) -> None:
    marker = "<REDACTED>"
    definition = MegatronBridgeTestDefinition(
        name="overlap",
        description="overlap",
        test_template_name="MegatronBridge",
        cmd_args=MegatronBridgeCmdArgs.model_validate(
            {
                "hf_token": hf_token,
                "wandb_key": wandb_key,
                "model_family_name": "qwen3",
                "model_recipe_name": "30b_a3b",
                "aliases": [hf_token, wandb_key, marker],
            }
        ),
        extra_env_vars={"HF": hf_token, "WANDB": wandb_key, "MARKER": marker},
        git_repos=[GitRepo(url="https://github.com/NVIDIA-NeMo/Megatron-Bridge.git", commit="r0.2.0")],
    )
    ordinary = definition.model_dump()
    projected = definition.model_dump(mode="json", context={"purpose": "trajectory"})
    assert projected["cmd_args"]["hf_token"] == projected["cmd_args"]["wandb_key"] == marker
    assert projected["cmd_args"]["aliases"] == [marker, marker, marker]
    assert projected["extra_env_vars"] == {"HF": marker, "WANDB": marker, "MARKER": marker}
    projected_model = MegatronBridgeTestDefinition.model_validate(projected)
    assert projected_model.model_dump(mode="json", context={"purpose": "trajectory"}) == projected
    writer = Trajectory(iteration_dir=tmp_path)
    run = TestRun(name="overlap", test=definition, num_nodes=1, nodes=[], step=1)
    writer.record(
        test_run=run,
        action_space={},
        action={},
        reward=1.0,
        observation=[],
        system=slurm_system,
        rewards=RewardOverrides(),
    )
    for record in (writer.records()[0], Trajectory.load(tmp_path).records()[0]):
        environment = record.get("env_params", decode_numbers=True)
        assert environment["hf_token"] == environment["wandb_key"] == marker
        assert environment["aliases"] == [marker, marker, marker]
    assert definition.model_dump() == ordinary
