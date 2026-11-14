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

import gc
import os
import pickle
import weakref
from pathlib import Path

import cloudpickle
import pytest

from cloudai.configurator.trajectory import Trajectory
from cloudai.core import (
    CloudAIGymEnv,
    CmdArgs,
    GymnasiumAdapter,
    RewardOverrides,
    TestDefinition,
    TestRun,
    TestScenario,
)
from cloudai.systems.standalone import StandaloneRunner, StandaloneSystem


@pytest.fixture
def real_env(tmp_path: Path) -> CloudAIGymEnv:
    test = TestDefinition(
        name="serialization",
        description="serialization",
        test_template_name="serialization",
        cmd_args=CmdArgs.model_validate({"tuned": [1, 2]}),
    )
    run = TestRun(name="serialization", test=test, num_nodes=1, nodes=[])
    scenario = TestScenario(name="serialization", test_runs=[run])
    system = StandaloneSystem(name="serialization", install_path=tmp_path / "install", output_path=tmp_path / "output")
    runner = StandaloneRunner(mode="dry-run", system=system, test_scenario=scenario, output_path=tmp_path / "run")
    return CloudAIGymEnv(run, runner, RewardOverrides())


@pytest.mark.parametrize("serializer", [pickle, cloudpickle], ids=["pickle", "cloudpickle"])
def test_serialized_adapter_preserves_live_identity_and_writer(real_env: CloudAIGymEnv, serializer) -> None:
    adapter = GymnasiumAdapter(real_env)
    real_env.trajectory.append(step=1, action={"tuned": 1}, reward=1.0, observation={"metric": 1.0})
    restored = serializer.loads(serializer.dumps(adapter))
    assert restored is adapter
    assert restored.action_space["tuned"].n == 2
    assert restored.observation_space.shape == (1,)
    observation, info = restored.reset(seed=7)
    assert observation.shape == (1,)
    assert info == {}
    assert real_env.test_run.step == 0
    registered_creator = cloudpickle.loads(cloudpickle.dumps(lambda config: adapter))
    assert registered_creator({}) is adapter
    with pytest.raises(RuntimeError, match=r"writer|owned|locked"):
        Trajectory(iteration_dir=real_env.iteration_dir, mode="resume")
    real_env.trajectory.append(step=2, action={"tuned": 2}, reward=2.0, observation={"metric": 2.0})
    loaded = Trajectory.load(real_env.iteration_dir)
    assert [record["step"] for record in loaded.records()] == [1, 2]
    assert loaded.records()[1].get("action", decode_numbers=True) == {"tuned": 2}


@pytest.mark.parametrize("serializer", [pickle, cloudpickle], ids=["pickle", "cloudpickle"])
def test_serialized_adapter_refuses_another_process(
    real_env: CloudAIGymEnv, serializer, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = GymnasiumAdapter(real_env)
    payload = serializer.dumps(adapter)
    before = {path.name: path.read_bytes() for path in real_env.iteration_dir.iterdir() if path.is_file()}
    original_pid = os.getpid()
    with monkeypatch.context() as changed_process:
        changed_process.setattr(os, "getpid", lambda: original_pid + 1)
        with pytest.raises(RuntimeError, match="process"):
            serializer.loads(payload)
    assert {path.name: path.read_bytes() for path in real_env.iteration_dir.iterdir() if path.is_file()} == before
    assert real_env.test_run.step == 0
    assert len(real_env.trajectory) == 0


@pytest.mark.parametrize("serializer", [pickle, cloudpickle], ids=["pickle", "cloudpickle"])
def test_serialized_adapter_refuses_expired_identity(real_env: CloudAIGymEnv, serializer) -> None:
    adapter = GymnasiumAdapter(real_env)
    payload = serializer.dumps(adapter)
    reference = weakref.ref(adapter)
    before = {path.name: path.read_bytes() for path in real_env.iteration_dir.iterdir() if path.is_file()}
    del adapter
    gc.collect()
    assert reference() is None
    with pytest.raises(RuntimeError, match=r"expired|alive"):
        serializer.loads(payload)
    assert {path.name: path.read_bytes() for path in real_env.iteration_dir.iterdir() if path.is_file()} == before
    with pytest.raises(RuntimeError, match=r"writer|owned|locked"):
        Trajectory(iteration_dir=real_env.iteration_dir, mode="resume")
