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


from pathlib import Path

import pytest

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


@pytest.mark.parametrize(
    "candidates",
    [[True, 1], [1, True], [1, 1.0], [1, 2], [[True], [1]], [(1,), [1]], [{"nested": [True]}, {"nested": [1]}]],
)
def test_native_action_roundtrip_preserves_exact_candidate_type(real_env: CloudAIGymEnv, candidates: list) -> None:
    real_env.test_run.test.cmd_args = CmdArgs.model_validate({"tuned": candidates})
    adapter = GymnasiumAdapter(real_env)

    for index, value in enumerate(candidates):
        encoded = adapter.encode_action({"tuned": value})
        assert encoded == {"tuned": index}
        decoded = adapter.decode_action(encoded)
        assert decoded == {"tuned": value}
        assert type(decoded["tuned"]) is type(value)


def test_native_nan_candidate_roundtrips_by_identity_without_merging_distinct_values(real_env: CloudAIGymEnv) -> None:
    nan = float("nan")
    real_env.test_run.test.cmd_args = CmdArgs.model_validate({"tuned": [nan, 1.0]})
    adapter = GymnasiumAdapter(real_env)
    decoded = adapter.decode_action({"tuned": 0})
    assert decoded["tuned"] is nan
    assert adapter.encode_action(decoded) == {"tuned": 0}
    assert adapter.encode_action(adapter.decode_action({"tuned": 1})) == {"tuned": 1}
    with pytest.raises(ValueError):
        adapter.encode_action({"tuned": float("nan")})


def test_exact_duplicate_candidates_keep_first_index_and_absent_values_raise(real_env: CloudAIGymEnv) -> None:
    real_env.test_run.test.cmd_args = CmdArgs.model_validate({"tuned": [{"nested": [1]}, {"nested": [1]}]})
    adapter = GymnasiumAdapter(real_env)
    assert adapter.encode_action({"tuned": {"nested": [1]}}) == {"tuned": 0}
    with pytest.raises(ValueError):
        adapter.encode_action({"tuned": {"nested": [True]}})
    with pytest.raises(ValueError):
        adapter.encode_action({"tuned": {"nested": [2]}})
