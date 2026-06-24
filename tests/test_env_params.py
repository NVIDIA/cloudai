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

"""Unit tests for the environment-parameter primitives in cloudai.configurator.env_params.

Annotation model: candidate values live in ``cmd_args`` (the single source of
truth); an ``env_params`` entry is a thin annotation that reclassifies that
field from action-space to env-sampled, optionally carrying sampling
``weights``. These tests use a dedicated, list-capable fixture workload so the
candidate list is a first-class typed field (no serialization warnings).

Worked example - Atari Breakout, the canonical RL arcade game:
``ball_speed`` is the env-sampled knob - the game serves the ball at a speed the
agent does not control, so we sample it per trial and the policy must stay
robust across ball speeds. ``paddle_width`` is an ordinary action-space
dimension - the knob the agent actually tunes.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List, Union

import pytest
from pydantic import BaseModel, ValidationError

from cloudai.configurator.env_params import (
    CsvSink,
    EnvParamsObserver,
    EnvParamSpec,
    EnvParamsSampler,
)
from cloudai.models.workload import CmdArgs, TestDefinition


class BrickGrid(BaseModel):
    """A structured (non-leaf) cmd_args field, used to prove env_params rejects such targets."""

    rows: int = 3


class EnvVarCmdArgs(CmdArgs):
    """cmd_args with top-level, list-capable fields for env_params annotation tests.

    ``ball_speed`` is the env-randomized knob; ``paddle_width`` stands in for an
    ordinary action-space dimension. Both accept either a scalar or a candidate
    list, so a candidate list round-trips through model validation/serialization
    cleanly. ``brick_grid`` is a structured field (not a leaf) for negative tests.
    """

    ball_speed: Union[int, List[int]] = 1
    paddle_width: Union[int, List[int]] = 8
    brick_grid: BrickGrid = BrickGrid()


class EnvVarTestDefinition(TestDefinition):
    """Minimal concrete TestDefinition wrapping ``EnvVarCmdArgs``."""

    cmd_args: EnvVarCmdArgs = EnvVarCmdArgs()


def _tdef(env_params: dict, **cmd_args_overrides) -> EnvVarTestDefinition:
    """Build the fixture TestDefinition through full model validation (so validators fire)."""
    return EnvVarTestDefinition(
        name="breakout",
        description="breakout",
        test_template_name="breakout_template",
        cmd_args=EnvVarCmdArgs(**cmd_args_overrides),
        env_params=env_params,
    )


# --- EnvParamSpec: weights are validated intrinsically (length is cross-field, see below) ---


def test_env_param_spec_accepts_normalized_weights() -> None:
    spec = EnvParamSpec(weights=[0.7, 0.3])
    assert spec.weights == [0.7, 0.3]


def test_env_param_spec_bare_marker_is_valid() -> None:
    """An annotation with no weights is a valid uniform marker."""
    assert EnvParamSpec().weights is None


def test_env_param_spec_rejects_unnormalized_weights() -> None:
    """Strict sum: weights must sum to ~1.0 (relative weights like [7, 3] are rejected)."""
    with pytest.raises(ValidationError, match="must sum to"):
        EnvParamSpec(weights=[7.0, 3.0])


def test_env_param_spec_rejects_negative_weights() -> None:
    with pytest.raises(ValidationError, match="finite and non-negative"):
        EnvParamSpec(weights=[-0.1, 1.1])


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), float("-inf")])
def test_env_param_spec_rejects_non_finite_weights(bad: float) -> None:
    with pytest.raises(ValidationError, match="finite and non-negative"):
        EnvParamSpec(weights=[bad, 1.0])


def test_env_param_spec_rejects_unknown_fields() -> None:
    """Candidate values live in cmd_args, never in the annotation (no ``values`` key)."""
    with pytest.raises(ValidationError):
        EnvParamSpec.model_validate({"values": [1, 2]})


# --- Sampler: draws from candidate lists resolved out of cmd_args ---


def test_sampler_is_deterministic_across_calls() -> None:
    candidates = {"ball_speed": [1, 2, 3]}
    a = EnvParamsSampler(candidates, seed=42)
    b = EnvParamsSampler(candidates, seed=42)
    seq_a = [a.sample(t) for t in range(1, 6)]
    seq_b = [b.sample(t) for t in range(1, 6)]
    assert seq_a == seq_b, "same (seed, trial) must produce the same draw across instances"


def test_sampler_each_param_is_independent() -> None:
    """Adding an unrelated parameter must not perturb existing parameters' draws."""
    base = {"ball_speed": [1, 2, 3]}
    extended = {"ball_speed": [1, 2, 3], "brick_rows": [3, 4, 5]}
    a = [EnvParamsSampler(base, seed=7).sample(t)["ball_speed"] for t in range(1, 11)]
    b = [EnvParamsSampler(extended, seed=7).sample(t)["ball_speed"] for t in range(1, 11)]
    assert a == b, "per-parameter RNG seeding must isolate parameters from each other"


def test_sampler_honors_weights() -> None:
    """A degenerate weight ([1, 0]) must always pick the first candidate."""
    sampler = EnvParamsSampler({"ball_speed": [1, 2]}, seed=1, weights={"ball_speed": [1.0, 0.0]})
    assert all(sampler.sample(t)["ball_speed"] == 1 for t in range(1, 20))


# --- CsvSink: unchanged persistence contract ---


def test_csv_sink_skips_empty_samples_and_rejects_zero_step(tmp_path: Path) -> None:
    sink = CsvSink(tmp_path / "env.csv")
    sink.write(1, {})  # empty -> no-op, no file
    assert not (tmp_path / "env.csv").exists()
    with pytest.raises(ValueError, match="must be a positive trial index"):
        sink.write(0, {"ball_speed": 1})


def test_csv_sink_writes_header_then_rows(tmp_path: Path) -> None:
    sink = CsvSink(tmp_path / "env.csv")
    sink.write(1, {"ball_speed": 2})
    sink.write(2, {"ball_speed": 3})
    contents = (tmp_path / "env.csv").read_text().strip().splitlines()
    assert contents[0] == "step,env"
    assert contents[1].startswith("1,")
    assert contents[2].startswith("2,")


# --- Observer: resolves candidates from cmd_args, stashes the per-trial sample ---


def test_observer_samples_from_cmd_args_and_stashes() -> None:
    """before_step samples a candidate that lives in cmd_args; persistence is the env's job."""
    cmd_args = EnvVarCmdArgs(ball_speed=[1, 2, 3])
    observer = EnvParamsObserver({"ball_speed": EnvParamSpec()}, cmd_args, seed=42)
    test_run = SimpleNamespace(step=3, current_env_params={})

    observer.before_step(test_run)

    assert test_run.current_env_params["ball_speed"] in {1, 2, 3}


def test_observer_skips_scalar_annotation() -> None:
    """A scalar cmd_args value is fixed; annotating it is a no-op (nothing to sample)."""
    cmd_args = EnvVarCmdArgs(ball_speed=2)
    observer = EnvParamsObserver({"ball_speed": EnvParamSpec()}, cmd_args, seed=42)
    test_run = SimpleNamespace(step=1, current_env_params={})

    observer.before_step(test_run)

    assert test_run.current_env_params == {}, "a scalar (fixed) cmd_args knob must not be sampled"


def test_observer_after_step_is_noop() -> None:
    """after_step must not mutate test_run; CloudAIGymEnv.write_trajectory owns persistence."""
    observer = EnvParamsObserver({}, EnvVarCmdArgs(), seed=0)
    test_run = SimpleNamespace(step=1, current_env_params={"x": 1})

    observer.after_step(test_run, observation=[0.0], reward=0.0)

    assert test_run.current_env_params == {"x": 1}


# --- TestDefinition.validate_env_params: annotation validity (cross-field with cmd_args) ---


def test_env_params_uniform_list_is_accepted() -> None:
    """A cmd_args candidate list annotated with a bare marker validates (uniform sampling)."""
    tdef = _tdef({"ball_speed": EnvParamSpec()}, ball_speed=[1, 2])
    assert "ball_speed" in tdef.env_params


def test_env_params_weighted_list_is_accepted() -> None:
    tdef = _tdef({"ball_speed": EnvParamSpec(weights=[0.7, 0.3])}, ball_speed=[1, 2])
    assert tdef.env_params["ball_speed"].weights == [0.7, 0.3]


def test_env_params_scalar_no_op_is_accepted() -> None:
    """Annotating a scalar (fixed) knob is tolerated as a no-op marker."""
    tdef = _tdef({"ball_speed": EnvParamSpec()}, ball_speed=2)
    assert tdef.cmd_args.ball_speed == 2


def test_env_params_unknown_key_rejected() -> None:
    """env_params keys must name real cmd_args fields (the overlay targets cmd_args)."""
    with pytest.raises(ValidationError, match="not cmd_args fields"):
        _tdef({"ghost": EnvParamSpec()}, ball_speed=[1, 2])


def test_env_params_weights_on_scalar_rejected() -> None:
    """Weights require a candidate list; a scalar knob cannot carry them."""
    with pytest.raises(ValidationError, match="not a candidate list"):
        _tdef({"ball_speed": EnvParamSpec(weights=[0.7, 0.3])}, ball_speed=2)


def test_env_params_weights_length_mismatch_rejected() -> None:
    """Weights must align 1:1 with the cmd_args candidate list."""
    with pytest.raises(ValidationError, match="weights length"):
        _tdef({"ball_speed": EnvParamSpec(weights=[0.5, 0.3, 0.2])}, ball_speed=[1, 2])


def test_env_params_structured_target_rejected() -> None:
    """A structured (non-leaf) cmd_args target is rejected: the observer can't sample it, yet
    param_space/is_dse_job exclude the whole key, silently dropping nested action dimensions."""
    with pytest.raises(ValidationError, match="must target a leaf cmd_args field"):
        _tdef({"brick_grid": EnvParamSpec()})


def test_env_params_empty_candidate_list_rejected() -> None:
    """An empty candidate list is rejected at build time: an unweighted spec would otherwise skip
    validation and let the sampler fail later on an empty draw (rng.choice([]))."""
    with pytest.raises(ValidationError, match="empty candidate list"):
        _tdef({"ball_speed": EnvParamSpec()}, ball_speed=[])


# --- is_dse_job: env-sampled lists are not search dimensions ---


def test_is_dse_job_false_for_env_param_only_workload() -> None:
    """A cmd_args list that is purely env-sampled is not a search dimension -> not a DSE job."""
    tdef = _tdef({"ball_speed": EnvParamSpec()}, ball_speed=[1, 2, 3])
    assert tdef.is_dse_job is False


def test_is_dse_job_true_when_a_real_action_dimension_exists() -> None:
    """An un-annotated cmd_args list is a real action dimension -> DSE, even alongside env_params."""
    tdef = _tdef({"ball_speed": EnvParamSpec()}, ball_speed=[1, 2, 3], paddle_width=[4, 8])
    assert tdef.is_dse_job is True
