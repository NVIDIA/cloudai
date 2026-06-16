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

import dataclasses
import random
from pathlib import Path
from typing import List, Union

import pytest
from pydantic import BaseModel, ValidationError

from cloudai.configurator.env_params import (
    EnvParam,
    EnvParams,
    EnvParamSpec,
    EnvParamsSink,
    ObsLeafDescriptor,
)
from cloudai.core import TestRun
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


# --- EnvParam: one resolved knob - candidate values, optional weights, a single draw ---


def test_env_param_defaults_to_unweighted() -> None:
    """A knob built without weights samples uniformly (weights is None)."""
    assert EnvParam(candidates=[1, 2, 3]).weights is None


def test_env_param_draw_returns_a_candidate() -> None:
    """draw() always yields one of the knob's own candidate values."""
    knob = EnvParam(candidates=[1, 2, 3])
    assert all(knob.draw(random.Random(s)) in {1, 2, 3} for s in range(50))


def test_env_param_draw_is_reproducible_for_a_given_rng() -> None:
    """draw() consumes the caller's RNG (no internal seeding), so equal RNG state yields equal draws."""
    knob = EnvParam(candidates=[10, 20, 30, 40])
    assert knob.draw(random.Random(123)) == knob.draw(random.Random(123))


def test_env_param_draw_honors_degenerate_weights() -> None:
    """A degenerate weight ([1, 0]) collapses the draw onto the first candidate, whatever the RNG."""
    knob = EnvParam(candidates=[1, 2], weights=[1.0, 0.0])
    assert all(knob.draw(random.Random(s)) == 1 for s in range(50))


def test_env_param_is_immutable() -> None:
    """Frozen value object: a resolved knob cannot be mutated after construction."""
    knob = EnvParam(candidates=[1, 2, 3])
    with pytest.raises(dataclasses.FrozenInstanceError):
        knob.candidates = [9]  # pyright: ignore[reportAttributeAccessIssue]


# --- EnvParams.sample: draws from candidate lists resolved out of cmd_args ---


def test_sampler_is_deterministic_across_calls() -> None:
    env_params = EnvParams(params={"ball_speed": EnvParam(candidates=[1, 2, 3])}, seed=42)
    seq_a = [env_params.sample(t) for t in range(1, 6)]
    seq_b = [env_params.sample(t) for t in range(1, 6)]
    assert seq_a == seq_b, "same (seed, trial) must produce the same draw across calls"


def test_sampler_each_param_is_independent() -> None:
    """Adding an unrelated parameter must not perturb existing parameters' draws."""
    base = EnvParams(params={"ball_speed": EnvParam(candidates=[1, 2, 3])}, seed=7)
    extended = EnvParams(
        params={"ball_speed": EnvParam(candidates=[1, 2, 3]), "brick_rows": EnvParam(candidates=[3, 4, 5])},
        seed=7,
    )
    a = [base.sample(t)["ball_speed"] for t in range(1, 11)]
    b = [extended.sample(t)["ball_speed"] for t in range(1, 11)]
    assert a == b, "per-parameter RNG seeding must isolate parameters from each other"


def test_sampler_honors_weights() -> None:
    """A degenerate weight ([1, 0]) must always pick the first candidate."""
    env_params = EnvParams(params={"ball_speed": EnvParam(candidates=[1, 2], weights=[1.0, 0.0])}, seed=1)
    assert all(env_params.sample(t)["ball_speed"] == 1 for t in range(1, 20))


def test_sample_covers_all_declared_params() -> None:
    """sample() returns exactly one value per declared knob, each drawn from that knob's candidates."""
    env_params = EnvParams(
        params={"ball_speed": EnvParam(candidates=[1, 2]), "paddle_width": EnvParam(candidates=[4, 8])},
        seed=3,
    )
    drawn = env_params.sample(5)
    assert set(drawn) == {"ball_speed", "paddle_width"}
    assert drawn["ball_speed"] in {1, 2}
    assert drawn["paddle_width"] in {4, 8}


def test_env_params_is_immutable() -> None:
    """Frozen value object: resolved sampling state cannot be mutated after construction."""
    env_params = EnvParams(params={"ball_speed": EnvParam(candidates=[1, 2])}, seed=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        env_params.seed = 1  # pyright: ignore[reportAttributeAccessIssue]


# --- EnvParamsSink: unchanged persistence contract ---


def test_csv_sink_skips_empty_samples_and_rejects_zero_step(tmp_path: Path) -> None:
    sink = EnvParamsSink()
    path = tmp_path / "env.csv"
    sink.write(path, 1, {})  # empty -> no-op, no file
    assert not path.exists()
    with pytest.raises(ValueError, match="must be a positive trial index"):
        sink.write(path, 0, {"ball_speed": 1})


def test_csv_sink_writes_header_then_rows(tmp_path: Path) -> None:
    sink = EnvParamsSink()
    path = tmp_path / "env.csv"
    sink.write(path, 1, {"ball_speed": 2})
    sink.write(path, 2, {"ball_speed": 3})
    contents = path.read_text().strip().splitlines()
    assert contents[0] == "step,env"
    assert contents[1].startswith("1,")
    assert contents[2].startswith("2,")


# --- EnvParams.from_test: resolves candidate lists from cmd_args, once at env formulation ---


def test_env_params_from_test_resolves_list_candidate() -> None:
    """A list-valued cmd_args field resolves to its candidate list; sampling then draws from it."""
    env_params = EnvParams.from_test(_tdef({"ball_speed": EnvParamSpec()}, ball_speed=[1, 2, 3]))

    assert env_params is not None
    assert env_params.params == {"ball_speed": EnvParam(candidates=[1, 2, 3], weights=None)}
    assert env_params.sample(3)["ball_speed"] in {1, 2, 3}


def test_env_params_from_test_carries_weights() -> None:
    """Weighted annotations propagate their weights alongside the resolved candidates."""
    env_params = EnvParams.from_test(_tdef({"ball_speed": EnvParamSpec(weights=[0.7, 0.3])}, ball_speed=[1, 2]))

    assert env_params is not None
    assert env_params.params["ball_speed"].weights == [0.7, 0.3]


def test_env_params_from_test_reads_seed_from_agent_config() -> None:
    """The sampler seed comes from agent_config.random_seed so a run is reproducible end to end."""
    tdef = _tdef({"ball_speed": EnvParamSpec()}, ball_speed=[1, 2, 3])
    tdef.agent_config = {"random_seed": 42}

    env_params = EnvParams.from_test(tdef)

    assert env_params is not None and env_params.seed == 42


def test_env_params_from_test_defaults_seed_to_zero_without_agent_config() -> None:
    """No agent_config (the default) -> seed 0, so a declared-but-unseeded run still samples."""
    env_params = EnvParams.from_test(_tdef({"ball_speed": EnvParamSpec()}, ball_speed=[1, 2, 3]))

    assert env_params is not None and env_params.seed == 0


def test_env_params_from_test_defaults_seed_to_zero_when_key_absent() -> None:
    """agent_config present but without random_seed still falls back to seed 0."""
    tdef = _tdef({"ball_speed": EnvParamSpec()}, ball_speed=[1, 2, 3])
    tdef.agent_config = {"other": 1}

    env_params = EnvParams.from_test(tdef)

    assert env_params is not None and env_params.seed == 0


def test_env_params_from_test_resolves_multiple_params() -> None:
    """Every list-valued annotation becomes its own knob, candidates resolved from cmd_args."""
    env_params = EnvParams.from_test(
        _tdef(
            {"ball_speed": EnvParamSpec(), "paddle_width": EnvParamSpec()},
            ball_speed=[1, 2],
            paddle_width=[4, 8],
        )
    )

    assert env_params is not None
    assert set(env_params.params) == {"ball_speed", "paddle_width"}
    assert env_params.params["ball_speed"].candidates == [1, 2]
    assert env_params.params["paddle_width"].candidates == [4, 8]


def test_env_params_from_test_none_when_no_env_params() -> None:
    """No annotations declared -> no EnvParams object (the zero-overhead path)."""
    assert EnvParams.from_test(_tdef({})) is None


# --- TestDefinition.validate_env_params: annotation validity (cross-field with cmd_args) ---


def test_env_params_uniform_list_is_accepted() -> None:
    """A cmd_args candidate list annotated with a bare marker validates (uniform sampling)."""
    tdef = _tdef({"ball_speed": EnvParamSpec()}, ball_speed=[1, 2])
    assert "ball_speed" in tdef.env_params


def test_env_params_weighted_list_is_accepted() -> None:
    tdef = _tdef({"ball_speed": EnvParamSpec(weights=[0.7, 0.3])}, ball_speed=[1, 2])
    assert tdef.env_params["ball_speed"].weights == [0.7, 0.3]


def test_env_params_scalar_annotation_rejected() -> None:
    """A scalar (fixed) knob carries nothing to sample; the annotation is a meaningless label and is
    rejected at parse time (it only reclassifies a list-valued sweep as env-sampled)."""
    with pytest.raises(ValidationError, match="not a candidate list"):
        _tdef({"ball_speed": EnvParamSpec()}, ball_speed=2)


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


def test_apply_params_set_accepts_weighted_env_param_draw() -> None:
    """Regression: apply_params_set re-validates after the overlay; a weighted env_param's scalar
    draw must not trip validate_env_params (which would reject 'weights but not a candidate list')."""
    tdef = _tdef({"ball_speed": EnvParamSpec(weights=[0.7, 0.3])}, ball_speed=[1, 2])
    tr = TestRun(name="tr", test=tdef, num_nodes=1, nodes=[])

    new_tr = tr.apply_params_set({}, env_params={"ball_speed": 1})

    assert new_tr.test.cmd_args.ball_speed == 1
    assert new_tr.current_env_params == {"ball_speed": 1}


# --- ObsLeafDescriptor: structured-observation leaf schema ---


def test_obs_leaf_descriptor_box_defaults() -> None:
    leaf = ObsLeafDescriptor(kind="box", dim=2)
    assert leaf.kind == "box"
    assert leaf.dim == 2
    assert leaf.n is None


def test_obs_leaf_descriptor_discrete_requires_n() -> None:
    leaf = ObsLeafDescriptor(kind="discrete", dim=1, n=3)
    assert leaf.n == 3
    with pytest.raises(ValidationError, match="requires n"):
        ObsLeafDescriptor(kind="discrete", dim=1)
    with pytest.raises(ValidationError, match="requires n"):
        ObsLeafDescriptor(kind="discrete", dim=1, n=0)


def test_obs_leaf_descriptor_rejects_bad_dim_and_extra_fields() -> None:
    with pytest.raises(ValidationError, match="dim must be"):
        ObsLeafDescriptor(kind="box", dim=0)
    with pytest.raises(ValidationError):
        ObsLeafDescriptor(kind="box", dim=1, unexpected=1)
    with pytest.raises(ValidationError):
        ObsLeafDescriptor(kind="categorical", dim=1)
