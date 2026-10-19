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

import pytest
from pydantic import ValidationError

import cloudai.configurator.env_params as runtime
from cloudai.models.env_params import CategoricalEncoding, EnvParamSpec, ObsLeafDescriptor


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
        ObsLeafDescriptor(kind="box", dim=1, unexpected=1)  # type: ignore
    with pytest.raises(ValidationError):
        ObsLeafDescriptor(kind="categorical", dim=1)  # type: ignore


def test_env_param_spec_defaults_to_categorical_encoding() -> None:
    """An unspecified encoding defaults to categorical (back-compat with bare ``EnvParamSpec()``)."""
    assert EnvParamSpec().encoding == CategoricalEncoding()


def test_env_param_spec_parses_encoding_from_config() -> None:
    """The encoding is config-selectable (TOML-style dict) via its ``type`` field."""
    spec = EnvParamSpec.model_validate({"encoding": {"type": "categorical"}})
    assert isinstance(spec.encoding, CategoricalEncoding)


def test_categorical_encoding_descriptor_and_encode() -> None:
    """Categorical maps a candidate list to a Discrete leaf and a value to its index."""
    enc = CategoricalEncoding()
    assert enc.observation_descriptor([10, 20, 30]) == ObsLeafDescriptor(kind="discrete", n=3)
    assert enc.encode(30, [10, 20, 30]) == 2


def test_existing_import_paths_reexport_configuration_models() -> None:
    assert runtime.EnvParamSpec is EnvParamSpec
    assert runtime.CategoricalEncoding is CategoricalEncoding
    assert runtime.ObsLeafDescriptor is ObsLeafDescriptor
