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

"""
Training report parsers: a workload's run artifacts -> TrainingResults.

STEP mappings are a plain field -> tag dict (value column); per-field unit scale is applied via SCALE.
"""

import ast
import json
import logging
from abc import ABC, abstractmethod
from functools import reduce
from pathlib import Path
from typing import Any, ClassVar, Optional

import yaml

from cloudai.core import System, TestRun

from .mappings import (
    MEGATRON_BRIDGE_CONFIG,
    MEGATRON_BRIDGE_SCALE,
    MEGATRON_BRIDGE_STEPS,
    MEGATRON_CONFIG,
    MEGATRON_STEPS,
    NEMO_CONFIG,
    NEMO_STEPS,
)
from .models import OPTIONAL_STEP_FIELDS, Scalar, TrainingConfig, TrainingResults, TrainingStep
from .tb_reader import read_scalars, read_text


class TrainingParser(ABC):
    """Base parser: a workload's run artifacts -> TrainingResults."""

    STEP_MAPPING: ClassVar[dict[str, str]]  # TrainingStep field -> TB scalar tag (read from the value column)
    CONFIG_MAPPING: ClassVar[dict[str, str]]
    SCALE: ClassVar[dict[str, float]] = {}  # TrainingStep field -> unit factor (e.g. m-bridge gigabytes -> bytes)
    DEFAULT_GPUS_PER_NODE: ClassVar[int] = 8

    @abstractmethod
    def get_tb_dir(self, tr: TestRun) -> Path:
        """Directory holding the run's TensorBoard event files."""

    @abstractmethod
    def get_config_path(self, tr: TestRun) -> Optional[Path]:
        """Config artifact path used by the parser, or None when it cannot be found."""

    @abstractmethod
    def get_config(self, tr: TestRun) -> dict:
        """Read the workload's config artifact into a (possibly nested) dict."""

    @abstractmethod
    def get_model_name(self, tr: TestRun) -> str:
        """CloudAI-sourced model identity for the run."""

    def can_parse(self, tr: TestRun) -> bool:
        """Return True when the run produced the TB events and config artifact this parser needs."""
        tb_dir = self.get_tb_dir(tr)
        if not (tb_dir.is_dir() and any(tb_dir.iterdir())):
            return False
        config_path = self.get_config_path(tr)
        return config_path is not None and config_path.is_file()

    def parse(self, tr: TestRun, system: System) -> TrainingResults:
        """Read TB scalars + the config artifact and assemble TrainingResults."""
        steps: list[TrainingStep] = self._build_steps(self._read_scalars(tr))
        config: TrainingConfig = self._build_config(self.get_config(tr), tr, system)
        return TrainingResults(config=config, steps=steps)

    def _build_steps(self, scalars: list[Scalar]) -> list[TrainingStep]:
        """list[Scalar] -> list[TrainingStep] via STEP_MAPPING; drop a step only if a required tag is missing."""
        steps = (self._build_step(step, group) for step, group in self._group_by_step(scalars).items())
        return [s for s in steps if s is not None]

    def _build_step(self, step: int, step_scalars: list[Scalar]) -> Optional[TrainingStep]:
        """Build one TrainingStep, or None when a required (non-optional) TB tag is missing."""
        scalars_dict = {scalar.tag: scalar.value for scalar in step_scalars}
        missing = [
            tag
            for field, tag in self.STEP_MAPPING.items()
            if field not in OPTIONAL_STEP_FIELDS and tag not in scalars_dict
        ]
        if missing:
            logging.warning(f"{type(self).__name__}: skipping step {step}, missing required TB tags {missing}")
            return None
        field_values: dict[str, Any] = {
            field: scalars_dict[tag] * self.SCALE.get(field, 1)
            for field, tag in self.STEP_MAPPING.items()
            if tag in scalars_dict
        }
        return TrainingStep(iteration=step, **field_values)

    def _build_config(self, raw: dict, tr: TestRun, system: System) -> TrainingConfig:
        """Map the raw config via CONFIG_MAPPING, then fill the CloudAI-computed fields."""
        field_values: dict[str, Any] = {
            field: self._get_from_dict(raw, path) for field, path in self.CONFIG_MAPPING.items()
        }
        config = TrainingConfig(**field_values)
        gpus_per_node = (
            getattr(system, "gpus_per_node", None)
            or getattr(system, "ntasks_per_node", None)
            or self.DEFAULT_GPUS_PER_NODE
        )
        config.test_template_name = tr.test.test_template_name
        config.num_nodes = tr.nnodes
        config.world_size = config.num_nodes * gpus_per_node
        config.model_name = self.get_model_name(tr)
        config.data_parallel_size = self._compute_data_parallel_size(config)
        return config

    def _read_scalars(self, tr: TestRun) -> list[Scalar]:
        """Read the run's scalar events (subclasses may drop workload-specific noise)."""
        return read_scalars(self.get_tb_dir(tr))

    @staticmethod
    def _group_by_step(scalars: list[Scalar]) -> dict[int, list[Scalar]]:
        """Group flat scalar events by step (one list of events per step), ordered by step."""
        by_step: dict[int, list[Scalar]] = {}
        for scalar in scalars:
            if scalar.step not in by_step:
                by_step[scalar.step] = []
            by_step[scalar.step].append(scalar)
        return dict(sorted(by_step.items()))

    @staticmethod
    def _get_from_dict(raw: dict, path: str) -> Any:
        """Read a dotted path (e.g. 'model.num_layers') from a nested dict; None if any key is missing."""
        return reduce(lambda value, key: value.get(key) if isinstance(value, dict) else None, path.split("."), raw)

    def _compute_data_parallel_size(self, config: TrainingConfig) -> int:
        """world_size / (tensor*pipeline*context parallel). EP is carved out of DP, not world_size."""
        cls = type(self).__name__
        tp = config.tensor_parallel_size
        pp = config.pipeline_parallel_size or 1
        cp = config.context_parallel_size or 1
        if not tp:
            raise ValueError(f"{cls}: tensor_parallel_size missing/invalid in parsed config (got {tp!r})")
        parallel = tp * pp * cp
        if config.world_size % parallel != 0:
            raise ValueError(
                f"{cls}: world_size {config.world_size} not a multiple of tp*pp*cp={parallel}; check topology"
            )
        return config.world_size // parallel


class NeMoRunParser(TrainingParser):
    """NeMoRun: config from nemo_config.json, steps from TB scalars."""

    STEP_MAPPING = NEMO_STEPS
    CONFIG_MAPPING = NEMO_CONFIG
    CONFIG_FILE = "nemo_config.json"

    def get_tb_dir(self, tr: TestRun) -> Path:
        return tr.output_path / "tb_logs" / "default"

    def get_config_path(self, tr: TestRun) -> Path:
        return tr.output_path / self.CONFIG_FILE

    def get_config(self, tr: TestRun) -> dict:
        return json.loads(self.get_config_path(tr).read_text())

    def get_model_name(self, tr: TestRun) -> str:
        return tr.test.cmd_args.recipe_name


# Megatron re-logs core metrics against sample count under this suffix; both Megatron parsers drop them.
SAMPLE_AXIS_SUFFIX = " vs samples"


class MegatronParser(TrainingParser):
    """MegatronRun: config from TB text summaries, steps from TB scalars."""

    STEP_MAPPING = MEGATRON_STEPS
    CONFIG_MAPPING = MEGATRON_CONFIG

    def get_config_path(self, tr: TestRun) -> Optional[Path]:
        # Megatron's config lives in the TB text stream, not a separate file; return any TB entry so the
        # can_parse() artifact check passes whenever the TB dir has content.
        return next(self.get_tb_dir(tr).iterdir(), None)

    def _read_scalars(self, tr: TestRun) -> list[Scalar]:
        return [s for s in super()._read_scalars(tr) if not s.tag.endswith(SAMPLE_AXIS_SUFFIX)]

    @staticmethod
    def _parse_literal(value: str) -> Any:
        # TB text config is all strings; parse "64" -> 64, "True" -> True, leave bare words ("rope") as str.
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    def get_tb_dir(self, tr: TestRun) -> Path:
        return tr.output_path / "tensorboard"

    def get_config(self, tr: TestRun) -> dict:
        return {tag: self._parse_literal(value) for tag, value in read_text(self.get_tb_dir(tr)).items()}

    def get_model_name(self, tr: TestRun) -> str:
        return tr.test.name


class MegatronBridgeParser(TrainingParser):
    """MegatronBridge: config from ConfigContainer.yaml, steps from TB scalars (deep nemo-run path)."""

    STEP_MAPPING = MEGATRON_BRIDGE_STEPS
    SCALE = MEGATRON_BRIDGE_SCALE
    CONFIG_MAPPING = MEGATRON_BRIDGE_CONFIG
    CONFIG_FILE = "ConfigContainer.yaml"

    def _read_scalars(self, tr: TestRun) -> list[Scalar]:
        return [s for s in super()._read_scalars(tr) if not s.tag.endswith(SAMPLE_AXIS_SUFFIX)]

    def get_tb_dir(self, tr: TestRun) -> Path:
        hits = sorted(tr.output_path.glob("**/tb_logs"))
        return hits[0] if hits else tr.output_path / "tb_logs"

    def get_config_path(self, tr: TestRun) -> Optional[Path]:
        hits = sorted(tr.output_path.glob(f"**/configs/{self.CONFIG_FILE}"))
        return hits[0] if hits else None

    def get_config(self, tr: TestRun) -> dict:
        path = self.get_config_path(tr)
        return yaml.safe_load(path.read_text()) if path else {}

    def get_model_name(self, tr: TestRun) -> str:
        return tr.test.cmd_args.model_recipe_name
