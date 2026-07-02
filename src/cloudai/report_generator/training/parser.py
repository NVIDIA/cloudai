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
import fnmatch
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Optional

import yaml

from cloudai.core import System, TestRun

from .mappings import (
    MEGATRON_BRIDGE_MODEL_CONFIG,
    MEGATRON_BRIDGE_SCALE,
    MEGATRON_BRIDGE_STEPS,
    MEGATRON_BRIDGE_TEST_CONFIG,
    MEGATRON_MODEL_CONFIG,
    MEGATRON_STEPS,
    MEGATRON_TEST_CONFIG,
    NEMO_MODEL_CONFIG,
    NEMO_STEPS,
    NEMO_TEST_CONFIG,
)
from .models import OPTIONAL_STEP_FIELDS, Scalar, TrainingConfig, TrainingResults, TrainingStep
from .tb_reader import read_scalars, read_text


class TrainingParser(ABC):
    """Base parser: a workload's run artifacts -> TrainingResults."""

    STEP_MAPPING: ClassVar[dict[str, str]]  # TrainingStep field -> TB scalar tag (read from the value column)
    SCALE: ClassVar[dict[str, float]] = {}  # TrainingStep field -> unit factor (e.g. m-bridge gigabytes -> bytes)
    MODEL_CONFIG_MAPPING: ClassVar[dict[str, str]]  # TrainingConfig field -> path in the framework's resolved config
    TEST_CONFIG_MAPPING: ClassVar[dict[str, str]] = {}  # TrainingConfig field -> dotted path in the TestDefinition

    @abstractmethod
    def get_tb_dir(self, tr: TestRun) -> Path:
        """Directory holding the run's TensorBoard event files."""

    @abstractmethod
    def get_config_path(self, tr: TestRun) -> Optional[Path]:
        """Config artifact path used by the parser, or None when it cannot be found."""

    @abstractmethod
    def get_model_config(self, tr: TestRun) -> dict:
        """Read the framework's resolved config artifact into a (possibly nested) dict."""

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
        config: TrainingConfig = self._build_config(tr, system)
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

    def _build_config(self, tr: TestRun, system: System) -> TrainingConfig:
        """Map the framework + test config into TrainingConfig, then fill the CloudAI-computed fields."""
        field_values: dict[str, Any] = {}
        field_values.update(self._resolve_model_config(tr))
        field_values.update(self._resolve_test_config(tr))
        config = TrainingConfig(**field_values)
        config.test_template_name = tr.test.test_template_name
        config.num_nodes = tr.nnodes
        config.model_name = self.get_model_name(tr)
        gpus_per_node = getattr(system, "gpus_per_node", None) or getattr(system, "ntasks_per_node", None)
        if gpus_per_node:
            world_size = config.num_nodes * gpus_per_node
            config.world_size = world_size
            config.data_parallel_size = self._compute_data_parallel_size(config, world_size)
        else:
            logging.warning(
                f"{type(self).__name__}: system has no gpus_per_node/ntasks_per_node; "
                "world_size and data_parallel_size left unset"
            )
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
    def _get_from_dict(source: dict, path: str) -> Any:
        """Read a dotted path; a '*' segment globs that key against the current dict's keys (first match)."""
        value: Any = source
        for part in path.split("."):
            if not isinstance(value, dict):
                return None
            if "*" not in part:
                value = value.get(part)
                continue
            matches = fnmatch.filter(value, part)
            if len(matches) > 1:
                logging.warning(f"Multiple config values match '{part}' ({matches}); using '{matches[0]}'.")
            value = value[matches[0]] if matches else None
        return value

    def _compute_data_parallel_size(self, config: TrainingConfig, world_size: int) -> int:
        """world_size / (tensor*pipeline*context parallel). EP is carved out of DP, not world_size."""
        cls = type(self).__name__
        tp = config.tensor_parallel_size
        pp = config.pipeline_parallel_size or 1
        cp = config.context_parallel_size or 1
        if not tp:
            raise ValueError(f"{cls}: tensor_parallel_size missing/invalid in parsed config (got {tp!r})")
        parallel = tp * pp * cp
        if world_size % parallel != 0:
            raise ValueError(f"{cls}: world_size {world_size} not a multiple of tp*pp*cp={parallel}; check topology")
        return world_size // parallel

    def _resolve_model_config(self, tr: TestRun) -> dict[str, Any]:
        """Resolve MODEL_CONFIG_MAPPING against the framework's resolved config (values already native)."""
        source = self.get_model_config(tr)
        return {field: self._get_from_dict(source, path) for field, path in self.MODEL_CONFIG_MAPPING.items()}

    def _resolve_test_config(self, tr: TestRun) -> dict[str, Any]:
        """Resolve TEST_CONFIG_MAPPING against the run's TestDefinition; coerce values, drop misses (keep defaults)."""
        source = tr.test.model_dump()
        return {
            field: self._parse_literal(value)
            for field, path in self.TEST_CONFIG_MAPPING.items()
            if (value := self._get_from_dict(source, path)) is not None
        }

    @staticmethod
    def _parse_literal(value: Any) -> Any:
        """Parse a string value to its native type ("20" -> 20, "True" -> True); pass non-strings through."""
        if not isinstance(value, str):
            return value
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value


class NeMoRunParser(TrainingParser):
    """NeMoRun: config from nemo_config.json, steps from TB scalars."""

    STEP_MAPPING = NEMO_STEPS
    MODEL_CONFIG_MAPPING = NEMO_MODEL_CONFIG
    TEST_CONFIG_MAPPING = NEMO_TEST_CONFIG
    CONFIG_FILE = "nemo_config.json"

    def get_tb_dir(self, tr: TestRun) -> Path:
        return tr.output_path / "tb_logs" / "default"

    def get_config_path(self, tr: TestRun) -> Path:
        return tr.output_path / self.CONFIG_FILE

    def get_model_config(self, tr: TestRun) -> dict:
        return json.loads(self.get_config_path(tr).read_text())

    def get_model_name(self, tr: TestRun) -> str:
        return tr.test.cmd_args.recipe_name


# Megatron re-logs core metrics against sample count under this suffix; both Megatron parsers drop them.
SAMPLE_AXIS_SUFFIX = " vs samples"


class MegatronParser(TrainingParser):
    """MegatronRun: config from TB text summaries, steps from TB scalars."""

    STEP_MAPPING = MEGATRON_STEPS
    MODEL_CONFIG_MAPPING = MEGATRON_MODEL_CONFIG
    TEST_CONFIG_MAPPING = MEGATRON_TEST_CONFIG

    def get_config_path(self, tr: TestRun) -> Optional[Path]:
        # Megatron's config lives in the TB text stream, not a separate file; return any TB entry so the
        # can_parse() artifact check passes whenever the TB dir has content.
        return next(self.get_tb_dir(tr).iterdir(), None)

    def _read_scalars(self, tr: TestRun) -> list[Scalar]:
        return [s for s in super()._read_scalars(tr) if not s.tag.endswith(SAMPLE_AXIS_SUFFIX)]

    def get_tb_dir(self, tr: TestRun) -> Path:
        return tr.output_path / "tensorboard"

    def get_model_config(self, tr: TestRun) -> dict:
        # TB text config is all strings; _parse_literal turns "64" -> 64, "True" -> True, leaves bare words as str.
        return {tag: self._parse_literal(value) for tag, value in read_text(self.get_tb_dir(tr)).items()}

    def get_model_name(self, tr: TestRun) -> str:
        return tr.test.name


class MegatronBridgeParser(TrainingParser):
    """MegatronBridge: config from ConfigContainer.yaml, steps from TB scalars (deep nemo-run path)."""

    STEP_MAPPING = MEGATRON_BRIDGE_STEPS
    SCALE = MEGATRON_BRIDGE_SCALE
    MODEL_CONFIG_MAPPING = MEGATRON_BRIDGE_MODEL_CONFIG
    TEST_CONFIG_MAPPING = MEGATRON_BRIDGE_TEST_CONFIG
    CONFIG_FILE = "ConfigContainer.yaml"

    def _read_scalars(self, tr: TestRun) -> list[Scalar]:
        return [s for s in super()._read_scalars(tr) if not s.tag.endswith(SAMPLE_AXIS_SUFFIX)]

    def get_tb_dir(self, tr: TestRun) -> Path:
        hits = sorted(tr.output_path.glob("**/tb_logs"))
        return hits[0] if hits else tr.output_path / "tb_logs"

    def get_config_path(self, tr: TestRun) -> Optional[Path]:
        hits = sorted(tr.output_path.glob(f"**/configs/{self.CONFIG_FILE}"))
        return hits[0] if hits else None

    def get_model_config(self, tr: TestRun) -> dict:
        path = self.get_config_path(tr)
        return yaml.safe_load(path.read_text()) if path else {}

    def get_model_name(self, tr: TestRun) -> str:
        return tr.test.cmd_args.model_recipe_name
