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

import types
from pathlib import Path
from typing import Any

import pytest

from cloudai.report_generator.training import parser as parser_mod
from cloudai.report_generator.training import tb_reader
from cloudai.report_generator.training.models import Scalar
from cloudai.report_generator.training.parser import MegatronBridgeParser, MegatronParser, NeMoRunParser
from cloudai.report_generator.training.report_generation_strategy import TrainingReportGenerationStrategy


def _scalars(rows: list[tuple]) -> list[Scalar]:
    return [Scalar(tag=tag, step=step, value=value, wall_time=wall_time) for tag, step, value, wall_time in rows]


def _system(gpus_per_node: int | None = 4, ntasks_per_node: int | None = None) -> Any:
    return types.SimpleNamespace(gpus_per_node=gpus_per_node, ntasks_per_node=ntasks_per_node)


def _tr(
    output_path: str | Path = "run", nnodes: int = 8, name: str = "t", template: str = "NeMoRun", **cmd_args: Any
) -> Any:
    test = types.SimpleNamespace(name=name, test_template_name=template, cmd_args=types.SimpleNamespace(**cmd_args))
    return types.SimpleNamespace(output_path=Path(output_path), nnodes=nnodes, test=test)


# --- steps ---------------------------------------------------------------------------------------


def test_build_steps_maps_value_tags():
    scalars = _scalars(
        [
            ("reduced_train_loss", 0, 10.0, 100.0),
            ("reduced_train_loss", 1, 9.0, 102.0),
            ("train_step_timing in s", 0, 0.5, 100.0),
            ("train_step_timing in s", 1, 0.6, 102.0),
            ("max_memory_reserved", 0, 100.0, 100.0),
            ("max_memory_reserved", 1, 110.0, 102.0),
            ("max_memory_allocated", 0, 90.0, 100.0),
            ("max_memory_allocated", 1, 95.0, 102.0),
            ("TFLOPS_per_GPU", 0, 500.0, 100.0),
            ("TFLOPS_per_GPU", 1, 510.0, 102.0),
        ]
    )
    steps = NeMoRunParser()._build_steps(scalars)

    assert [s.iteration for s in steps] == [0, 1]
    second = steps[1]
    assert (second.step_time_sec, second.loss, second.tflops_per_gpu) == (0.6, 9.0, 510.0)
    assert (second.memory_reserved_bytes, second.memory_allocated_bytes) == (110.0, 95.0)


def test_build_steps_applies_scale():
    # M-Bridge logs memory in gigabytes; SCALE multiplies it to bytes (other fields untouched).
    scalars = _scalars(
        [
            ("lm loss", 1, 9.0, 1.0),
            ("iteration-time", 1, 0.8, 1.0),
            ("memory/mem-reserved-gigabytes", 1, 75.0, 1.0),
            ("memory/mem-allocated-gigabytes", 1, 68.0, 1.0),
            ("throughput/tflops/device", 1, 600.0, 1.0),
        ]
    )
    step = MegatronBridgeParser()._build_steps(scalars)[0]

    assert step.memory_reserved_bytes == 75e9
    assert step.memory_allocated_bytes == 68e9
    assert step.loss == 9.0


def test_build_steps_keeps_step_missing_optional_tflops():
    # tflops_per_gpu is optional (NeMo skips FLOPs for unsupported families): the step survives with None.
    scalars = _scalars(
        [
            ("reduced_train_loss", 1, 9.0, 1.0),
            ("train_step_timing in s", 1, 0.8, 1.0),
            ("max_memory_reserved", 1, 1.0, 1.0),
            ("max_memory_allocated", 1, 1.0, 1.0),
        ]
    )
    steps = NeMoRunParser()._build_steps(scalars)

    assert [s.iteration for s in steps] == [1]
    assert steps[0].tflops_per_gpu is None
    assert steps[0].loss == 9.0


def test_build_steps_drops_step_missing_required_tag():
    # Step 2 lacks a required tag (loss) -> it's dropped (logged); the complete step survives.
    scalars = _scalars(
        [
            ("reduced_train_loss", 1, 9.0, 1.0),
            ("train_step_timing in s", 1, 0.8, 1.0),
            ("max_memory_reserved", 1, 1.0, 1.0),
            ("max_memory_allocated", 1, 1.0, 1.0),
            ("TFLOPS_per_GPU", 1, 5.0, 1.0),
            ("train_step_timing in s", 2, 0.8, 1.0),
            ("max_memory_reserved", 2, 1.0, 1.0),
            ("max_memory_allocated", 2, 1.0, 1.0),
            ("TFLOPS_per_GPU", 2, 5.0, 1.0),
        ]
    )
    steps = NeMoRunParser()._build_steps(scalars)

    assert [s.iteration for s in steps] == [1]


def test_build_steps_empty_returns_empty_list():
    assert NeMoRunParser()._build_steps([]) == []


# --- reading -------------------------------------------------------------------------------------


def test_megatron_read_scalars_drops_sample_axis_twins(monkeypatch):
    # Megatron double-logs core metrics on a "* vs samples" axis; the Megatron parser drops them at read.
    raw = _scalars(
        [
            ("lm loss", 1, 9.0, 100.0),
            ("iteration-time", 1, 0.8, 100.0),
            ("lm loss vs samples", 8, 9.0, 100.0),
            ("learning-rate vs samples", 8, 0.1, 100.0),
        ]
    )
    monkeypatch.setattr(parser_mod, "read_scalars", lambda _tb_dir: raw)

    kept = MegatronParser()._read_scalars(_tr())
    assert [s.tag for s in kept] == ["lm loss", "iteration-time"]


class _FakeFrame:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    @property
    def empty(self) -> bool:
        return not self.rows

    def sort_values(self, column: str) -> "_FakeFrame":
        return _FakeFrame(sorted(self.rows, key=lambda row: row[column]))

    def drop_duplicates(self, subset: list[str], keep: str) -> "_FakeFrame":
        assert subset == ["tag"]
        assert keep == "last"
        seen = set()
        rows = []
        for row in reversed(self.rows):
            if row["tag"] in seen:
                continue
            seen.add(row["tag"])
            rows.append(row)
        return _FakeFrame(list(reversed(rows)))

    def to_dict(self, orient: str) -> list[dict[str, Any]]:
        assert orient == "records"
        return self.rows


def test_read_text_keeps_latest_wall_time(monkeypatch):
    rows = [
        {"tag": "tensor_model_parallel_size", "value": "2", "wall_time": 100.0},
        {"tag": "tensor_model_parallel_size", "value": "4", "wall_time": 200.0},
        {"tag": "pipeline_model_parallel_size", "value": "1", "wall_time": 100.0},
    ]

    class FakeSummaryReader:
        def __init__(self, path, extra_columns):
            assert path == "tb"
            assert extra_columns == {"wall_time"}

        @property
        def text(self):
            return _FakeFrame(rows)

    monkeypatch.setattr(tb_reader, "SummaryReader", FakeSummaryReader)

    assert tb_reader.read_text(Path("tb")) == {
        "tensor_model_parallel_size": "4",
        "pipeline_model_parallel_size": "1",
    }


# --- config --------------------------------------------------------------------------------------


def test_build_config_resolves_paths_and_computes_fields():
    raw = {
        "data": {"micro_batch_size": 1, "global_batch_size": 8},
        "parallelism": {"tensor_model_parallel_size": 4, "pipeline_model_parallel_size": 1, "context_parallel_size": 1},
        "model": {"num_layers": 30},
    }
    config = NeMoRunParser()._build_config(
        raw, _tr(nnodes=8, template="NeMoRun", recipe_name="gpt3"), _system(gpus_per_node=4)
    )

    assert config.test_template_name == "NeMoRun"  # CloudAI-computed
    assert config.micro_batch_size == 1  # nested dotted-path resolve
    assert config.num_layers == 30
    assert config.tensor_parallel_size == 4
    assert config.model_name == "gpt3"  # CloudAI-computed
    assert (config.num_nodes, config.world_size) == (8, 32)  # 8 nodes x 4 gpus
    assert config.data_parallel_size == 8  # 32 / (tp4 * pp1 * cp1)


def test_build_config_leaves_world_size_none_without_gpus_per_node():
    # No gpus_per_node/ntasks_per_node on the system: world_size/data_parallel_size stay None, rest still resolves.
    raw = {"parallelism": {"tensor_model_parallel_size": 4, "pipeline_model_parallel_size": 1}}
    config = NeMoRunParser()._build_config(raw, _tr(nnodes=8, recipe_name="gpt3"), _system(gpus_per_node=None))

    assert config.world_size is None
    assert config.data_parallel_size is None
    assert config.num_nodes == 8  # still set
    assert config.tensor_parallel_size == 4


def test_megatron_config_parses_string_literals(monkeypatch):
    # Megatron config comes from TB text (all strings); values are parsed back to native types.
    text = {
        "micro_batch_size": "1",
        "tensor_model_parallel_size": "2",
        "pipeline_model_parallel_size": "2",
        "context_parallel_size": "1",
        "sequence_parallel": "True",
    }
    monkeypatch.setattr(parser_mod, "read_text", lambda _tb_dir: text)

    parser = MegatronParser()
    config = parser._build_config(parser.get_config(_tr()), _tr(nnodes=16, name="dsv3"), _system(gpus_per_node=4))

    assert config.micro_batch_size == 1  # "1" -> int
    assert config.sequence_parallel is True  # "True" -> bool
    assert config.model_name == "dsv3"
    assert config.data_parallel_size == 16  # 64 / (tp2 * pp2 * cp1)


def test_compute_data_parallel_size_rejects_invalid_topology():
    parser = NeMoRunParser()
    parallel = {"tensor_model_parallel_size": 4, "pipeline_model_parallel_size": 1, "context_parallel_size": 1}
    with pytest.raises(ValueError, match="world_size"):  # world_size 34 is not a multiple of tp*pp*cp=4
        parser._build_config({"parallelism": parallel}, _tr(nnodes=17, recipe_name="x"), _system(gpus_per_node=2))
    with pytest.raises(ValueError, match="tensor_parallel_size"):  # tp missing from the parsed config
        parser._build_config({"parallelism": {}}, _tr(recipe_name="x"), _system())


# --- can_parse -----------------------------------------------------------------------------------


def test_can_parse_requires_nonempty_tb_dir(tmp_path):
    parser = NeMoRunParser()
    assert parser.can_parse(_tr(output_path=tmp_path)) is False  # no TB dir yet

    tb_dir = tmp_path / "tb_logs" / "default"
    tb_dir.mkdir(parents=True)
    (tb_dir / "events.out.tfevents").write_text("x")
    assert parser.can_parse(_tr(output_path=tmp_path)) is False  # file-backed parsers also need config

    (tmp_path / "nemo_config.json").write_text("{}")
    assert parser.can_parse(_tr(output_path=tmp_path)) is True


def test_megatron_config_path_points_to_tb_event_file(tmp_path):
    tb_dir = tmp_path / "tensorboard"
    tb_dir.mkdir()
    event_file = tb_dir / "events.out.tfevents"
    event_file.write_text("x")

    assert MegatronParser().get_config_path(_tr(output_path=tmp_path)) == event_file


# --- report generation ---------------------------------------------------------------------------


def test_training_report_json_default_supports_scalar_item():
    class ScalarWithItem:
        def item(self):
            return 7

    class StringFallback:
        def __str__(self):
            return "fallback"

    assert TrainingReportGenerationStrategy._json_default(ScalarWithItem()) == 7
    assert TrainingReportGenerationStrategy._json_default(StringFallback()) == "fallback"
