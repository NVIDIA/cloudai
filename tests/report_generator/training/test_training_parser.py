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

import logging
import types
from pathlib import Path
from typing import Any

import pytest

from cloudai.report_generator.training import parser as parser_mod
from cloudai.report_generator.training import tb_reader
from cloudai.report_generator.training.models import Scalar, TrainingStep
from cloudai.report_generator.training.parser import MegatronBridgeParser, MegatronParser, NeMoRunParser
from cloudai.report_generator.training.report_generation_strategy import TrainingReportGenerationStrategy


def _scalars(rows: list[tuple]) -> list[Scalar]:
    return [Scalar(tag=tag, step=step, value=value, wall_time=wall_time) for tag, step, value, wall_time in rows]


def _system(gpus_per_node: int | None = 4, ntasks_per_node: int | None = None) -> Any:
    return types.SimpleNamespace(gpus_per_node=gpus_per_node, ntasks_per_node=ntasks_per_node)


class _Test(types.SimpleNamespace):
    """Stand-in for the pydantic TestDefinition: model_dump() returns nested cmd_args/nsys dicts."""

    def model_dump(self) -> dict[str, Any]:
        dumped = dict(vars(self))
        dumped["cmd_args"] = dict(vars(self.cmd_args))
        dumped["nsys"] = dict(vars(self.nsys)) if self.nsys is not None else None
        return dumped


def _tr(
    output_path: str | Path = "run",
    nnodes: int = 8,
    name: str = "t",
    template: str = "NeMoRun",
    nsys: Any = None,
    extra_cmd_args: dict[str, Any] | None = None,
    training_report: dict[str, Any] | None = None,
    **cmd_args: Any,
) -> Any:
    test = _Test(
        name=name,
        test_template_name=template,
        cmd_args=types.SimpleNamespace(**cmd_args),
        nsys=nsys,
        extra_cmd_args=extra_cmd_args or {},
        training_report=training_report,
    )
    return types.SimpleNamespace(output_path=Path(output_path), nnodes=nnodes, test=test)


def _nsys(enable: bool) -> Any:
    return types.SimpleNamespace(enable=enable)


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
    parser = NeMoRunParser()
    parser.get_model_config = lambda tr: raw
    config = parser._build_config(_tr(nnodes=8, template="NeMoRun", recipe_name="gpt3"), _system(gpus_per_node=4))

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
    parser = NeMoRunParser()
    parser.get_model_config = lambda tr: raw
    config = parser._build_config(_tr(nnodes=8, recipe_name="gpt3"), _system(gpus_per_node=None))

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
    config = parser._build_config(_tr(nnodes=16, name="dsv3"), _system(gpus_per_node=4))

    assert config.micro_batch_size == 1  # "1" -> int
    assert config.sequence_parallel is True  # "True" -> bool
    assert config.model_name == "dsv3"
    assert config.data_parallel_size == 16  # 64 / (tp2 * pp2 * cp1)


def test_compute_data_parallel_size_rejects_invalid_topology():
    parser = NeMoRunParser()
    parallel = {"tensor_model_parallel_size": 4, "pipeline_model_parallel_size": 1, "context_parallel_size": 1}
    parser.get_model_config = lambda tr: {"parallelism": parallel}
    with pytest.raises(ValueError, match="world_size"):  # world_size 34 is not a multiple of tp*pp*cp=4
        parser._build_config(_tr(nnodes=17, recipe_name="x"), _system(gpus_per_node=2))
    parser.get_model_config = lambda tr: {"parallelism": {}}
    with pytest.raises(ValueError, match="tensor_parallel_size"):  # tp missing from the parsed config
        parser._build_config(_tr(recipe_name="x"), _system())


# --- profiling -----------------------------------------------------------------------------------


def test_nemo_profiling_reads_nsys_and_callback_steps():
    # enable maps from [nsys]; the step bounds are extracted from extra_cmd_args by suffix (index-agnostic).
    tr = _tr(
        nsys=_nsys(True),
        extra_cmd_args={"trainer.callbacks[2].start_step": "20", "trainer.callbacks[2].end_step": "25"},
    )
    assert NeMoRunParser()._resolve_test_config(tr) == {
        "profiling_enabled": True,
        "profiling_start_step": 20,
        "profiling_stop_step": 25,
    }


def test_nemo_profiling_disabled_and_no_steps():
    tr = _tr(nsys=_nsys(False))
    assert NeMoRunParser()._resolve_test_config(tr) == {"profiling_enabled": False}


def test_nemo_profiling_step_extraction_is_index_agnostic():
    # a different user-chosen callback index still resolves via the suffix match
    tr = _tr(nsys=_nsys(True), extra_cmd_args={"trainer.callbacks[0].start_step": "7"})
    resolved = NeMoRunParser()._resolve_test_config(tr)
    assert resolved["profiling_start_step"] == 7
    assert "profiling_stop_step" not in resolved


def test_megatron_profiling_reads_profile_steps():
    tr = _tr(nsys=_nsys(True), profile_step_start=50, profile_step_end=55)
    assert MegatronParser()._resolve_test_config(tr) == {
        "profiling_enabled": True,
        "profiling_start_step": 50,
        "profiling_stop_step": 55,
    }


def test_megatron_profiling_absent_is_dropped():
    # no [nsys] section and no profile_step_* args: everything is dropped, so config keeps model defaults.
    tr = _tr(nsys=None)
    assert MegatronParser()._resolve_test_config(tr) == {}


def test_megatron_bridge_profiling_reads_typed_fields():
    tr = _tr(enable_nsys=True, profiling_start_step=10, profiling_stop_step=12)
    assert MegatronBridgeParser()._resolve_test_config(tr) == {
        "profiling_enabled": True,
        "profiling_start_step": 10,
        "profiling_stop_step": 12,
    }


def test_build_config_sets_profiling_fields():
    # M-Bridge exposes enable + step bounds as typed cmd_args, so _build_config folds all three into the config.
    raw = {"model": {"tensor_model_parallel_size": 4, "pipeline_model_parallel_size": 1}}
    parser = MegatronBridgeParser()
    parser.get_model_config = lambda tr: raw
    tr = _tr(model_recipe_name="gpt3", enable_nsys=True, profiling_start_step=3, profiling_stop_step=7)
    config = parser._build_config(tr, _system(gpus_per_node=4))

    assert config.profiling_enabled is True
    assert (config.profiling_start_step, config.profiling_stop_step) == (3, 7)


def test_build_config_reads_aggregation_flags_from_toml():
    parser = NeMoRunParser()
    parser.get_model_config = lambda tr: {}
    tr = _tr(recipe_name="gpt3", training_report={"exclude_start_steps": 10, "exclude_post_profiling_steps": 3})
    config = parser._build_config(tr, _system(gpus_per_node=None))
    assert (config.exclude_start_steps, config.exclude_post_profiling_steps) == (10, 3)


def test_build_config_aggregation_flags_default_when_absent():
    parser = NeMoRunParser()
    parser.get_model_config = lambda tr: {}
    config = parser._build_config(_tr(recipe_name="gpt3"), _system(gpus_per_node=None))
    assert (config.exclude_start_steps, config.exclude_post_profiling_steps) == (5, 2)


# --- aggregation ---------------------------------------------------------------------------------


def _agg_config(**overrides: Any) -> Any:
    base = {
        "profiling_enabled": False,
        "profiling_start_step": None,
        "profiling_stop_step": None,
        "exclude_start_steps": 0,
        "exclude_post_profiling_steps": 0,
    }
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _agg_steps(step_times: list[float]) -> list[TrainingStep]:
    return [
        TrainingStep(
            iteration=i,
            step_time_sec=t,
            loss=1.0,
            memory_reserved_bytes=1.0,
            memory_allocated_bytes=1.0,
            tflops_per_gpu=None,
        )
        for i, t in enumerate(step_times)
    ]


def test_filter_steps_drops_start_then_profiling_window():
    steps = _agg_steps([1.0] * 10)  # iterations 0..9
    config = _agg_config(
        exclude_start_steps=2,
        profiling_enabled=True,
        profiling_start_step=4,
        profiling_stop_step=5,
        exclude_post_profiling_steps=1,
    )
    # drop first 2 (0,1) then the profiling window [4, 5+1] -> 4,5,6
    assert [s.iteration for s in NeMoRunParser._filter_steps(steps, config)] == [2, 3, 7, 8, 9]


def test_filter_steps_ignores_profiling_window_when_disabled():
    steps = _agg_steps([1.0] * 6)
    config = _agg_config(exclude_start_steps=2, profiling_enabled=False, profiling_start_step=3, profiling_stop_step=4)
    assert [s.iteration for s in NeMoRunParser._filter_steps(steps, config)] == [2, 3, 4, 5]


def test_aggregate_computes_per_metric_stats():
    agg = NeMoRunParser()._aggregate(_agg_steps([10.0, 20.0, 30.0]), _agg_config())
    assert agg is not None
    st = agg.step_time_sec
    assert st is not None
    assert st.mean == 20.0
    assert (st.min, st.max) == (10.0, 30.0)
    assert st.std == pytest.approx(8.16496, rel=1e-4)  # population stdev
    assert st.mean <= st.t95 <= st.max


def test_aggregate_tflops_is_none_when_absent():
    agg = NeMoRunParser()._aggregate(_agg_steps([1.0, 2.0]), _agg_config())
    assert agg is not None
    assert agg.tflops_per_gpu is None  # tflops is None on every step
    assert agg.step_time_sec is not None


def test_aggregate_returns_none_when_all_filtered_out():
    # only 2 steps but exclude_start_steps=5 -> nothing remains -> aggregation is None
    assert NeMoRunParser()._aggregate(_agg_steps([1.0, 2.0]), _agg_config(exclude_start_steps=5)) is None


# --- glob lookup ---------------------------------------------------------------------------------


def test_get_from_dict_warns_on_multiple_glob_matches(caplog):
    source = {"extra_cmd_args": {"a.start_step": "1", "b.start_step": "2"}}
    with caplog.at_level(logging.WARNING):
        result = NeMoRunParser._get_from_dict(source, "extra_cmd_args.*start_step")
    assert result == "1"  # first match kept
    assert "Multiple config values match" in caplog.text


def test_get_from_dict_single_glob_match_does_not_warn(caplog):
    source = {"extra_cmd_args": {"trainer.callbacks[2].start_step": "20"}}
    with caplog.at_level(logging.WARNING):
        result = NeMoRunParser._get_from_dict(source, "extra_cmd_args.*start_step")
    assert result == "20"
    assert "Multiple config values match" not in caplog.text


# --- can_parse -----------------------------------------------------------------------------------


def test_can_parse_requires_nonempty_tb_dir(tmp_path):
    parser = NeMoRunParser()
    assert parser.can_parse(_tr(output_path=tmp_path)) is False  # no TB dir yet

    tb_dir = tmp_path / "tb_logs" / "default"
    tb_dir.mkdir(parents=True)
    (tb_dir / "placeholder.txt").write_text("x")
    assert parser.can_parse(_tr(output_path=tmp_path)) is False  # non-empty dir without event files

    (tb_dir / "custom.tfevents.log").write_text("x")
    assert parser.can_parse(_tr(output_path=tmp_path)) is False  # file-backed parsers also need config

    (tmp_path / "nemo_config.json").write_text('{"model": {}}')
    assert parser.can_parse(_tr(output_path=tmp_path)) is True


@pytest.mark.parametrize(
    ("parser", "tb_path", "config_path", "config_contents"),
    (
        pytest.param(NeMoRunParser(), "tb_logs/default", "nemo_config.json", "", id="nemo-empty-file"),
        pytest.param(NeMoRunParser(), "tb_logs/default", "nemo_config.json", "{}", id="nemo-empty-object"),
        pytest.param(NeMoRunParser(), "tb_logs/default", "nemo_config.json", "{", id="nemo-malformed"),
        pytest.param(
            MegatronBridgeParser(),
            "experiment/tb_logs",
            "experiment/configs/ConfigContainer.yaml",
            "",
            id="bridge-empty-file",
        ),
        pytest.param(
            MegatronBridgeParser(),
            "experiment/tb_logs",
            "experiment/configs/ConfigContainer.yaml",
            "{}",
            id="bridge-empty-object",
        ),
        pytest.param(
            MegatronBridgeParser(),
            "experiment/tb_logs",
            "experiment/configs/ConfigContainer.yaml",
            "key: [",
            id="bridge-malformed",
        ),
    ),
)
def test_can_parse_rejects_invalid_file_config(tmp_path, parser, tb_path, config_path, config_contents):
    tb_dir = tmp_path / tb_path
    tb_dir.mkdir(parents=True)
    (tb_dir / "events.out.tfevents.1").write_text("x")
    artifact = tmp_path / config_path
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(config_contents)

    assert parser.can_parse(_tr(output_path=tmp_path)) is False


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
