from typing import Any

import pytest
from pydantic import ValidationError

from cloudai._core.test import DSEValuesRange
from cloudai.test_definitions.gpt import GPTTestDefinition

GPT_TEST_DEFINITION = {
    "name": "n",
    "description": "d",
    "test_template_name": "ttn",
    "cmd_args": {
        "docker_image_url": "docker://url",
        "fdl_config": "/path",
    },
}


@pytest.mark.parametrize(
    "dse_field,value",
    [("fdl_config", ["/p1"]), ("fdl.num_groups", [1, 2])],
)
def test_dse_valid(dse_field: str, value: Any):
    data = GPT_TEST_DEFINITION.copy()
    data["dse"] = {dse_field: value}
    gpt = GPTTestDefinition(**data)

    assert isinstance(gpt, GPTTestDefinition)
    assert gpt.dse[dse_field] == value


@pytest.mark.parametrize("value", [1, "1", 1.0, 1j, None])
def test_dse_invalid_field_top_type(value: Any):
    data = GPT_TEST_DEFINITION.copy()
    data["dse"] = {"fdl_config": value}

    with pytest.raises(ValidationError) as exc_info:
        GPTTestDefinition(**data)

    errors = [err["msg"] for err in exc_info.value.errors(include_url=False)]
    assert len(errors) == 2
    assert "Input should be a valid dictionary or instance of DSEValuesRange" in errors
    assert "Input should be a valid list" in errors


@pytest.mark.parametrize("field", ["fdl_configA", "fdl.num_groupsA"])
def test_dse_raises_on_unknown_field(field: str):
    data = GPT_TEST_DEFINITION.copy()
    data["dse"] = {field: [1]}

    with pytest.raises(ValidationError) as exc_info:
        GPTTestDefinition(**data)

    errors = [err["msg"] for err in exc_info.value.errors(include_url=False)]
    assert len(errors) == 1
    assert errors[0] == f"Value error, 'GPTCmdArgs' doesn't have field '{field}'"


def test_dse_invalid_field_value_type():
    data = GPT_TEST_DEFINITION.copy()
    data["dse"] = {"fdl_config": ["/p", 1]}

    with pytest.raises(ValidationError) as exc_info:
        GPTTestDefinition(**data)

    errors = [err["msg"] for err in exc_info.value.errors(include_url=False)]
    assert len(errors) == 1
    assert "Invalid type of value=1 ('int')" in errors[0]


@pytest.mark.parametrize(
    "input,output",
    [
        ({"start": 1, "end": 2}, DSEValuesRange(start=1, end=2)),
        ({"start": 1, "end": 2, "step": 0.5}, DSEValuesRange(start=1, end=2, step=0.5)),
    ],
)
def test_dse_range(input: dict, output: DSEValuesRange):
    data = GPT_TEST_DEFINITION.copy()
    data["dse"] = {"fdl.num_gpus": input}

    gpt = GPTTestDefinition(**data)

    assert isinstance(gpt.dse["fdl.num_gpus"], DSEValuesRange)
    assert gpt.dse["fdl.num_gpus"] == output
    assert gpt.dse["fdl.num_gpus"].step
