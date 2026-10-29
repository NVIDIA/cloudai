# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import importlib.util
from pathlib import Path

import pytest

griffe = pytest.importorskip("griffe")
pytest.importorskip("griffe_pydantic")
pytest.importorskip("jinja2")

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("generate_reference", ROOT / "doc/generate_reference.py")
assert SPEC is not None and SPEC.loader is not None
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


@pytest.fixture(scope="module")
def loader():
    result = griffe.GriffeLoader(
        search_paths=[str(ROOT / "src")],
        extensions=griffe.load_extensions("griffe_pydantic"),
        docstring_parser=griffe.Parser.google,
    )
    result.load("cloudai")
    result.resolve_aliases(external=True)
    return result


@pytest.mark.parametrize(
    "symbol",
    [
        "cloudai.systems.slurm.slurm_system.SlurmSystem",
        "cloudai.workloads.nixl_bench.nixl_bench.NIXLBenchCmdArgs",
        "cloudai.workloads.vllm.vllm.VllmArgs",
    ],
)
def test_all_model_fields_and_descriptions(loader, symbol):
    result = reference.describe_class(symbol, loader.modules_collection[symbol])
    actual = {field["name"]: field for field in result["fields"]}
    model = reference.load_class(symbol)
    assert set(actual) == set(model.model_fields)
    for name, field in model.model_fields.items():
        assert actual[name]["required"] == field.is_required()
        if field.description:
            assert actual[name]["description"] == field.description


def test_runtime_default_and_constraints(loader):
    symbol = "cloudai.workloads.nixl_bench.nixl_bench.NIXLBenchCmdArgs"
    result = reference.describe_class(symbol, loader.modules_collection[symbol])
    fields = {field["name"]: field for field in result["fields"]}
    assert fields["etcd_endpoints"]["default"] == "'http://$NIXL_ETCD_ENDPOINTS'"
    assert fields["asio_port"]["default"] == "12345"
    assert fields["asio_port"]["constraints"] == ["Ge(ge=1)", "Le(le=65535)"]
    assert fields["docker_image_url"]["required"]


def test_system_exclusions_and_factory(loader):
    symbol = "cloudai.systems.slurm.slurm_system.SlurmSystem"
    result = reference.describe_class(symbol, loader.modules_collection[symbol], ["cmd_shell"], fields_only=True)
    fields = {field["name"]: field for field in result["fields"]}
    assert "cmd_shell" not in fields
    assert fields["extra_sbatch_args"]["factory"] == "list"
    assert not fields["extra_sbatch_args"]["required"]
    assert not result["members"]


def test_dataclass_fields(loader):
    symbol = "cloudai.report_generator.training.models.TrainingConfig"
    result = reference.describe_class(symbol, loader.modules_collection[symbol], ["__init__"])
    fields = {field["name"]: field for field in result["fields"]}
    assert len(fields) == 47
    assert fields["test_id"]["required"]
    assert fields["container_image"]["default"] == "''"
    assert all(member["name"] != "__init__" for member in result["members"])


def test_mdx_prose_escaping():
    assert reference.mdx_text("x < y and {name}") == "x &lt; y and &#123;name&#125;"
