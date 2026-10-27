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

"""Generate Fern snippets from Python objects; authored MDX is never rewritten."""

import dataclasses
import html
import importlib
import inspect
import sys
from pathlib import Path

import griffe
import jinja2
import pydantic
import yaml

ROOT = Path(__file__).resolve().parents[1]


def mdx_text(value):
    """Escape prose leaves for MDX without interpreting or rewriting Markdown."""
    return html.escape(str(value), quote=False).translate(str.maketrans({"{": "&#123;", "}": "&#125;"}))


def load_class(symbol):
    module, name = symbol.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


def describe_fields(cls, obj, exclude):
    """Read fields without evaluating their default factories."""
    fields = []
    if issubclass(cls, pydantic.BaseModel):
        for name, field in cls.model_fields.items():
            if name in exclude:
                continue
            source = obj.all_members.get(name)
            description = field.description or (source.docstring.value if source and source.docstring else "")
            factory = field.default_factory
            fields.append(
                {
                    "name": name,
                    "annotation": inspect.formatannotation(field.annotation),
                    "description": description,
                    "required": field.is_required(),
                    "default": repr(field.default) if not field.is_required() and factory is None else None,
                    "factory": getattr(factory, "__qualname__", str(factory)) if factory else None,
                    "constraints": [repr(item) for item in field.metadata],
                    "alias": field.alias,
                }
            )
    elif dataclasses.is_dataclass(cls):
        for field in dataclasses.fields(cls):
            if field.name in exclude:
                continue
            source = obj.all_members.get(field.name)
            fields.append(
                {
                    "name": field.name,
                    "annotation": inspect.formatannotation(field.type),
                    "description": source.docstring.value if source and source.docstring else "",
                    "required": field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING,
                    "default": repr(field.default) if field.default is not dataclasses.MISSING else None,
                    "factory": getattr(field.default_factory, "__qualname__", str(field.default_factory))
                    if field.default_factory is not dataclasses.MISSING
                    else None,
                    "constraints": [],
                    "alias": None,
                }
            )
    return fields


def describe_members(cls, obj, exclude):
    """Read documented public members, excluding model implementation details."""
    members = []
    for name, member in obj.all_members.items():
        if member.canonical_path.startswith("pydantic."):
            continue
        if name in exclude or (name.startswith("_") and name != "__init__") or name.startswith("model_"):
            continue
        if not member.docstring:
            continue
        runtime = inspect.getattr_static(cls, name, None)
        # Pydantic validators were hidden by the previous documentation configuration.
        decorators = getattr(cls, "__pydantic_decorators__", None)
        if decorators and (name in decorators.field_validators or name in decorators.model_validators):
            continue
        if isinstance(runtime, (staticmethod, classmethod)):
            runtime = runtime.__func__
        if isinstance(runtime, property):
            signature = ": " + str(getattr(member, "annotation", ""))
        elif callable(runtime):
            signature = str(inspect.signature(runtime))
        else:
            signature = ": " + str(getattr(member, "annotation", ""))
        members.append({"name": name, "signature": signature, "sections": member.docstring.parsed})
    return members


def describe_class(symbol, obj, exclude=(), fields_only=False):
    """Combine Griffe's docstrings with authoritative runtime field metadata."""
    cls = load_class(symbol)
    fields = describe_fields(cls, obj, exclude)
    excluded = set(exclude) | {field["name"] for field in fields}
    members = [] if fields_only else describe_members(cls, obj, excluded)
    return {
        "symbol": symbol,
        "description": obj.docstring.value if obj.docstring else "",
        "fields": fields,
        "members": members,
    }


def generate(root=ROOT):
    config = yaml.safe_load((root / "fern/reference.yml").read_text())
    loader = griffe.GriffeLoader(
        search_paths=[str(root / "src"), *sys.path],
        extensions=griffe.load_extensions("griffe_pydantic"),
        docstring_parser=griffe.Parser.google,
    )
    loader.load("cloudai")
    loader.resolve_aliases(external=True)
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(root / "doc/templates"),
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
        autoescape=False,
    )
    env.filters["mdx"] = mdx_text
    template = env.get_template("reference.mdx.jinja")
    output = root / "fern/generated"
    output.mkdir(parents=True, exist_ok=True)
    expected = set()
    for entry in config["references"]:
        symbol = entry["symbol"]
        data = describe_class(
            symbol, loader.modules_collection[symbol], entry.get("exclude", []), entry.get("fields_only", False)
        )
        path = output / f"{symbol}.mdx"
        path.write_text(template.render(**data))
        expected.add(path)
    for stale in set(output.glob("*.mdx")) - expected:
        stale.unlink()
    print(f"Generated {len(expected)} reference snippets in {output}")


if __name__ == "__main__":
    generate()
