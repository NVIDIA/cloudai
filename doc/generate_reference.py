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

import html
import importlib
import inspect
import shutil
import subprocess
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
    if not issubclass(cls, pydantic.BaseModel):
        raise TypeError(f"{cls.__name__} is not a Pydantic model")
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
                "default": repr(field.default).replace(f"{ROOT}/", "")
                if not field.is_required() and factory is None
                else None,
                "factory": getattr(factory, "__qualname__", str(factory)) if factory else None,
                "constraints": [repr(item) for item in field.metadata],
                "alias": field.alias,
            }
        )
    return fields


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
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=jinja2.select_autoescape(),
    )
    env.filters["mdx"] = mdx_text
    env.filters["cell"] = lambda value: mdx_text(value).replace("|", "&#124;").replace("\n", "<br />")
    env.filters["code"] = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    template = env.get_template("reference.mdx.jinja2")
    output = root / "fern/generated"
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True, exist_ok=True)
    for entry in config["references"]:
        symbol = entry["symbol"]
        obj = loader.modules_collection[symbol]
        data = {
            "symbol": symbol,
            "description": obj.docstring.value if obj.docstring else "",
            "fields": describe_fields(load_class(symbol), obj, entry.get("exclude", [])),
        }
        filename = symbol.lower().replace("_", "-")
        path = output / f"{filename}.mdx"
        path.write_text(template.render(**data).rstrip() + "\n")
    print(f"Generated {len(config['references'])} reference snippets in {output}")


if __name__ == "__main__":
    generate()
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "fern/generated/"], cwd=ROOT, text=True
    )
    if untracked:
        sys.exit("Stage newly generated references with: git add fern/generated")
