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

import argparse
import subprocess
from pathlib import Path

import yaml


class Dumper(yaml.SafeDumper):
    """Indent YAML lists to match the repository's configuration style."""

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def release_number(slug):
    parts = slug.removeprefix("v").split("-", 1)[0].split(".")
    return tuple(map(int, parts)) if all(part.isdecimal() for part in parts) else ()


def configure(source, state, template, slug, sha, production):
    """Prepare a complete version list without copying any version's content."""
    if not slug or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789.-" for c in slug):
        raise ValueError("Version slugs must contain only lowercase letters, numbers, dots, and hyphens.")
    config = yaml.safe_load(template.read_text())
    published = yaml.safe_load(state.read_text()) if state.exists() else {"versions": []}
    versions = published["versions"]
    if any("ref" not in entry or "path" in entry for entry in versions):
        raise ValueError("release/docs must contain only versions pinned to source commit refs.")
    previous = next((entry for entry in versions if entry["slug"] == slug), None)
    if production and previous:
        if slug == "dev":
            subprocess.run(["git", "-C", str(source), "merge-base", "--is-ancestor", previous["ref"], sha], check=True)
        elif previous["ref"] != sha:
            raise ValueError(f"Published release {slug} is immutable; its source commit cannot be replaced.")
    candidate = {"display-name": slug, "slug": slug, "ref": sha, "availability": "beta" if slug == "dev" else "stable"}
    if production:
        versions = [entry for entry in versions if entry["slug"] != slug] + [candidate]
        versions.sort(key=lambda entry: (entry["slug"] != "dev", release_number(entry["slug"])), reverse=True)
    else:
        # Preserve every production entry, including an already published release with this name.
        candidate.update({"display-name": f"Preview: {slug}", "slug": f"preview-{slug}", "availability": "beta"})
        versions = [candidate, *versions]
    config["versions"] = versions
    rendered = yaml.dump(config, Dumper=Dumper, sort_keys=False)
    (source / "fern/docs.yml").write_text(rendered)
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(rendered)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--production", action="store_true")
    args = parser.parse_args()
    configure(args.source, args.state, args.template, args.slug, args.sha, args.production)
