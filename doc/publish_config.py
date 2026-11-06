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


def version_order(version):
    """Sort releases numerically, newest first, with dev last (reverse=True)."""
    slug = version["slug"]
    parts = slug.removeprefix("v").split("-", 1)[0].split(".")
    number = tuple(map(int, parts)) if all(part.isdecimal() for part in parts) else ()
    return slug != "dev", number


def production_versions(versions, slug, sha, source):
    """Update one version while preventing release replacement or dev rollback."""
    previous = next((version for version in versions if version["slug"] == slug), None)
    if previous:
        if slug == "dev":
            subprocess.run(["git", "-C", str(source), "merge-base", "--is-ancestor", previous["ref"], sha], check=True)
        elif previous["ref"] != sha:
            raise ValueError(f"Published release {slug} is immutable; its source commit cannot be replaced.")
    candidate = {
        "display-name": slug,
        "slug": slug,
        "ref": sha,
        "availability": "beta" if slug == "dev" else "stable",
    }
    updated = [version for version in versions if version["slug"] != slug]
    updated.append(candidate)
    return sorted(updated, key=version_order, reverse=True)


def preview_versions(versions, slug, sha):
    """Make the preview the default while preserving all production versions."""
    candidate = {
        "display-name": f"Preview: {slug}",
        "slug": f"preview-{slug}",
        "ref": sha,
        "availability": "beta",
    }
    return [candidate, *versions]


def configure(source, state, template, slug, sha, production):
    """Read production state and write the deployment config into the source checkout."""
    if not slug or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789.-" for c in slug):
        raise ValueError("Version slugs must contain only lowercase letters, numbers, dots, and hyphens.")
    config = yaml.safe_load(template.read_text())
    if state.exists():
        versions = yaml.safe_load(state.read_text())["versions"]
    else:
        versions = [version for version in config["versions"] if "ref" in version]
    if any("ref" not in version or "path" in version for version in versions):
        raise ValueError("release/docs must contain only versions pinned to source commit refs.")
    if production:
        config["versions"] = production_versions(versions, slug, sha, source)
    else:
        config["versions"] = preview_versions(versions, slug, sha)
    (source / "fern/docs.yml").write_text(yaml.dump(config, Dumper=Dumper, sort_keys=False))


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
