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

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re
import sys
from pathlib import Path

# Add the project source to Python path for autodoc
sys.path.insert(0, os.path.abspath("../src"))


# Custom autodoc processing to clean up Pydantic classes
def autodoc_skip_member(app, what, name, obj, skip, options):
    """Skip unwanted Pydantic and other internal members."""
    exclude_patterns = {re.compile(r"model_.*")}

    if any(pattern.match(name) for pattern in exclude_patterns):
        return True

    # Skip private methods starting with underscore (except __init__)
    if name.startswith("_") and name != "__init__":
        return True

    return skip


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "CloudAI"
copyright = "2026, NVIDIA CORPORATION & AFFILIATES"
author = "NVIDIA CORPORATION & AFFILIATES"
version = "1.6.0-beta"
release = "1.6.0-beta"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
]

exclude_patterns = ["_build"]

# -- Autodoc configuration ---------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": False,  # Don't show undocumented members
}

# Generate autosummary even if no references
autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

is_cms_build = os.environ.get("CLOUDAI_DOC_TARGET") == "cms"
html_theme, mermaid_output_format, mermaid_cmd, mermaid_params = "nvidia_sphinx_theme", "raw", "", []
if is_cms_build:
    html_theme = "sphinx_rtd_theme"
    mermaid_output_format = "png"
    mermaid_cmd = str(Path(__file__).parent / "_build-cms/node_modules/.bin/mmdc")
    mermaid_params = ["-p", str(Path(__file__).parent / "_build-cms/puppeteer-config.json")]

# Set the root document to index
root_doc = "index"

source_suffix = [".rst"]
