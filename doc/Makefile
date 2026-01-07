# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

SPHINXOPTS    ?=
SOURCEDIR     = .
BUILDDIR      = _build
BUILDDIR_CMS  = _build-cms

NODE_MODULES  = $(BUILDDIR_CMS)/node_modules
MMDC          = $(NODE_MODULES)/.bin/mmdc
PUPPETEER_CFG = $(BUILDDIR_CMS)/puppeteer-config.json

.PHONY: help html cms-html install-mermaid clean

help:
	@echo "Available targets:"
	@echo "  html       - Build documentation for GitHub Pages (nvidia_sphinx_theme) -> $(BUILDDIR)/"
	@echo "  cms-html   - Build documentation for BrightSpot CMS (sphinx_rtd_theme, PNG mermaid) -> $(BUILDDIR_CMS)/"
	@echo "  clean      - Remove all build directories"

# Default html target (GitHub Pages)
html:
	@echo "Setting up environment for GitHub Pages build..."
	@cd .. && uv sync --extra docs
	@echo "Building documentation..."
	@cd .. && uv run sphinx-build -M html "doc" "doc/$(BUILDDIR)" $(SPHINXOPTS)
	@touch $(BUILDDIR)/html/.nojekyll

cms-html: install-mermaid
	@echo "Setting up environment for CMS build..."
	@cd .. && uv sync --extra docs-cms
	@echo "Building documentation..."
	@cd .. && CLOUDAI_DOC_TARGET=cms uv run sphinx-build -M html "doc" "doc/$(BUILDDIR_CMS)" $(SPHINXOPTS)

install-mermaid: $(MMDC)

$(MMDC):
	@command -v npm >/dev/null 2>&1 || { echo "Error: npm is required for CMS build but not found. Please install Node.js/npm first."; exit 1; }
	@echo "Installing @mermaid-js/mermaid-cli via npm..."
	@mkdir -p $(BUILDDIR_CMS)
	@cd $(BUILDDIR_CMS) && npm install @mermaid-js/mermaid-cli@^11
	@echo '{"args": ["--no-sandbox"]}' > $(PUPPETEER_CFG)
	@echo "mermaid-cli installed successfully."

clean:
	@rm -rf $(BUILDDIR) $(BUILDDIR_CMS)
