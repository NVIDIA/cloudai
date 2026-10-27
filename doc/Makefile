# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Set the provisioned instance URL in the publishing environment before deploying.
export FERN_DOCS_URL ?= cloudai-local-test.docs.buildwithfern.com

.PHONY: help reference check preview

help:
	@echo "reference - Generate Python reference snippets"
	@echo "check     - Generate references and validate Fern documentation"
	@echo "preview   - Generate references and start the local Fern server"

reference:
	cd .. && uv run --locked --extra docs python doc/generate_reference.py

check: reference
	cd .. && npm --prefix fern ci
	cd .. && npm --prefix fern run check

preview: reference
	cd .. && npm --prefix fern ci
	cd .. && npm --prefix fern run dev
