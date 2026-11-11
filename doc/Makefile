# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

.PHONY: help reference build preview

help:
	@echo "reference - Generate Python reference snippets"
	@echo "build     - Generate references and install Fern"
	@echo "preview   - Generate references and start the local Fern server"

reference:
	cd .. && uv run --locked --extra docs python doc/generate_reference.py

build: reference
	cd .. && npm --prefix fern ci

preview: reference
	cd .. && npm --prefix fern ci
	cd .. && npm --prefix fern run dev
