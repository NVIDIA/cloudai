# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, cast

import yaml

from cloudai.core import JsonGenStrategy

from .ai_dynamo import AIDynamoTestDefinition


class AIDynamoKubernetesJsonGenStrategy(JsonGenStrategy):
    """JSON generation strategy for AI Dynamo on Kubernetes systems."""

    def _install_python_packages(self, repo_root: Path, venv_pip: Path) -> None:
        installs = [
            ("perf_analyzer", repo_root),
            ("genai-perf", repo_root / "genai-perf"),
        ]

        for package, path in installs:
            install_cmd = f"cd {path} && {venv_pip} install ."
            logging.info(f"Installing {package} with command: {install_cmd}")
            subprocess.run(install_cmd, shell=True, capture_output=True, text=True, check=True)

    def _setup_dynamo_graph_deployment(self, td: AIDynamoTestDefinition) -> None:
        python_exec = td.python_executable
        if not python_exec.venv_path:
            raise ValueError(
                f"The virtual environment for git repo {python_exec.git_repo} does not exist. "
                "Please ensure to run installation before running the test."
            )

        venv_pip = python_exec.venv_path.absolute() / "bin" / "pip"
        assert python_exec.git_repo.installed_path
        repo_root = python_exec.git_repo.installed_path.absolute()

        self._install_python_packages(repo_root, venv_pip)

    def gen_json(self) -> Dict[Any, Any]:
        td = cast(AIDynamoTestDefinition, self.test_run.test.test_definition)

        if td.cmd_args.dynamo_graph_path is None:
            raise ValueError("dynamo_graph_path must be provided in cmd_args")

        self._setup_dynamo_graph_deployment(td)

        with open(td.cmd_args.dynamo_graph_path, "r") as f:
            yaml_data = yaml.safe_load(f)
            if not isinstance(yaml_data, dict):
                raise ValueError(f"YAML content must be a dictionary/object, got {type(yaml_data)}")
            return yaml_data
