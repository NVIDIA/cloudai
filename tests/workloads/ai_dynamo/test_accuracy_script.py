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

import json
import subprocess
import sys
from pathlib import Path

ACCURACY_SCRIPT = Path("src/cloudai/workloads/ai_dynamo/accuracy.sh")


def test_accuracy_script_runs_custom_accuracy_command(tmp_path: Path) -> None:
    custom_script = tmp_path / "custom_accuracy.py"
    custom_script.write_text(
        """
import argparse
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--url", required=True)
parser.add_argument("--endpoint", required=True)
parser.add_argument("--result-dir", required=True)
parser.add_argument("--artifact-dir", required=True)
parser.add_argument("--prompt", required=True)
args = parser.parse_args()

artifact_dir = Path(args.artifact_dir)
artifact_dir.mkdir(parents=True, exist_ok=True)
(artifact_dir / "args.json").write_text(json.dumps(vars(args)), encoding="utf-8")
with (artifact_dir / "accuracy_results.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["task", "correct", "total", "accuracy"])
    writer.writerow(["OVERALL", 1, 1, "100.00%"])
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            str(ACCURACY_SCRIPT),
            "--result-dir",
            str(tmp_path),
            "--model",
            "Qwen/Qwen3-0.6B",
            "--url",
            "http://frontend",
            "--port",
            "8000",
            "--endpoint",
            "v1/chat/completions",
            "--entrypoint",
            f"{sys.executable} {custom_script}",
            "--cli",
            (
                "--model {model} --url {url} --endpoint {endpoint} "
                "--result-dir {result_dir} --artifact-dir {artifact_dir} --prompt ping"
            ),
            "--artifact-dir-name",
            "custom_accuracy_artifacts",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert (tmp_path / "accuracy_results.csv").read_text(encoding="utf-8").splitlines()[-1] == "OVERALL,1,1,100.00%"
    args = json.loads((tmp_path / "custom_accuracy_artifacts" / "args.json").read_text(encoding="utf-8"))
    assert args == {
        "model": "Qwen/Qwen3-0.6B",
        "url": "http://frontend:8000",
        "endpoint": "v1/chat/completions",
        "result_dir": str(tmp_path),
        "artifact_dir": str(tmp_path / "custom_accuracy_artifacts"),
        "prompt": "ping",
    }


def test_accuracy_script_fails_custom_accuracy_without_accuracy_csv(tmp_path: Path) -> None:
    custom_script = tmp_path / "custom_accuracy.py"
    custom_script.write_text("from pathlib import Path\nPath(__file__).exists()\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(ACCURACY_SCRIPT),
            "--result-dir",
            str(tmp_path),
            "--model",
            "Qwen/Qwen3-0.6B",
            "--url",
            "http://frontend",
            "--port",
            "8000",
            "--entrypoint",
            f"{sys.executable} {custom_script}",
            "--cli",
            "--artifact-dir {artifact_dir}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "accuracy benchmark was requested" in result.stdout
