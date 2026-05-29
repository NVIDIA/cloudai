# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

from cloudai.workloads.ai_dynamo.runtime import aiperf


def _write_fake_aiperf(tmp_path: Path) -> Path:
    script = tmp_path / "fake_aiperf.py"
    script.write_text(
        """
import sys
from pathlib import Path

artifact_dir = Path(sys.argv[sys.argv.index("--artifact-dir") + 1])
url = sys.argv[sys.argv.index("--url") + 1]
artifact_dir.mkdir(parents=True, exist_ok=True)
(artifact_dir / "profile_export_aiperf.csv").write_text(f"url\\n{url}\\n", encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )
    return script


def test_runtime_executes_entries_and_copies_final_report(tmp_path: Path) -> None:
    fake_aiperf = _write_fake_aiperf(tmp_path)
    commands_file = tmp_path / "aiperf_commands.json"
    artifact_dir = tmp_path / "aiperf_artifacts" / "round_1"
    report_file = tmp_path / "aiperf_round_1_report.csv"
    final_report_file = tmp_path / "aiperf_report.csv"
    commands_file.write_text(
        json.dumps(
            [
                {
                    "name": "round_1",
                    "cmd": [sys.executable, str(fake_aiperf)],
                    "cli": [
                        "--url",
                        "{frontend_url}:8000",
                        "--artifact-dir",
                        str(artifact_dir),
                    ],
                    "output_folder": str(artifact_dir),
                    "log_file": str(tmp_path / "aiperf_round_1.log"),
                    "report_source": str(artifact_dir / "profile_export_aiperf.csv"),
                    "report_file": str(report_file),
                    "final_report_file": str(final_report_file),
                }
            ]
        ),
        encoding="utf-8",
    )

    result = aiperf.main(["--url", "http://frontend", "--commands-file", str(commands_file)])

    assert result == 0
    assert report_file.read_text(encoding="utf-8") == "url\nhttp://frontend:8000\n"
    assert final_report_file.read_text(encoding="utf-8") == "url\nhttp://frontend:8000\n"
    assert (tmp_path / "aiperf_round_1.log").is_file()
