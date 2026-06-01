# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute generated AIPerf runtime entries."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def log(message: str) -> None:
    print(message, flush=True)


def substitute_frontend_url(values: list[str], frontend_url: str) -> list[str]:
    return [value.replace("{frontend_url}", frontend_url) for value in values]


def copy_file(source: str, destination: str, message: str) -> None:
    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() == destination_path.resolve():
        log(f"{message} {destination_path}")
        return

    shutil.copy2(source_path, destination_path)
    log(f"{message} {destination_path}")


def run_entry(entry: dict[str, Any], frontend_url: str) -> None:
    argv = substitute_frontend_url([*entry["cmd"], *entry.get("cli", [])], frontend_url)
    output_folder = entry.get("output_folder")
    if output_folder:
        shutil.rmtree(output_folder, ignore_errors=True)

    log(f"Running {entry['name']}: {shlex.join(argv)}")
    log_file = entry.get("log_file")
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as fp:
            subprocess.run(argv, stdout=fp, stderr=subprocess.STDOUT, check=True)
    else:
        subprocess.run(argv, check=True)

    report_source = entry.get("report_source")
    report_file = entry.get("report_file")
    if report_source and report_file:
        copy_file(report_source, report_file, "AIPerf report saved to")

    final_report_file = entry.get("final_report_file")
    if final_report_file and report_file:
        copy_file(report_file, final_report_file, "Final AIPerf report saved to")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commands-file", required=True)
    parser.add_argument("--url", required=True)
    args, _ = parser.parse_known_args(argv)
    return args


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        with Path(args.commands_file).open(encoding="utf-8") as fp:
            entries = json.load(fp)

        for entry in entries:
            run_entry(entry, args.url)
    except Exception as exc:
        log(f"ERROR: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
