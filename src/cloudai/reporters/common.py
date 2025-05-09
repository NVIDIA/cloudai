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

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import toml

from cloudai import TestRun
from cloudai.systems.slurm.metadata import SlurmSystemMetadata


def case_name(tr: TestRun) -> str:
    name = tr.name
    if tr.current_iteration > 0:
        name = f"{name} iter={tr.current_iteration}"
    if tr.step > 0:
        name = f"{name} step={tr.step}"
    return name


@dataclass
class ReportItem:
    """Report item for a test run."""

    name: str
    description: str
    logs_path: Optional[str] = None

    @classmethod
    def from_test_runs(cls, test_runs: list[TestRun], results_root: Path) -> list["ReportItem"]:
        report_items: list[ReportItem] = []
        for tr in test_runs:
            report_items.append(ReportItem(case_name(tr), tr.test.description))
            if tr.output_path.exists():
                report_items[-1].logs_path = f"./{tr.output_path.relative_to(results_root)}"
        return report_items


@dataclass
class SlurmReportItem:
    """Report item for Slurm systems."""

    name: str
    description: str
    logs_path: Optional[str] = None
    nodes: Optional[str] = None

    @classmethod
    def get_metadata(cls, run_dir: Path) -> Optional[SlurmSystemMetadata]:
        if not (run_dir / "metadata").exists():
            logging.debug(f"No metadata folder found in {run_dir}")
            return None

        node_files = list(run_dir.glob("metadata/node-*.toml"))
        if not node_files:
            logging.debug(f"No node files found in {run_dir}/metadata")
            return None

        node_file = node_files[0]
        with node_file.open() as f:
            try:
                return SlurmSystemMetadata.model_validate(toml.load(f))
            except Exception as e:
                logging.debug(f"Error validating metadata for {node_file}: {e}")

        return None

    @classmethod
    def from_test_runs(cls, test_runs: list[TestRun], results_root: Path) -> list["SlurmReportItem"]:
        report_items: list[SlurmReportItem] = []
        for tr in test_runs:
            ri = SlurmReportItem(case_name(tr), tr.test.description)
            if tr.output_path.exists():
                ri.logs_path = f"./{tr.output_path.relative_to(results_root)}"
            if metadata := cls.get_metadata(tr.output_path):
                ri.nodes = metadata.slurm.node_list
            report_items.append(ri)

        return report_items
