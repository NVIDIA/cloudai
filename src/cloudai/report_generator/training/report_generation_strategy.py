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
import logging
from dataclasses import asdict
from typing import ClassVar

from cloudai.core import ReportGenerationStrategy

from .parser import MegatronBridgeParser, MegatronParser, NeMoRunParser, TrainingParser


class TrainingReportGenerationStrategy(ReportGenerationStrategy):
    """Writes training_report.json for training workloads (NeMoRun, MegatronRun, MegatronBridge)."""

    REPORT_FILE_NAME = "training_report.json"

    PARSERS: ClassVar[dict[str, type[TrainingParser]]] = {
        "NeMoRun": NeMoRunParser,
        "MegatronRun": MegatronParser,
        "MegatronBridge": MegatronBridgeParser,
    }

    def can_handle_directory(self) -> bool:
        parser_cls = self.PARSERS.get(self.test_run.test.test_template_name)
        return parser_cls is not None and parser_cls().can_parse(self.test_run)

    def generate_report(self) -> None:
        parser_cls = self.PARSERS[self.test_run.test.test_template_name]
        training_results = parser_cls().parse(self.test_run, self.system)

        report_path = self.test_run.output_path / self.REPORT_FILE_NAME
        report_path.write_text(json.dumps(asdict(training_results), indent=2, default=self._json_default))

        logging.info(f"Generated training report for '{self.test_run.name}' at {report_path}")

    @staticmethod
    def _json_default(value: object) -> object:
        """Convert unusual scalar values into stdlib JSON-compatible values."""
        item = getattr(value, "item", None)
        if callable(item):
            return item()
        return str(value)
