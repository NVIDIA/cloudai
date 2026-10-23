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

from cloudai.core import Reporter

from .parser import MegatronBridgeParser, MegatronParser, NeMoRunParser, TrainingParser


class TrainingReporter(Reporter):
    """Generates a training report for each supported test run in a scenario."""

    REPORT_FILE_NAME = "training_report.json"

    PARSERS: ClassVar[dict[str, type[TrainingParser]]] = {
        "NeMoRun": NeMoRunParser,
        "MegatronRun": MegatronParser,
        "MegatronBridge": MegatronBridgeParser,
    }

    def generate(self) -> None:
        self.load_test_runs()

        for tr in self.trs:
            parser_cls = self.PARSERS.get(tr.test_definition.test_template_name)
            if parser_cls is None:
                continue
            parser = parser_cls()
            if not parser.can_parse(tr):
                continue
            try:
                training_results = parser.parse(tr, self.system, self.test_scenario)

                report_path = tr.output_path / self.REPORT_FILE_NAME
                report_path.write_text(json.dumps(asdict(training_results), indent=2, default=self._json_default))

                logging.info(f"Generated training report for '{tr.name}' at {report_path}")
            except Exception as exc:
                logging.warning(f"Error generating training report for '{tr.output_path}': {exc}")

    @staticmethod
    def _json_default(value: object) -> object:
        """Convert unusual scalar values into stdlib JSON-compatible values."""
        item = getattr(value, "item", None)
        if callable(item):
            return item()
        return str(value)
