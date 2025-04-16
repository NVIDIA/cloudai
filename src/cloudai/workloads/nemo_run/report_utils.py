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
from functools import cache
from pathlib import Path
from typing import List


@cache
def parse_step_timings(filepath: Path) -> List[float]:
    if not filepath.exists():
        logging.debug(f"{filepath} not found")
        return []
    step_timings: List[float] = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "train_step_timing in s:" in line:
                try:
                    timing = float(line.split("train_step_timing in s:")[1].strip().split()[0])
                    step_timings.append(timing)
                except (ValueError, IndexError):
                    continue
    if not step_timings:
        logging.debug(f"No train_step_timing found in {filepath}")
        return []
    return _filter_step_timings(step_timings)


def _filter_step_timings(step_timings: List[float]) -> List[float]:
    return step_timings[-20:] if len(step_timings) > 100 else step_timings
