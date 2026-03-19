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

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_FLOAT_RE = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_COMBINED_BW_RE = re.compile(
    rf"\[rank (?P<rank>\d+)\] Dispatch \+ combine bandwidth: "
    rf"(?P<bandwidth>{_FLOAT_RE}) GB/s, "
    rf"avg_t=(?P<avg_time>{_FLOAT_RE}) us, "
    rf"min_t=(?P<min_time>{_FLOAT_RE}) us, "
    rf"max_t=(?P<max_time>{_FLOAT_RE}) us"
)
_KINETO_BW_RE = re.compile(
    rf"\[rank (?P<rank>\d+)\] Dispatch bandwidth: "
    rf"(?P<dispatch_bandwidth>{_FLOAT_RE}) GB/s \| "
    rf"Combine bandwidth: (?P<combine_bandwidth>{_FLOAT_RE}) GB/s"
)


@dataclass(frozen=True)
class NixlEPBandwidthSample:
    rank: int
    dispatch_combine_bandwidth_gbps: float | None = None
    avg_time_us: float | None = None
    min_time_us: float | None = None
    max_time_us: float | None = None
    dispatch_bandwidth_gbps: float | None = None
    combine_bandwidth_gbps: float | None = None


def parse_nixl_ep_bandwidth_samples(path: Path) -> list[NixlEPBandwidthSample]:
    if not path.is_file():
        return []

    samples: list[NixlEPBandwidthSample] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if match := _COMBINED_BW_RE.search(line):
            samples.append(
                NixlEPBandwidthSample(
                    rank=int(match.group("rank")),
                    dispatch_combine_bandwidth_gbps=float(match.group("bandwidth")),
                    avg_time_us=float(match.group("avg_time")),
                    min_time_us=float(match.group("min_time")),
                    max_time_us=float(match.group("max_time")),
                )
            )
            continue

        if match := _KINETO_BW_RE.search(line):
            samples.append(
                NixlEPBandwidthSample(
                    rank=int(match.group("rank")),
                    dispatch_bandwidth_gbps=float(match.group("dispatch_bandwidth")),
                    combine_bandwidth_gbps=float(match.group("combine_bandwidth")),
                )
            )

    return samples
