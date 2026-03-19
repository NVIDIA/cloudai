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

from pathlib import Path

from cloudai.workloads.nixl_ep.log_parsing import parse_nixl_ep_bandwidth_samples


def test_parse_combined_bandwidth_output(tmp_path: Path) -> None:
    log_path = tmp_path / "nixl-ep-node-0.log"
    log_path.write_text(
        "[rank 3] Dispatch + combine bandwidth: 45.67 GB/s, avg_t=123.4 us, min_t=120.0 us, max_t=130.0 us\n",
        encoding="utf-8",
    )

    samples = parse_nixl_ep_bandwidth_samples(log_path)

    assert len(samples) == 1
    assert samples[0].rank == 3
    assert samples[0].dispatch_combine_bandwidth_gbps == 45.67
    assert samples[0].avg_time_us == 123.4
    assert samples[0].min_time_us == 120.0
    assert samples[0].max_time_us == 130.0


def test_parse_kineto_bandwidth_output(tmp_path: Path) -> None:
    log_path = tmp_path / "nixl-ep-node-0.log"
    log_path.write_text(
        "[rank 7] Dispatch bandwidth: 30.25 GB/s | Combine bandwidth: 28.75 GB/s\n",
        encoding="utf-8",
    )

    samples = parse_nixl_ep_bandwidth_samples(log_path)

    assert len(samples) == 1
    assert samples[0].rank == 7
    assert samples[0].dispatch_bandwidth_gbps == 30.25
    assert samples[0].combine_bandwidth_gbps == 28.75


def test_parse_nixl_ep_bandwidth_samples_ignores_unrelated_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "nixl-ep-node-0.log"
    log_path.write_text("GpuFreq=control_disabled\nrun completed\n", encoding="utf-8")

    assert parse_nixl_ep_bandwidth_samples(log_path) == []
