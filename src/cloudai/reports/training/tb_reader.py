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

"""Thin tbparse wrapper for the training report parsers (keeps pandas contained here)."""

from pathlib import Path

from tbparse import SummaryReader

from .models import Scalar


def read_scalars(tb_dir: Path) -> list[Scalar]:
    """
    Read merged scalar events.

    SummaryReader merges every event file under the directory (so restarted runs that append a
    new file are combined); duplicates are deduped by (tag, step), keeping the latest wall_time.
    """
    df = SummaryReader(str(tb_dir), extra_columns={"wall_time"}).scalars
    if df.empty:
        return []
    df = df.sort_values("wall_time").drop_duplicates(subset=["tag", "step"], keep="last")
    return [Scalar.from_record(record) for record in df.to_dict("records")]


def read_text(tb_dir: Path) -> dict[str, str]:
    """Read text summaries as a tag -> value mapping (latest value per tag)."""
    df = SummaryReader(str(tb_dir), extra_columns={"wall_time"}).text
    if df.empty:
        return {}
    df = df.sort_values("wall_time").drop_duplicates(subset=["tag"], keep="last")
    return {str(r["tag"]): str(r["value"]) for r in df.to_dict("records")}


if __name__ == "__main__":
    # Debug: python -m cloudai.reports.training.tb_reader /path/to/tensorboard/logs
    import dataclasses
    import json
    import sys

    print(json.dumps([dataclasses.asdict(s) for s in read_scalars(Path(sys.argv[1]))], indent=4))
