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

"""Process of one real probe trial: sleep, then report ``x`` and whether the trial succeeded."""

import json
import sys
import time
from pathlib import Path


def main() -> int:
    output_path, x, duration = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
    fail, no_metric = "--fail" in sys.argv[4:], "--no-metric" in sys.argv[4:]
    start = time.time()
    time.sleep(duration)
    (output_path / "probe_timing.json").write_text(json.dumps({"start": start, "finish": time.time()}))
    if not no_metric:
        (output_path / "probe_result.txt").write_text(x)
    (output_path / "probe_status.txt").write_text("failed" if fail else "ok")
    return 3 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
