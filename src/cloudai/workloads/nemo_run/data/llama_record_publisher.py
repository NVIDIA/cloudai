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


from typing import Dict

from cloudai import TestRun
from cloudai.data.publisher.base_record_publisher import BaseRecordPublisher
from cloudai.workloads.nemo_run.data.llama import NeMoRunLLAMARecord


class NeMoRunLLAMARecordPublisher(BaseRecordPublisher):
    """Publisher for NeMoRun LLAMA records to the HTTP data repository."""

    def build_record(self, raw_data: Dict) -> NeMoRunLLAMARecord:
        return NeMoRunLLAMARecord.from_flat_dict(raw_data)

    def publish_from_test_run(self, tr: TestRun) -> None:
        pass  # TODO: Implement this
