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


from typing import Any, Dict

from .http_data_repository import HttpDataRepository
from .llama_record import NeMoRunLLAMARecord


class NeMoRunLLAMARecordPublisher:
    """Publisher for NeMoRun LLAMA records to the HTTP data repository."""

    def __init__(self, repository: HttpDataRepository) -> None:
        """Initialize the publisher with a repository."""
        self.repository = repository

    def publish(self, raw_data: Dict[str, Any]) -> None:
        record = NeMoRunLLAMARecord.from_flat_dict(raw_data)
        flat_record = record.to_flat_dict()
        self.repository.store(flat_record)
