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
from typing import Any, Dict

from .http_data_repository import HttpDataRepository
from .llama_record import NeMoRunLLAMARecord


class NeMoRunLLAMARecordPublisher:
    """Publisher for NeMoRun LLAMA records to the HTTP data repository."""

    def __init__(self, repository: HttpDataRepository) -> None:
        """Initialize the publisher with a repository."""
        self.repository = repository

    def build_record(self, raw_data: Dict) -> NeMoRunLLAMARecord:
        return NeMoRunLLAMARecord.from_flat_dict(raw_data)

    def publish(self, raw_data: Dict[str, Any]) -> None:
        """
        Build the record from raw_data and publish it.

        Args:
            raw_data (Dict[str, Any]): The raw data used to build the record.
        """
        record = self.build_record(raw_data)
        self.publish_record(record)

    def publish_record(self, record: NeMoRunLLAMARecord) -> None:
        """
        Flatten the record and store it in the repository.

        Args:
            record (NeMoRunLLAMARecord): The record to be published. Must have a to_flat_dict() method.
        """
        flat_record = record.to_flat_dict()
        self.repository.store(flat_record)
        logging.info("Published record successfully.")
