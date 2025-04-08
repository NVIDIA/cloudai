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
from abc import ABC, abstractmethod
from typing import Any, Dict

from cloudai import TestRun
from cloudai.data.model.base_record import BaseRecord

from ..repository.base import BaseDataRepository


class BaseRecordPublisher(ABC):
    """Abstract base class for record publishers."""

    def __init__(self, repository: BaseDataRepository) -> None:
        """Initialize the publisher with a repository."""
        self.repository = repository

    @abstractmethod
    def build_record(self, raw_data: Dict[str, Any]) -> BaseRecord:
        """
        Build a record from raw data.

        Args:
            raw_data (Dict[str, Any]): The raw data used to build the record.

        Returns:
            BaseRecord: The record built from raw_data.
        """
        pass

    def publish(self, raw_data: Dict[str, Any]) -> None:
        """
        Build the record from raw_data and publish it.

        Args:
            raw_data (Dict[str, Any]): The raw data used to build the record.
        """
        record = self.build_record(raw_data)
        self.publish_record(record)

    def publish_record(self, record: BaseRecord) -> None:
        """
        Flatten the record and store it in the repository.

        Args:
            record (BaseRecord): The record to be published. Must have a to_flat_dict() method.
        """
        flat_record = record.to_flat_dict()
        self.repository.store(flat_record)
        logging.info("Published record successfully.")

    @abstractmethod
    def publish_from_test_run(self, tr: TestRun) -> None:
        """
        Build and publish a record from a test run object.

        Attempts to obtain the raw record data by first checking if the 'get_raw_record()' method exists,
        and if not, falls back to the 'raw_record' attribute.

        Args:
            tr (TestRun): The test run object containing raw record data.
        """
        pass
