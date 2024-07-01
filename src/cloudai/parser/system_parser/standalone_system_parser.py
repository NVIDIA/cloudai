#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from typing import Any, Dict

from cloudai import BaseSystemParser
from cloudai.systems import StandaloneSystem


class StandaloneSystemParser(BaseSystemParser):
    """Parser for parsing Standalone system configurations."""

    def parse(self, data: Dict[str, Any]) -> StandaloneSystem:
        """
        Parse the Standalone system configuration.

        Args:
            data (Dict[str, Any]): The loaded configuration data.

        Returns:
            StandaloneSystem: The parsed Standalone system object.

        Raises:
            ValueError: If 'name' or 'output_path' are missing from the data or if there are node list parsing issues
            or group membership conflicts.
        """
        name = data.get("name")
        if not name:
            raise ValueError("Missing mandatory field: 'name'")

        output_path = data.get("output_path")
        if not output_path:
            raise ValueError("Field 'output_path' is required.")
        output_path = os.path.abspath(output_path)

        return StandaloneSystem(name=name, output_path=output_path)
