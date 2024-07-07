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

from cloudai import System


class StandaloneSystem(System):
    """
    Class representing a Standalone system.

    This class is used for systems that execute commands directly without
    a job scheduler.

    Attributes
        name (str): Name of the standalone system.
        output_path (str): Path to the output directory.
    """

    def __init__(self, name: str, output_path: str) -> None:
        """
        Initialize a StandaloneSystem instance.

        Args:
            name (str): Name of the standalone system.
            output_path (str): Path to the output directory.
        """
        super().__init__(name, "standalone", output_path)

    def __repr__(self) -> str:
        """
        Provide a string representation of the StandaloneSystem instance.

        Returns
            str: String representation of the standalone system.
        """
        return f"StandaloneSystem(name={self.name}, " f"scheduler={self.scheduler})"

    def update(self) -> None:
        pass
