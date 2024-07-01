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


class JobStatusResult:
    """
    Encapsulates the result of a job status retrieval.

    Attributes
        is_successful (bool): Indicates if the job was successful.
        error_message (str): Error message if the job was not successful.
    """

    def __init__(self, is_successful: bool, error_message: str = ""):
        self.is_successful = is_successful
        self.error_message = error_message

    def __str__(self):
        return f"JobStatusResult(is_successful={self.is_successful}, error_message={self.error_message})"
