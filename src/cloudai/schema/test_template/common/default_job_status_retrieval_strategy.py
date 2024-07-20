# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from cloudai import JobStatusResult, JobStatusRetrievalStrategy


class DefaultJobStatusRetrievalStrategy(JobStatusRetrievalStrategy):
    """
    Dummy strategy for retrieving job statuses.

    This strategy is used in scenarios where job status retrieval logic is not required or is yet to be implemented.
    It always returns a success result, indicating that the job has successfully completed.
    """

    def get_job_status(self, output_path: str) -> JobStatusResult:
        return JobStatusResult(is_successful=True)
