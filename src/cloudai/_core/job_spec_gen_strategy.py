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

from abc import abstractmethod

from .job_context import JobContext
from .job_specification import JobSpecification
from .test_template_strategy import TestTemplateStrategy


class JobSpecGenStrategy(TestTemplateStrategy):
    """
    Abstract base class defining the interface for job specification generation strategies.

    It specifies how to generate job specifications based on system and test parameters.
    """

    @abstractmethod
    def gen_job_spec(self, context: JobContext) -> JobSpecification:
        """
        Generate the job specification for a test based on the given context.

        Args:
            context (JobContext): The context containing all necessary parameters.

        Returns:
            JobSpecification: The generated job specification, which could be a command string,
                              a JSON object, or other format suitable for the system environment.
        """
        pass
