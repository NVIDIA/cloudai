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

import glob
import os
from typing import Iterator, List, Optional, Union, cast

from .base_job import BaseJob
from .base_job_with_output import BaseJobWithOutput
from .system import OutputType, System


class FileOutputSystem(System):
    """Intermediate class for systems where job output (stdout, stderr) is stored in a directory as files."""

    def _get_single_output_file_path(self, job: BaseJobWithOutput, output_type: OutputType) -> Optional[str]:
        """
        Retrieve the single output file path for the specified output type (stdout.txt, stderr.txt).

        Args:
            job (BaseJobWithOutput): The job for which to retrieve output.
            output_type (OutputType): The type of output to retrieve (e.g., stdout, stderr).

        Returns:
            Optional[str]: The file path for the specified output type, or None if not found.
        """
        file_name = f"{output_type.value}.txt"
        file_path = os.path.join(job.output_path, file_name)
        return file_path if os.path.isfile(file_path) else None

    def _get_multiple_output_file_paths(self, job: BaseJobWithOutput, output_type: OutputType) -> List[str]:
        """
        Retrieve the file paths for the specified output type (stdout-[rank_id].txt, stderr-[rank_id].txt).

        Args:
            job (BaseJobWithOutput): The job for which to retrieve output.
            output_type (OutputType): The type of output to retrieve (e.g., stdout, stderr).

        Returns:
            List[str]: A list of file paths associated with the specified output type.
        """
        output_pattern = os.path.join(job.output_path, f"{output_type}-*.txt")
        return glob.glob(output_pattern)

    def retrieve_output_streams(
        self, job: BaseJob, output_type: OutputType, multiple_files: bool = False, line_by_line: bool = False
    ) -> Union[Optional[str], Optional[Iterator[str]], Optional[List[Union[str, Iterator[str]]]]]:
        """
        Retrieve output streams (stdout or stderr) for a given job.

        Args:
            job (BaseJob): The job for which to retrieve output.
            output_type (OutputType): The type of output to retrieve.
            multiple_files (bool): Whether to return output from multiple files separately.
            line_by_line (bool): Whether to return the output line by line as an iterator.

        Returns:
            Union[Optional[str], Optional[Iterator[str]], Optional[List[Union[str, Iterator[str]]]]]:
            The output as a string, an iterator for line-by-line reading, or a list of such outputs.
        """
        o_job = cast(BaseJobWithOutput, job)

        if multiple_files:
            file_paths = self._get_multiple_output_file_paths(o_job, output_type)
            outputs = []
            for path in file_paths:
                with open(path, "r") as file:
                    outputs.append(iter(file) if line_by_line else file.read())
            return outputs if outputs else None
        else:
            file_path = self._get_single_output_file_path(o_job, output_type)
            if file_path:
                with open(file_path, "r") as file:
                    return iter(file) if line_by_line else file.read()

        return None
