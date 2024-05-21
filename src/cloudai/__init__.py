# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import cloudai.schema.test_template  # noqa
from cloudai.parser.system_parser.slurm_system_parser import SlurmSystemParser
from cloudai.parser.system_parser.standalone_system_parser import StandaloneSystemParser
from cloudai.runner.slurm.slurm_runner import SlurmRunner
from cloudai.runner.standalone.standalone_runner import StandaloneRunner

from ._core.registry import Registry
from .grader import Grader
from .installer import Installer
from .parser import Parser
from .report_generator import ReportGenerator
from .runner import Runner
from .system_object_updater import SystemObjectUpdater

Registry().add_system_parser("standalone", StandaloneSystemParser)
Registry().add_system_parser("slurm", SlurmSystemParser)

Registry().add_runner("slurm", SlurmRunner)
Registry().add_runner("standalone", StandaloneRunner)


__all__ = [
    "Grader",
    "Installer",
    "Parser",
    "ReportGenerator",
    "Runner",
    "SystemObjectUpdater",
]
