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

from .chakra_replay import (
    ChakraReplay,
    ChakraReplayGradingStrategy,
    ChakraReplayReportGenerationStrategy,
    ChakraReplaySlurmCommandGenStrategy,
    ChakraReplaySlurmInstallStrategy,
)
from .common import SlurmJobIdRetrievalStrategy, StandaloneJobIdRetrievalStrategy
from .jax_toolbox import (
    JaxToolbox,
    JaxToolboxGradingStrategy,
    JaxToolboxReportGenerationStrategy,
    JaxToolboxSlurmCommandGenStrategy,
    JaxToolboxSlurmInstallStrategy,
)
from .nccl_miner import (
    NcclMiner,
    NcclMinerGradingStrategy,
    NcclMinerReportGenerationStrategy,
    NcclMinerSlurmCommandGenStrategy,
    NcclMinerSlurmInstallStrategy,
)
from .nccl_test import (
    NcclTest,
    NcclTestGradingStrategy,
    NcclTestReportGenerationStrategy,
    NcclTestSlurmCommandGenStrategy,
    NcclTestSlurmInstallStrategy,
)
from .nemo_launcher import (
    NeMoLauncher,
    NeMoLauncherGradingStrategy,
    NeMoLauncherReportGenerationStrategy,
    NeMoLauncherSlurmCommandGenStrategy,
    NeMoLauncherSlurmInstallStrategy,
    NeMoLauncherSlurmJobIdRetrievalStrategy,
)
from .sleep import (
    Sleep,
    SleepGradingStrategy,
    SleepReportGenerationStrategy,
    SleepStandaloneCommandGenStrategy,
    SleepStandaloneInstallStrategy,
)
from .ucc_test import (
    UCCTest,
    UCCTestGradingStrategy,
    UCCTestReportGenerationStrategy,
    UCCTestSlurmCommandGenStrategy,
    UCCTestSlurmInstallStrategy,
)

__all__ = [
    "ChakraReplay",
    "ChakraReplayGradingStrategy",
    "ChakraReplayReportGenerationStrategy",
    "ChakraReplaySlurmCommandGenStrategy",
    "ChakraReplaySlurmInstallStrategy",
    "SlurmJobIdRetrievalStrategy",
    "StandaloneJobIdRetrievalStrategy",
    "JaxToolbox",
    "JaxToolboxGradingStrategy",
    "JaxToolboxReportGenerationStrategy",
    "JaxToolboxSlurmCommandGenStrategy",
    "JaxToolboxSlurmInstallStrategy",
    "NcclMiner",
    "NcclMinerGradingStrategy",
    "NcclMinerReportGenerationStrategy",
    "NcclMinerSlurmCommandGenStrategy",
    "NcclMinerSlurmInstallStrategy",
    "NcclTest",
    "NcclTestGradingStrategy",
    "NcclTestReportGenerationStrategy",
    "NcclTestSlurmCommandGenStrategy",
    "NcclTestSlurmInstallStrategy",
    "NeMoLauncher",
    "NeMoLauncherGradingStrategy",
    "NeMoLauncherReportGenerationStrategy",
    "NeMoLauncherSlurmCommandGenStrategy",
    "NeMoLauncherSlurmInstallStrategy",
    "NeMoLauncherSlurmJobIdRetrievalStrategy",
    "Sleep",
    "SleepGradingStrategy",
    "SleepReportGenerationStrategy",
    "SleepStandaloneCommandGenStrategy",
    "SleepStandaloneInstallStrategy",
    "UCCTest",
    "UCCTestGradingStrategy",
    "UCCTestReportGenerationStrategy",
    "UCCTestSlurmCommandGenStrategy",
    "UCCTestSlurmInstallStrategy",
]
