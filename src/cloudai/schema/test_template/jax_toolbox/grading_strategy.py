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

from cloudai.schema.core.strategy import GradingStrategy, StrategyRegistry
from cloudai.schema.system import SlurmSystem

from .template import JaxToolbox


@StrategyRegistry.strategy(GradingStrategy, [SlurmSystem], [JaxToolbox])
class JaxToolboxGradingStrategy(GradingStrategy):
    """Performance grading strategy for JaxToolbox test templates on Slurm systems."""

    def grade(self, directory_path: str, ideal_perf: float) -> float:
        return 0.0
