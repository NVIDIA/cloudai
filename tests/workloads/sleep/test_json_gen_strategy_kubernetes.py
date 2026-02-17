# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from cloudai.core import TestRun
from cloudai.systems.kubernetes import KubernetesSystem
from cloudai.workloads.sleep import SleepCmdArgs, SleepKubernetesJsonGenStrategy, SleepTestDefinition


def test_job_name_sanitization(k8s_system: KubernetesSystem) -> None:
    tdef = SleepTestDefinition(name="name", description="desc", test_template_name="tt", cmd_args=SleepCmdArgs())
    tr = TestRun(name="t!e@st#-n$am%e^", test=tdef, nodes=["node1"], num_nodes=1)
    json_gen = SleepKubernetesJsonGenStrategy(k8s_system, tr)

    assert json_gen.gen_json()["metadata"]["name"] == json_gen.sanitize_k8s_job_name(tr.name)
