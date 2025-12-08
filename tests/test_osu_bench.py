# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from cloudai.workloads.osu_bench.osu_bench import OSUBenchCmdArgs


def test_osu_bench_getting_args():
    data = OSUBenchCmdArgs(
        docker_image_url="nvcr.io#nvidia/pytorch:24.02-py3",
        location="/osu/loc",
        benchmark="osu_coll",
        message_size="1024",
        iterations=10,
    )

    args = data.get_args()

    cmd = ' '.join(f'{name} {value}' for name, value in args.items())
    assert cmd == '-m 1024 -n 10 -r cpu'
