# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, ConfigDict


class ReportConfig(BaseModel):
    """Configuration shared by system and scenario reports."""

    model_config = ConfigDict(extra="forbid")
    enable: bool = False
