# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from cloudai.models.report import ReportConfig
from cloudai.models.scenario import ReportConfig as ScenarioReportConfig


def test_report_config_import_remains_compatible() -> None:
    assert ScenarioReportConfig is ReportConfig


def test_report_config_rejects_unknown_settings() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ReportConfig.model_validate({"enable": True, "unknown": True})
