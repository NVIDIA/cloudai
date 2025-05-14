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

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import bokeh
    import bokeh.layouts as bokeh_layouts
    import bokeh.models as bokeh_models
    import bokeh.palettes as bokeh_pallettes
    import bokeh.plotting as bokeh_plotting
    import bokeh.transform as bokeh_transform
    import kubernetes as k8s
    import numpy as np
    import pandas as pd


class LazyImports:
    """Class to handle lazy imports of heavy dependencies."""

    def __init__(self):
        self._np: ModuleType | None = None
        self._pd: ModuleType | None = None
        self._k8s: ModuleType | None = None
        self._bokeh: ModuleType | None = None
        self._bokeh_plotting: ModuleType | None = None
        self._bokeh_models: ModuleType | None = None
        self._bokeh_layouts: ModuleType | None = None
        self._bokeh_transform: ModuleType | None = None
        self._bokeh_pallettes: ModuleType | None = None

    @property
    def np(self) -> "np":  # type: ignore[no-any-return]
        """Lazy import of numpy."""
        if self._np is None:
            import numpy as np

            self._np = np
        return cast("np", self._np)

    @property
    def pd(self) -> "pd":  # type: ignore[no-any-return]
        """Lazy import of pandas."""
        if self._pd is None:
            import pandas as pd

            self._pd = pd
        return cast("pd", self._pd)

    @property
    def k8s(self) -> "k8s":  # type: ignore[no-any-return]
        """Lazy import of kubernetes."""
        if self._k8s is None:
            import kubernetes as k8s

            self._k8s = k8s

        return cast("k8s", self._k8s)

    @property
    def bokeh(self) -> "bokeh":  # type: ignore[no-any-return]
        """Lazy import of bokeh."""
        if self._bokeh is None:
            import bokeh as bokeh

            self._bokeh = bokeh

        return cast("bokeh", self._bokeh)

    @property
    def bokeh_plotting(self) -> "bokeh_plotting":  # type: ignore[no-any-return]
        """Lazy import of bokeh.plotting."""
        if self._bokeh_plotting is None:
            import bokeh.plotting as bokeh_plotting

            self._bokeh_plotting = bokeh_plotting

        return cast("bokeh_plotting", self._bokeh_plotting)

    @property
    def bokeh_models(self) -> "bokeh_models":  # type: ignore[no-any-return]
        """Lazy import of bokeh.models."""
        if self._bokeh_models is None:
            import bokeh.models as bokeh_models

            self._bokeh_models = bokeh_models

        return cast("bokeh_models", self._bokeh_models)

    @property
    def bokeh_layouts(self) -> "bokeh_layouts":  # type: ignore[no-any-return]
        """Lazy import of bokeh.layouts."""
        if self._bokeh_layouts is None:
            import bokeh.layouts as bokeh_layouts

            self._bokeh_layouts = bokeh_layouts

        return cast("bokeh_layouts", self._bokeh_layouts)

    @property
    def bokeh_transform(self) -> "bokeh_transform":  # type: ignore[no-any-return]
        """Lazy import of bokeh.transform."""
        if self._bokeh_transform is None:
            import bokeh.transform as bokeh_transform

            self._bokeh_transform = bokeh_transform

        return cast("bokeh_transform", self._bokeh_transform)

    @property
    def bokeh_pallettes(self) -> "bokeh_pallettes":  # type: ignore[no-any-return]
        """Lazy import of bokeh.palettes."""
        if self._bokeh_pallettes is None:
            import bokeh.palettes as bokeh_pallettes

            self._bokeh_pallettes = bokeh_pallettes

        return cast("bokeh_pallettes", self._bokeh_pallettes)


lazy = LazyImports()
