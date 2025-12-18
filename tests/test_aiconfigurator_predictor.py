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

import pandas as pd
import pytest

from cloudai.workloads.aiconfig.predictor import predict_disagg_single, predict_ifb_single


class _Summary:
    def __init__(self, df: pd.DataFrame, oom: bool = False):
        self._df = df
        self._oom = oom

    def get_summary_df(self) -> pd.DataFrame:
        return self._df

    def check_oom(self) -> bool:
        return self._oom


class _Backend:
    def __init__(self, ifb_df: pd.DataFrame):
        self._ifb_df = ifb_df

    def run_ifb(self, *, model, database, runtime_config, ctx_tokens: int) -> _Summary:
        return _Summary(self._ifb_df, oom=False)


class _FakeInferenceSession:
    def __init__(self, p_df: pd.DataFrame, d_df: pd.DataFrame):
        self._summaries = {
            "static_ctx": _Summary(p_df, oom=False),
            "static_gen": _Summary(d_df, oom=False),
        }

    def run_static(self, *, mode: str, runtime_config) -> _Summary:
        try:
            return self._summaries[mode]
        except KeyError as e:
            raise ValueError("unexpected mode") from e


def _patch_aiconfigurator(
    monkeypatch: pytest.MonkeyPatch, *, ifb_df: pd.DataFrame, p_df: pd.DataFrame, d_df: pd.DataFrame
) -> None:
    """
    Patch the installed `aiconfigurator` package so predictor unit tests don't depend on perf DB files
    or any heavy backend behavior.
    """
    pytest.importorskip("aiconfigurator")

    from aiconfigurator.sdk.backends import factory as backends_factory

    def get_backend(name: str):
        return _Backend(ifb_df)

    def get_database(*, system: str, backend: str, version: str):
        return object()

    def get_model(model_name: str, model_config, backend_name: str):
        return {"model_name": model_name, "backend_name": backend_name, "cfg": model_config}

    class InferenceSession:
        def __init__(self, model, db, backend):
            self._model = model
            self._db = db
            self._backend = backend
            self._impl = _FakeInferenceSession(p_df=p_df, d_df=d_df)

        def run_static(self, *, mode: str, runtime_config):
            return self._impl.run_static(mode=mode, runtime_config=runtime_config)

    monkeypatch.setattr(backends_factory, "get_backend", get_backend, raising=True)
    monkeypatch.setattr("aiconfigurator.sdk.perf_database.get_database", get_database, raising=True)
    monkeypatch.setattr("aiconfigurator.sdk.models.get_model", get_model, raising=True)
    monkeypatch.setattr("aiconfigurator.sdk.inference_session.InferenceSession", InferenceSession, raising=True)


def test_predict_ifb_single_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    ifb_df = pd.DataFrame(
        [
            {
                "ttft": 10.0,
                "tpot": 2.0,
                "tokens/s/gpu": 3.0,
                "tokens/s/user": 4.0,
                "seq/s/gpu": 5.0,
                "tokens/s": 6.0,
                "seq/s": 7.0,
                "concurrency": 8.0,
                "num_total_gpus": 2,
                "global_bs": 16,
                "bs": 8,
                "backend": "trtllm",
                "version": "0.20.0",
                "system": "h200_sxm",
                "pp": 1,
                "tp": 1,
                "dp": 1,
            }
        ]
    )
    p_df = pd.DataFrame([{"seq/s": 1.0, "pp": 1, "tp": 1, "dp": 1, "ttft": 10.0, "bs": 1}])
    d_df = pd.DataFrame(
        [{"seq/s": 2.0, "pp": 1, "tp": 1, "dp": 1, "tpot": 2.0, "bs": 8, "tokens/s/user": 4.0, "concurrency": 1.0}]
    )
    _patch_aiconfigurator(monkeypatch, ifb_df=ifb_df, p_df=p_df, d_df=d_df)

    out = predict_ifb_single(
        model_name="LLAMA3.1_70B",
        system="h200_sxm",
        backend="trtllm",
        version="0.20.0",
        isl=4000,
        osl=500,
        batch_size=8,
        ctx_tokens=16,
    )
    assert out["oom"] is False
    assert out["ttft_ms"] == 10.0
    assert out["tpot_ms"] == 2.0
    assert out["tokens_per_s_per_gpu"] == 3.0


def test_predict_disagg_single_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    ifb_df = pd.DataFrame(
        [
            {
                "ttft": 10.0,
                "tpot": 2.0,
                "tokens/s/gpu": 3.0,
                "tokens/s/user": 4.0,
                "seq/s/gpu": 5.0,
                "tokens/s": 6.0,
                "seq/s": 7.0,
                "concurrency": 8.0,
                "num_total_gpus": 2,
                "global_bs": 16,
                "bs": 8,
                "backend": "trtllm",
                "version": "0.20.0",
                "system": "h200_sxm",
                "pp": 1,
                "tp": 1,
                "dp": 1,
            }
        ]
    )
    p_df = pd.DataFrame([{"seq/s": 1.0, "pp": 1, "tp": 1, "dp": 1, "ttft": 10.0, "bs": 1}])
    d_df = pd.DataFrame(
        [{"seq/s": 10.0, "pp": 1, "tp": 1, "dp": 1, "tpot": 2.0, "bs": 8, "tokens/s/user": 4.0, "concurrency": 2.0}]
    )
    _patch_aiconfigurator(monkeypatch, ifb_df=ifb_df, p_df=p_df, d_df=d_df)

    out = predict_disagg_single(
        model_name="LLAMA3.1_70B",
        system="h200_sxm",
        backend="trtllm",
        version="0.20.0",
        isl=4000,
        osl=500,
        p_tp=1,
        p_pp=1,
        p_dp=1,
        p_bs=1,
        p_workers=1,
        d_tp=1,
        d_pp=1,
        d_dp=1,
        d_bs=8,
        d_workers=1,
    )
    assert out["oom"] is False
    assert out["seq_per_s_total"] == 1.0
    assert out["tokens_per_s_total"] == 1.0 * 500
    assert out["ttft_ms"] == 10.0
    assert out["tpot_ms"] == 2.0
