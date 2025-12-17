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

from typing import Any, Dict, Optional, cast

from aiconfigurator.sdk import common
from aiconfigurator.sdk import config as aic_config
from aiconfigurator.sdk import inference_session as aic_inference_session
from aiconfigurator.sdk import models as aic_models
from aiconfigurator.sdk import perf_database as aic_perf_database
from aiconfigurator.sdk.backends import factory as aic_backends_factory


def _to_enum(enum_cls: Any, value_or_name: Any) -> Any:
    """
    Convert a string or enum value to the corresponding enum instance.

    If value_or_name is already an enum instance, return it as-is.
    """
    if isinstance(value_or_name, enum_cls):
        return value_or_name
    if isinstance(value_or_name, str):
        return enum_cls[value_or_name]
    return enum_cls(value_or_name)


def _validate_nextn(nextn: int, nextn_accept_rates: Optional[list[float]]) -> list[float]:
    if nextn > 0 and nextn_accept_rates is None:
        raise ValueError("nextn_accept_rates must be provided when nextn > 0")
    return nextn_accept_rates or []


def predict_ifb_single(
    *,
    model_name: str,
    system: str,
    backend: str = "trtllm",
    version: str = "0.20.0",
    # runtime
    isl: int,
    osl: int,
    batch_size: int,
    ctx_tokens: int,
    # parallel config
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    moe_tp: int = 1,
    moe_ep: int = 1,
    # quantization
    gemm_quant_mode: Any = "fp8_block",
    moe_quant_mode: Any = "fp8",
    kvcache_quant_mode: Any = "fp8",
    fmha_quant_mode: Any = "fp8",
    comm_quant_mode: Any = "half",
    # features
    nextn: int = 0,
    nextn_accept_rates: Optional[list[float]] = None,
    # advanced model options
    overwrite_num_layers: int = 0,
) -> Dict[str, Any]:
    """Predict metrics for a single IFB configuration using the aiconfigurator SDK primitives."""
    database = aic_perf_database.get_database(system=system, backend=backend, version=version)
    if database is None:
        raise ValueError(f"No perf database found for system={system} backend={backend} version={version}")
    backend_impl = cast(Any, aic_backends_factory.get_backend(backend))

    accept_rates = _validate_nextn(nextn, nextn_accept_rates)

    mc = aic_config.ModelConfig(
        tp_size=tp,
        pp_size=pp,
        attention_dp_size=dp,
        moe_tp_size=moe_tp,
        moe_ep_size=moe_ep,
        gemm_quant_mode=_to_enum(common.GEMMQuantMode, gemm_quant_mode),
        moe_quant_mode=_to_enum(common.MoEQuantMode, moe_quant_mode),
        kvcache_quant_mode=_to_enum(common.KVCacheQuantMode, kvcache_quant_mode),
        fmha_quant_mode=_to_enum(common.FMHAQuantMode, fmha_quant_mode),
        comm_quant_mode=_to_enum(common.CommQuantMode, comm_quant_mode),
        nextn=nextn,
        nextn_accept_rates=accept_rates,
        overwrite_num_layers=overwrite_num_layers,
    )
    model = aic_models.get_model(model_name, mc, backend)

    rc = aic_config.RuntimeConfig(batch_size=batch_size, isl=isl, osl=osl)
    summary = backend_impl.run_ifb(model=model, database=database, runtime_config=rc, ctx_tokens=ctx_tokens)
    df = summary.get_summary_df()
    if df is None or df.empty:
        return {"oom": summary.check_oom()}

    row = df.iloc[0]
    return {
        "ttft_ms": float(row["ttft"]),
        "tpot_ms": float(row["tpot"]),
        "tokens_per_s_per_gpu": float(row["tokens/s/gpu"]),
        "tokens_per_s_per_user": float(row["tokens/s/user"]),
        "seq_per_s_per_gpu": float(row["seq/s/gpu"]),
        "tokens_per_s_total": float(row["tokens/s"]),
        "seq_per_s_total": float(row["seq/s"]),
        "concurrency": float(row["concurrency"]),
        "num_total_gpus": int(row["num_total_gpus"]),
        "global_batch_size": int(row["global_bs"]),
        "batch_size": int(row["bs"]),
        "backend": str(row["backend"]),
        "version": str(row["version"]),
        "system": str(row["system"]),
        "oom": bool(summary.check_oom()),
    }


def predict_disagg_single(
    *,
    model_name: str,
    system: str,
    backend: str = "trtllm",
    version: str = "0.20.0",
    # runtime
    isl: int,
    osl: int,
    # prefill worker config
    p_tp: int,
    p_pp: int,
    p_dp: int,
    p_bs: int,
    p_workers: int,
    # decode worker config
    d_tp: int,
    d_pp: int,
    d_dp: int,
    d_bs: int,
    d_workers: int,
    # quantization (can be same for both)
    gemm_quant_mode: Any = "fp8_block",
    moe_quant_mode: Any = "fp8",
    kvcache_quant_mode: Any = "fp8",
    fmha_quant_mode: Any = "fp8",
    comm_quant_mode: Any = "half",
    # features
    nextn: int = 0,
    nextn_accept_rates: Optional[list[float]] = None,
    overwrite_num_layers: int = 0,
    # correction scales
    prefill_correction_scale: float = 1.0,
    decode_correction_scale: float = 1.0,
) -> Dict[str, Any]:
    """Predict metrics for a single disaggregated configuration (explicit prefill/decode workers)."""
    perf_db = aic_perf_database.get_database(system=system, backend=backend, version=version)
    if perf_db is None:
        raise ValueError(f"No perf database found for system={system} backend={backend} version={version}")

    perf_backend = cast(Any, aic_backends_factory.get_backend(backend))

    accept_rates = _validate_nextn(nextn, nextn_accept_rates)

    p_mc = aic_config.ModelConfig(
        tp_size=p_tp,
        pp_size=p_pp,
        attention_dp_size=p_dp,
        gemm_quant_mode=_to_enum(common.GEMMQuantMode, gemm_quant_mode),
        moe_quant_mode=_to_enum(common.MoEQuantMode, moe_quant_mode),
        kvcache_quant_mode=_to_enum(common.KVCacheQuantMode, kvcache_quant_mode),
        fmha_quant_mode=_to_enum(common.FMHAQuantMode, fmha_quant_mode),
        comm_quant_mode=_to_enum(common.CommQuantMode, comm_quant_mode),
        moe_tp_size=1,
        moe_ep_size=1,
        nextn=nextn,
        nextn_accept_rates=accept_rates,
        overwrite_num_layers=overwrite_num_layers,
    )
    d_mc = aic_config.ModelConfig(
        tp_size=d_tp,
        pp_size=d_pp,
        attention_dp_size=d_dp,
        gemm_quant_mode=_to_enum(common.GEMMQuantMode, gemm_quant_mode),
        moe_quant_mode=_to_enum(common.MoEQuantMode, moe_quant_mode),
        kvcache_quant_mode=_to_enum(common.KVCacheQuantMode, kvcache_quant_mode),
        fmha_quant_mode=_to_enum(common.FMHAQuantMode, fmha_quant_mode),
        comm_quant_mode=_to_enum(common.CommQuantMode, comm_quant_mode),
        moe_tp_size=1,
        moe_ep_size=1,
        nextn=nextn,
        nextn_accept_rates=accept_rates,
        overwrite_num_layers=overwrite_num_layers,
    )

    rc_prefill = aic_config.RuntimeConfig(batch_size=p_bs, isl=isl, osl=osl)
    rc_decode = aic_config.RuntimeConfig(batch_size=d_bs, isl=isl, osl=osl)

    prefill_model = aic_models.get_model(model_name, p_mc, backend)
    decode_model = aic_models.get_model(model_name, d_mc, backend)

    prefill_sess = aic_inference_session.InferenceSession(prefill_model, perf_db, perf_backend)
    decode_sess = aic_inference_session.InferenceSession(decode_model, perf_db, perf_backend)

    prefill_summary = prefill_sess.run_static(mode="static_ctx", runtime_config=rc_prefill)
    decode_summary = decode_sess.run_static(mode="static_gen", runtime_config=rc_decode)

    if prefill_summary.check_oom() or decode_summary.check_oom():
        return {"oom": True}

    p_df = prefill_summary.get_summary_df()
    d_df = decode_summary.get_summary_df()
    if p_df is None or p_df.empty or d_df is None or d_df.empty:
        return {"oom": True}

    p = p_df.iloc[0]
    d = d_df.iloc[0]

    seq_s_prefill = float(p["seq/s"]) * p_workers * prefill_correction_scale
    seq_s_decode = float(d["seq/s"]) * d_workers * decode_correction_scale
    seq_s = min(seq_s_prefill, seq_s_decode)

    prefill_gpus = int(p["pp"]) * int(p["tp"]) * int(p["dp"])
    decode_gpus = int(d["pp"]) * int(d["tp"]) * int(d["dp"])
    total_gpus = prefill_gpus * p_workers + decode_gpus * d_workers

    seq_s_gpu = seq_s / total_gpus if total_gpus > 0 else 0.0
    ttft = float(p["ttft"])
    tpot = float(d["tpot"])

    tokens_s = seq_s * osl
    tokens_s_gpu = tokens_s / total_gpus if total_gpus > 0 else 0.0
    tokens_s_user = float(d["tokens/s/user"])
    concurrency = float(d["concurrency"]) * d_workers

    return {
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "tokens_per_s_per_gpu": tokens_s_gpu,
        "tokens_per_s_per_user": tokens_s_user,
        "seq_per_s_per_gpu": seq_s_gpu,
        "tokens_per_s_total": tokens_s,
        "seq_per_s_total": seq_s,
        "concurrency": concurrency,
        "num_total_gpus": int(total_gpus),
        "prefill_workers": int(p_workers),
        "decode_workers": int(d_workers),
        "prefill_bs": int(p["bs"]),
        "decode_bs": int(d["bs"]),
        "oom": False,
    }
