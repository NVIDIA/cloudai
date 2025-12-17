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

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run aiconfigurator simple predictor and write results to a file.",
    )
    # mode
    parser.add_argument("--mode", choices=["agg", "disagg"], default="agg")

    # core
    parser.add_argument("--model-name", required=True, type=str)
    parser.add_argument("--system", required=True, type=str)
    parser.add_argument("--backend", default="trtllm", type=str)
    parser.add_argument("--version", default="0.20.0", type=str)

    # runtime
    parser.add_argument("--isl", required=True, type=int)
    parser.add_argument("--osl", required=True, type=int)
    parser.add_argument("--batch-size", required=False, type=int, help="Batch size (agg mode)")
    parser.add_argument("--ctx-tokens", required=False, type=int, help="Context tokens (agg mode)")

    # parallel config (agg)
    parser.add_argument("--tp", default=1, type=int)
    parser.add_argument("--pp", default=1, type=int)
    parser.add_argument("--dp", default=1, type=int)
    parser.add_argument("--moe-tp", default=1, type=int)
    parser.add_argument("--moe-ep", default=1, type=int)

    # disagg params
    parser.add_argument("--p-tp", dest="p_tp", type=int)
    parser.add_argument("--p-pp", dest="p_pp", type=int)
    parser.add_argument("--p-dp", dest="p_dp", type=int)
    parser.add_argument("--p-bs", dest="p_bs", type=int)
    parser.add_argument("--p-workers", dest="p_workers", type=int)

    parser.add_argument("--d-tp", dest="d_tp", type=int)
    parser.add_argument("--d-pp", dest="d_pp", type=int)
    parser.add_argument("--d-dp", dest="d_dp", type=int)
    parser.add_argument("--d-bs", dest="d_bs", type=int)
    parser.add_argument("--d-workers", dest="d_workers", type=int)

    parser.add_argument("--prefill-correction-scale", type=float, default=1.0)
    parser.add_argument("--decode-correction-scale", type=float, default=1.0)

    # output
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to write predictor JSON output (filename is user-specified).",
    )

    # optional quantization and features (strings to be converted by SDK)
    parser.add_argument("--gemm-quant-mode", default="fp8_block")
    parser.add_argument("--moe-quant-mode", default="fp8")
    parser.add_argument("--kvcache-quant-mode", default="fp8")
    parser.add_argument("--fmha-quant-mode", default="fp8")
    parser.add_argument("--comm-quant-mode", default="half")
    parser.add_argument("--nextn", default=0, type=int)
    parser.add_argument(
        "--nextn-accept-rates",
        nargs="+",
        type=float,
        default=None,
        help="Acceptance rates for nextn speculative decoding (space-separated floats). Required when --nextn > 0.",
    )

    return parser.parse_args()


def _run_agg(ns: argparse.Namespace) -> dict:
    from cloudai.workloads.aiconfig.predictor import predict_ifb_single

    if ns.batch_size is None or ns.ctx_tokens is None:
        raise ValueError("--batch-size and --ctx-tokens are required in agg mode")

    return predict_ifb_single(
        model_name=ns.model_name,
        system=ns.system,
        backend=ns.backend,
        version=ns.version,
        isl=ns.isl,
        osl=ns.osl,
        batch_size=ns.batch_size,
        ctx_tokens=ns.ctx_tokens,
        tp=ns.tp,
        pp=ns.pp,
        dp=ns.dp,
        moe_tp=ns.moe_tp,
        moe_ep=ns.moe_ep,
        gemm_quant_mode=ns.gemm_quant_mode,
        moe_quant_mode=ns.moe_quant_mode,
        kvcache_quant_mode=ns.kvcache_quant_mode,
        fmha_quant_mode=ns.fmha_quant_mode,
        comm_quant_mode=ns.comm_quant_mode,
        nextn=ns.nextn,
        nextn_accept_rates=ns.nextn_accept_rates,
    )


def _run_disagg(ns: argparse.Namespace) -> dict:
    from cloudai.workloads.aiconfig.predictor import predict_disagg_single

    required = ["p_tp", "p_pp", "p_dp", "p_bs", "p_workers", "d_tp", "d_pp", "d_dp", "d_bs", "d_workers"]
    missing = [k for k in required if getattr(ns, k) is None]
    if missing:
        raise ValueError(f"Missing required disagg params: {', '.join(missing)}")

    return predict_disagg_single(
        model_name=ns.model_name,
        system=ns.system,
        backend=ns.backend,
        version=ns.version,
        isl=ns.isl,
        osl=ns.osl,
        p_tp=ns.p_tp,
        p_pp=ns.p_pp,
        p_dp=ns.p_dp,
        p_bs=ns.p_bs,
        p_workers=ns.p_workers,
        d_tp=ns.d_tp,
        d_pp=ns.d_pp,
        d_dp=ns.d_dp,
        d_bs=ns.d_bs,
        d_workers=ns.d_workers,
        gemm_quant_mode=ns.gemm_quant_mode,
        moe_quant_mode=ns.moe_quant_mode,
        kvcache_quant_mode=ns.kvcache_quant_mode,
        fmha_quant_mode=ns.fmha_quant_mode,
        comm_quant_mode=ns.comm_quant_mode,
        nextn=ns.nextn,
        nextn_accept_rates=ns.nextn_accept_rates,
        prefill_correction_scale=ns.prefill_correction_scale,
        decode_correction_scale=ns.decode_correction_scale,
    )


def main() -> int:
    ns = parse_args()

    try:
        result = _run_agg(ns) if ns.mode == "agg" else _run_disagg(ns)
    except Exception as e:
        print(f"Prediction failed: {e}", file=sys.stderr)
        return 2

    try:
        ns.output.parent.mkdir(parents=True, exist_ok=True)
        with ns.output.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        keys = ["ttft_ms", "tpot_ms", "tokens_per_s_per_gpu", "tokens_per_s_per_user", "oom"]
        summary = {k: result.get(k) for k in keys if k in result}
        print(json.dumps(summary))
    except Exception as e:
        print(f"Failed to write output: {e}", file=sys.stderr)
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
