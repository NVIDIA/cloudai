# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Single-shot chat completion client for TTFT benchmark."""

# Future
from __future__ import annotations

# Standard
import argparse
import json
import random
import string
import sys
import time
from pathlib import Path

# Third Party
from openai import OpenAI
from transformers import AutoTokenizer

# ----------------------------------------------------------------------
FILLER_LEN_CHARS = 10_000  # ≈ length of each cache-filler prompt
NUM_FILLER_PROMPTS = 100  # how many fillers to send for eviction
# ----------------------------------------------------------------------


# ---------------- helper utilities ------------------------------------


def log_jsonl(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        json.dump(rec, fh)
        fh.write("\n")


class TtftStats:
    """Holds TTFT benchmark results including timing and token counts."""

    def __init__(self, ttft_seconds: float, prompt_tokens: int, cached_tokens: int):
        self.ttft_seconds = ttft_seconds
        self.prompt_tokens = prompt_tokens
        self.cached_tokens = cached_tokens


class Chat:
    """Represents a chat context with a document for TTFT benchmarking."""

    def __init__(self, isl: int, model_id: str, max_ctx_tokens: int):
        self.isl = isl
        self.model_id = model_id
        self.max_ctx_tokens = max_ctx_tokens
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        raw_doc = "".join(random.choices(string.ascii_letters + string.digits, k=self.isl * 4))

        ids = self.tok.encode(raw_doc, add_special_tokens=False, truncation=True, max_length=self.isl)
        assert len(ids) == self.isl, f"Expected {self.isl} tokens, got {len(ids)}"
        doc = self.tok.decode(ids, skip_special_tokens=True)

        self.messages = [
            {"role": "user", "content": f"I've got a document:\n```\n{doc}\n```"},
            {"role": "assistant", "content": "I've got your document."},
            {"role": "user", "content": "summarize"},
        ]

    def stream(self, client: OpenAI, max_tokens: int) -> TtftStats:
        stats = TtftStats(0, 0, 0)

        start = time.perf_counter()
        stream = client.chat.completions.create(
            model=self.model_id,
            messages=self.messages,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
            max_tokens=max_tokens,
        )

        first_tok_t: float | None = None
        for chunk in stream:
            usage_details = chunk.usage and chunk.usage.prompt_tokens_details
            if usage_details and usage_details.cached_tokens is not None:
                stats.cached_tokens = usage_details.cached_tokens
                stats.prompt_tokens = chunk.usage.prompt_tokens
            if first_tok_t is None and chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                first_tok_t = time.perf_counter()

        if first_tok_t is None:
            raise RuntimeError("no tokens returned")

        stats.ttft_seconds = first_tok_t - start
        return stats


def flush_kv_cache(args: argparse.Namespace, client: OpenAI) -> None:
    """Flush KV cache by sending filler prompts."""
    for _ in range(args.num_filler_prompts):
        _ = Chat(args.filler_len_chars, args.model, args.max_ctx_tokens).stream(client, 1)


# ---------------- command-line parsing --------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name,
        description="Single-shot chat completion client for TTFT benchmark.",
    )
    ap.add_argument("--url", required=True, help="URL of the API endpoint.")
    ap.add_argument("--model", required=True, help="Model name/ID.")
    ap.add_argument("--max_ctx_tokens", type=int, default=131_072, help="Max context tokens.")
    ap.add_argument("--isl", type=int, required=True, help="Input tokens.")
    ap.add_argument("--osl", type=int, required=True, help="Output tokens.")
    ap.add_argument("--out", required=True, help="JSONL file for results.")
    ap.add_argument(
        "--num_filler_prompts",
        type=int,
        default=NUM_FILLER_PROMPTS,
        help="Number of filler prompts to send for cache flush.",
    )
    ap.add_argument(
        "--filler_len_chars",
        type=int,
        default=FILLER_LEN_CHARS,
        help="Length of filler prompt in characters.",
    )
    return ap.parse_args()


# ---------------- main routine ----------------------------------------
def main() -> None:
    args = parse_args()

    chat = Chat(args.isl, args.model, args.max_ctx_tokens)

    client = OpenAI(base_url=args.url)

    # ---------------- RUN 1 ----------------
    print("\n=== Run 1: baseline TTFT ===")
    baseline = chat.stream(client, args.osl)
    print(f"Run 1: TTFT = {baseline.ttft_seconds:.3f}s with {baseline.cached_tokens} cached tokens")

    # Run 2 with same doc without cache flush
    print("\n=== Run 2: TTFT without cache flush ===")
    no_flush = chat.stream(client, args.osl)
    print(f"Run 2: TTFT = {no_flush.ttft_seconds:.3f}s with {no_flush.cached_tokens} cached tokens")

    # Flush cache
    print(f"\nFlushing KV-cache with {NUM_FILLER_PROMPTS} prompts …")
    flush_kv_cache(args, client)

    # Run 3 with same doc with cache flush
    print("\n=== Run 3: TTFT with cache flush ===")
    post_flush = chat.stream(client, args.osl)
    print(f"Run 3: TTFT = {post_flush.ttft_seconds:.3f}s with {post_flush.cached_tokens} cached tokens")

    out_path = Path(args.out)
    with out_path.open("a", encoding="utf-8") as f:
        if out_path.suffix == ".csv":
            f.write(
                f"{args.isl},{baseline.prompt_tokens},{baseline.cached_tokens},"
                f"{baseline.ttft_seconds},{no_flush.cached_tokens},{no_flush.ttft_seconds},"
                f"{post_flush.cached_tokens},{post_flush.ttft_seconds}\n"
            )
        else:
            json_data = {
                "isl": args.isl,
                "context_tokens": baseline.prompt_tokens,
                "baseline_cached_tokens": baseline.cached_tokens,
                "baseline_ttft_seconds": baseline.ttft_seconds,
                "no_flush_cached_tokens": no_flush.cached_tokens,
                "no_flush_ttft_seconds": no_flush.ttft_seconds,
                "post_flush_cached_tokens": post_flush.cached_tokens,
                "post_flush_ttft_seconds": post_flush.ttft_seconds,
            }

            json.dump(json_data, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
