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
import os
import random
import string
import sys
import time
from pathlib import Path

# Third Party
from openai import OpenAI  # type: ignore[import-untyped]
from transformers import AutoTokenizer  # type: ignore[import-untyped]

# ----------------------------------------------------------------------
NUM_FILLER_TOKENS = 10_000  # ≈ length of each cache-filler prompt
NUM_FILLER_PROMPTS = 100  # how many fillers to send for eviction
# ----------------------------------------------------------------------


# ---------------- helper utilities ------------------------------------


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')} {os.getenv('HOSTNAME') or ''}]: [openai_chat_client] {message}")
    sys.stdout.flush()
    sys.stderr.flush()


class TtftStats:
    """Holds TTFT benchmark results including timing and token counts."""

    def __init__(self, ttft_seconds: float, prompt_tokens: int, cached_tokens: int):
        self.ttft_seconds = ttft_seconds
        self.prompt_tokens = prompt_tokens
        self.cached_tokens = cached_tokens


class Chat:
    """Represents a chat context with a document for TTFT benchmarking."""

    def __init__(self, model: str, isl: int):
        self.isl = isl
        self.model = model
        self.tok = AutoTokenizer.from_pretrained(self.model, use_fast=True)

        raw_doc = "".join(random.choices(string.ascii_letters + string.digits, k=self.isl * 4))

        num_tokens = self.isl - 37
        ids = self.tok.encode(raw_doc, add_special_tokens=False, truncation=True, max_length=num_tokens)
        assert len(ids) == num_tokens, f"Expected {num_tokens} tokens, got {len(ids)}"
        doc = self.tok.decode(ids, skip_special_tokens=True)

        self.messages = [
            {"role": "user", "content": f"I've got a document:\n```\n{doc}\n```"},
            {"role": "assistant", "content": "I've got your document."},
            {"role": "user", "content": "summarize"},
        ]

    def stream(self, client: OpenAI, max_tokens: int) -> TtftStats:
        stats = TtftStats(0, 0, 0)

        start = time.perf_counter()
        try:
            stream = client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=0.0,
                stream=True,
                stream_options={"include_usage": True},
                max_tokens=max_tokens,
            )

            first_tok_t: float | None = None
            for chunk in stream:
                if first_tok_t is None and chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    first_tok_t = time.perf_counter()
                # Capture prompt_tokens from usage if available
                if chunk.usage and chunk.usage.prompt_tokens:
                    stats.prompt_tokens = chunk.usage.prompt_tokens
                # Capture cached_tokens from prompt_tokens_details if available
                usage_details = chunk.usage and chunk.usage.prompt_tokens_details
                if usage_details and usage_details.cached_tokens is not None:
                    stats.cached_tokens = usage_details.cached_tokens

            if first_tok_t is None:
                raise RuntimeError("no tokens returned")

            stats.ttft_seconds = round(first_tok_t - start, 3)
            return stats
        except json.JSONDecodeError as e:
            log(f"Error: JSON decode error during streaming: {e}")
            log("This may indicate empty SSE events from the server - likely a server-side bug")
            # Return partial stats with error indication
            stats.ttft_seconds = -1  # Indicate error
            return stats
        except Exception as e:
            log(f"Error during streaming: {type(e).__name__}: {e}")
            stats.ttft_seconds = -1  # Indicate error
            return stats


class KVCacheFlusher:
    """Flushes the KV cache by streaming filler chat completions."""

    def __init__(self, args: argparse.Namespace, client: OpenAI):
        self.client = client
        self.args = args
        self.filler_chats = [Chat(args.model, args.num_filler_tokens) for _ in range(args.num_filler_prompts)]

    def flush(self) -> None:
        log(f"Stream {self.args.num_filler_prompts} filler chats with {self.args.num_filler_tokens} tokens each...")
        for _n, chat in enumerate(self.filler_chats):
            chat.stream(self.client, 1)


# ---------------- command-line parsing --------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name,
        description="Single-shot chat completion client for TTFT benchmark.",
    )
    ap.add_argument("--dump_csv_header", action="store_true", help="Only dump CSV header and exit.")
    ap.add_argument("--url", help="URL of the API endpoint.")
    ap.add_argument("--model", help="Model name/ID.")
    ap.add_argument("--isl", type=int, help="Input tokens.")
    ap.add_argument("--osl", type=int, help="Output tokens.")
    ap.add_argument("--out", help="JSONL file for results.")
    ap.add_argument(
        "--max_filler_prompts",
        type=int,
        default=200,
        help="Max number of filler prompts (used to compute the KV cache token size) to send for cache flush.",
    )
    ap.add_argument(
        "--min_filler_prompts",
        type=int,
        default=1,
        help="Min number of filler prompts (used to compute the KV cache token size) to send for cache flush.",
    )
    ap.add_argument(
        "--num_filler_prompts",
        type=int,
        default=NUM_FILLER_PROMPTS,
        help="Number of filler prompts to send for cache flush.",
    )
    ap.add_argument(
        "--num_filler_tokens",
        type=int,
        default=NUM_FILLER_TOKENS,
        help="Number of filler tokens.",
    )
    ap.add_argument("--compute_kv_cache_token_size", action="store_true", help="Compute KV cache token size and exit.")
    return ap.parse_args()


def SendFillerQueries(args: argparse.Namespace, client: OpenAI, num: int):
    for n in range(num):
        log(f"Sending filler query {n + 1} of {num}...")
        _ = Chat(args.model, args.isl).stream(client, 1)


def compute_kv_cache_token_size(args: argparse.Namespace, client: OpenAI) -> int:
    # We want to compute the number of tokens required to flush the KV cache. To
    # do this, we start by sending a canary query with 1000 tokens.
    # Next we send a filler queries with 10000 tokens and after each query we
    # send the original query again aand measure the cached_tokens. If
    # cached_tokens is not zero, we increase the number of filler queries and
    # repeat.  At some point, the cached_tokens for the original query will be
    # zero and we have the number of filler queries required to flush the KV
    # cache.

    # Do a binary search for the number of filler prompts required to flush the KV cache.
    maxFillerPrompts = args.max_filler_prompts
    minFillerPrompts = min(1, args.min_filler_prompts)
    log(
        f"Doing binary search for the number of filler prompts required to flush the KV cache"
        f" between {minFillerPrompts} and {maxFillerPrompts}..."
    )

    log("Sending an initial canary query with 1000 tokens...")
    canary_chat = Chat(args.model, args.isl)
    canary_stats = canary_chat.stream(client, 1)
    log(f"Initial Canary query: {canary_stats.ttft_seconds:.3f}s with {canary_stats.cached_tokens} cached tokens")

    while minFillerPrompts < maxFillerPrompts:
        numFillerPrompts = (maxFillerPrompts + minFillerPrompts) // 2
        log(f"Trying {numFillerPrompts} filler prompts with {args.num_filler_tokens} tokens each...")
        SendFillerQueries(args, client, numFillerPrompts)
        log(f"Sending canary query after {numFillerPrompts} filler prompts...")
        canary_stats = canary_chat.stream(client, 1)
        log(f"Canary query: {canary_stats.ttft_seconds:.3f}s with {canary_stats.cached_tokens} cached tokens")
        if canary_stats.cached_tokens < 500:
            maxFillerPrompts = numFillerPrompts
        else:
            minFillerPrompts = numFillerPrompts + 1
        log(f"Min filler prompts: {minFillerPrompts}, Max filler prompts: {maxFillerPrompts}")
    return minFillerPrompts * args.num_filler_tokens


# ---------------- main routine ----------------------------------------
def main() -> None:
    args = parse_args()

    result = {
        "isl": args.isl,
        "baseline_cached_tokens": 0,
        "baseline_ttft_seconds": 0,
        "no_flush_cached_tokens": 0,
        "no_flush_ttft_seconds": 0,
        "post_flush_cached_tokens": 0,
        "post_flush_ttft_seconds": 0,
    }

    client = OpenAI(base_url=args.url, api_key="dummy-key-for-local-server")

    if args.compute_kv_cache_token_size:
        log("Computing KV cache token size...")
        kv_cache_token_size = compute_kv_cache_token_size(args, client)
        log(f"KV cache token size: {kv_cache_token_size}")
        with Path(args.out).open("a", encoding="utf-8") as f:
            f.write(f"KV cache token size: {kv_cache_token_size}\n")
        return

    if args.dump_csv_header:
        with Path(args.out).open("a", encoding="utf-8") as f:
            f.write(",".join(result.keys()))
            f.write("\n")
        return

    chat = Chat(args.model, args.isl)

    log("=== Run 1: warmup ===")
    warmup = Chat(args.model, args.isl).stream(client, 1)
    log(f"Run 1: warmup: TTFT = {warmup.ttft_seconds:.3f}s with {warmup.cached_tokens} cached tokens")

    # ---------------- RUN 1 ----------------
    log("=== Run 1: baseline TTFT ===")
    baseline = chat.stream(client, args.osl)
    log(f"Run 1: TTFT = {baseline.ttft_seconds:.3f}s with {baseline.cached_tokens} cached tokens")

    # Run 2 with same doc without cache flush
    log("=== Run 2: TTFT without cache flush ===")
    no_flush = chat.stream(client, args.osl)
    log(f"Run 2: TTFT = {no_flush.ttft_seconds:.3f}s with {no_flush.cached_tokens} cached tokens")

    # Flush cache
    log(f"Flushing KV-cache with {args.num_filler_prompts} prompts …")
    KVCacheFlusher(args, client).flush()

    # Run 3 with same doc with cache flush
    log("=== Run 3: warmup ===")
    warmup = Chat(args.model, args.isl).stream(client, 1)
    log(f"Run 3: warmup: TTFT = {warmup.ttft_seconds:.3f}s with {warmup.cached_tokens} cached tokens")

    log("=== Run 3: TTFT with cache flush ===")
    post_flush = chat.stream(client, args.osl)
    log(f"Run 3: TTFT = {post_flush.ttft_seconds:.3f}s with {post_flush.cached_tokens} cached tokens")

    result["baseline_cached_tokens"] = baseline.cached_tokens
    result["baseline_ttft_seconds"] = baseline.ttft_seconds
    result["no_flush_cached_tokens"] = no_flush.cached_tokens
    result["no_flush_ttft_seconds"] = no_flush.ttft_seconds
    result["post_flush_cached_tokens"] = post_flush.cached_tokens
    result["post_flush_ttft_seconds"] = post_flush.ttft_seconds

    out_path = Path(args.out)
    with out_path.open("a", encoding="utf-8") as f:
        if out_path.suffix == ".csv":
            line = ",".join(str(v) for v in result.values())
            f.write(line + "\n")
        else:
            json.dump(result, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
