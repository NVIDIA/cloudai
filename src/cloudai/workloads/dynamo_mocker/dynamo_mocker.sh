#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES All rights reserved.
#
# dynamo_mocker.sh — GPU-free dynamo inference simulation mirroring ai_dynamo.sh.
#
# Mockable counterparts of ai_dynamo.sh components:
#   etcd              → --discovery-backend file  (no etcd process needed)
#   nats-server       → still required (DistributedRuntime event plane)
#   dynamo.frontend   → identical, used as-is
#   prefill workers   → dynamo.mocker --disaggregation-mode prefill
#   decode workers    → dynamo.mocker --disaggregation-mode decode
#   KV block manager  → simulated internally by mocker (block alloc + LRU eviction)
#   KV events         → published to NATS; kv router mode on frontend responds to them
#   nixl KV transfer  → simulated via --kv-transfer-bandwidth (no real GPU memory move)
#   LMCache           → NOT supported (multi-tier memory N/A for mocker)
#   GPU allocation    → N/A (no GPUs needed)
#   nvidia-smi        → N/A
#   genai-perf        → identical, used as-is
#
# Modes:
#   none           — single mocker process handles combined prefill+decode (default)
#   prefill_decode — separate prefill and decode mocker instances, mirrors
#                    ai_dynamo.sh's disaggregated worker topology
#
# Usage:
#   bash dynamo_mocker.sh --result-dir <dir> [options...]

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Python: resolved in main() after parse_args ──────────────────────────
# Set by resolve_python() using --venv-python (CloudAI PythonEnvironment managed venv).
# No default — resolve_python() will fail if --venv-python is not provided.
PYTHON=""


# ── Thread-pool caps ─────────────────────────────────────────────────────
# The user cgroup on login nodes is typically limited to ~640 PIDs total.
# On a 128-CPU node, Python processes that import numpy load OpenBLAS, which
# spawns 128 threads by default. Two mocker processes + cloudai + genai-perf
# each doing this would alone account for 4×128=512 threads, leaving almost
# no room before hitting the 640 PID cap.
# OMP_NUM_THREADS=1 limits all OpenBLAS/OpenMP thread pools to 1 thread.
# GOMAXPROCS=4 limits NATS's Go runtime to 4 OS threads (default is cpu_count).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export GOMAXPROCS=4               # Go runtime: 4 OS threads (NATS)

# ── Logging env setup (mirrors ai_dynamo.sh) ──────────────────────────────
export DYN_SDK_DISABLE_ANSI_LOGGING=1
export VLLM_DISABLE_COLORED_OUTPUT=1
export VLLM_NO_COLOR=1
export DYN_LOGGING_DISABLE_ANSI_COLORS=1
export TERM=dumb
export NO_COLOR=1
export TQDM_DISABLE=1
export TQDM_MININTERVAL=999999

# ── Defaults ──────────────────────────────────────────────────────────────
venv_python=""          # path to venv python3, passed via --venv-python from CloudAI
result_dir=""
model_path="Qwen/Qwen3-0.6B"
speedup_ratio=1.0
block_size=64
num_gpu_blocks_override=16384
enable_prefix_caching="true"
http_port=8000
router_mode="round_robin"
input_tokens=5000
output_tokens=500
request_count=1000
replay_concurrency=100
replay_mode="offline"

# Disaggregated mode
disaggregation_mode="none"      # none | prefill_decode
num_workers=1                   # combined mode: simulated workers in one process
# Disaggregated mode: per-role instance counts and launch commands
prefill_num_nodes=1             # number of prefill mocker instances
decode_num_nodes=1              # number of decode mocker instances
prefill_cmd=""   # set after resolve_python() in main(); overridable via --prefill-cmd
decode_cmd=""    # set after resolve_python() in main(); overridable via --decode-cmd
kv_transfer_bandwidth=200.0     # GB/s — simulates nixl KV migration latency
# Per-worker regex: scanned in log files to determine when each role is ready.
# Mirrors ai_dynamo.sh's worker-initialized-regex (now split per role).
prefill_initialized_regex="created and running"
decode_initialized_regex="created and running"

# NATS server command (override with full path if nats-server is not on PATH)
nats_cmd="nats-server -js"

# Benchmark tool selection
benchmark_tool="genai_perf"          # genai_perf | aiperf
aiperf_cmd=""                        # override aiperf command (empty = aiperf.sh default)
genai_perf_cmd=""                    # override genai-perf command (empty = genai_perf.sh default)
aiperf_extra_args_raw=""             # raw CLI string appended to aiperf profile command
genai_perf_extra_args_raw=""         # raw CLI string appended to genai-perf profile command
declare -A aiperf_extra_args=()      # extra aiperf flags from --aiperf-<key> <val>
declare -A genai_perf_extra_args=()  # extra genai-perf flags from --genai-perf-<key> <val>
declare -A engine_extra_args=()      # extra dynamo.mocker engine flags from --engine-<key> <val>
declare -A mocker_extra_args=()      # extra dynamo.mocker topology flags from --mocker-<key> <val>
declare -A frontend_extra_args=()    # extra dynamo.frontend flags from --frontend-<key> <val>
declare -A prefill_args_extra=()     # named flags for prefill mockers from --prefill-args-<key> <val>
declare -A decode_args_extra=()      # named flags for decode mockers from --decode-args-<key> <val>
prefill_extra_args_raw=""            # raw CLI string for prefill mockers from --prefill-extra-args
decode_extra_args_raw=""             # raw CLI string for decode mockers from --decode-extra-args

# ── ERR trap ──────────────────────────────────────────────────────────────
on_error() {
  local exit_code=$? file=$1 line=$2
  log "ERROR: command failed with exit code $exit_code at $file:$line"
}
trap 'on_error "${BASH_SOURCE[0]}" "$LINENO"' ERR

# ── PID tracking ──────────────────────────────────────────────────────────
NATS_PID=""
FRONTEND_PID=""
COMBINED_MOCKER_PID=""
declare -a PREFILL_PIDS=()
declare -a DECODE_PIDS=()

# ── Logging ───────────────────────────────────────────────────────────────
log() {
  echo "[$(date +%F\ %T) $(hostname)]: $*"
}

# ── Failure / cleanup ─────────────────────────────────────────────────────
write_failure_marker() {
  local msg="${1:-Unknown error}"
  echo "$msg" > "$result_dir/failure-marker.txt"
  log "Failure marker written: $msg"
}

cleanup() {
  log "Stopping background processes..."
  local pid
  # Kill in reverse startup order (frontend first, then workers, then NATS)
  [[ -n "$FRONTEND_PID" ]]        && kill "$FRONTEND_PID"        2>/dev/null || true
  [[ -n "$COMBINED_MOCKER_PID" ]] && kill "$COMBINED_MOCKER_PID" 2>/dev/null || true
  for pid in "${PREFILL_PIDS[@]+"${PREFILL_PIDS[@]}"}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${DECODE_PIDS[@]+"${DECODE_PIDS[@]}"}";  do kill "$pid" 2>/dev/null || true; done
  [[ -n "$NATS_PID" ]]            && kill "$NATS_PID"            2>/dev/null || true
  # Wait for all to exit cleanly
  [[ -n "$FRONTEND_PID" ]]        && wait "$FRONTEND_PID"        2>/dev/null || true
  [[ -n "$COMBINED_MOCKER_PID" ]] && wait "$COMBINED_MOCKER_PID" 2>/dev/null || true
  for pid in "${PREFILL_PIDS[@]+"${PREFILL_PIDS[@]}"}"; do wait "$pid" 2>/dev/null || true; done
  for pid in "${DECODE_PIDS[@]+"${DECODE_PIDS[@]}"}";  do wait "$pid" 2>/dev/null || true; done
  [[ -n "$NATS_PID" ]]            && wait "$NATS_PID"            2>/dev/null || true
  log "Background processes stopped."
}

on_exit() {
  local exit_code=$?
  if [[ $exit_code -ne 0 && ! -f "$result_dir/failure-marker.txt" ]]; then
    write_failure_marker "Script exited with code $exit_code"
  fi
  cleanup
}

# ── Port checking (mirrors ai_dynamo.sh _check_free_port_or_die) ──────────
_port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -lnt 2>/dev/null | awk -v p=":$port$" '$4 ~ p {found=1} END{exit !found}'
    return $?
  elif command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
    return $?
  fi
  return 1  # assume free if cannot check
}

check_port_free() {
  local name="$1" port="$2"
  log "Checking if port $port ($name) is free..."
  if _port_in_use "$port"; then
    log "ERROR: Port $port ($name) is already in use on $(hostname)"
    write_failure_marker "Port $port ($name) is already in use"
    exit 1
  fi
  log "Port $port ($name) is free"
}

# ── Arg parsing ───────────────────────────────────────────────────────────
_require_value() {
  local flag="$1" val="${2-}"
  if [[ -z "$val" || "$val" == --* ]]; then
    echo "ERROR: $flag requires a value (got: '${val:-<empty>}')" >&2
    exit 1
  fi
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --venv-python)              _require_value "$1" "${2-}"; venv_python="$2";              shift 2 ;;
      --nats-cmd)                 _require_value "$1" "${2-}"; nats_cmd="$2";                 shift 2 ;;
      --result-dir)               _require_value "$1" "${2-}"; result_dir="$2";               shift 2 ;;
      --model-path)               _require_value "$1" "${2-}"; model_path="$2";               shift 2 ;;
      --num-workers)              _require_value "$1" "${2-}"; num_workers="$2";              shift 2 ;;
      --speedup-ratio)            _require_value "$1" "${2-}"; speedup_ratio="$2";            shift 2 ;;
      --block-size)               _require_value "$1" "${2-}"; block_size="$2";               shift 2 ;;
      --num-gpu-blocks-override)  _require_value "$1" "${2-}"; num_gpu_blocks_override="$2";  shift 2 ;;
      --enable-prefix-caching)    _require_value "$1" "${2-}"; enable_prefix_caching="$2";    shift 2 ;;
      --http-port)
        _require_value "$1" "${2-}"
        if [[ ! "$2" =~ ^[0-9]+$ ]]; then
          echo "ERROR: --http-port must be numeric (got: '$2')" >&2; exit 1
        fi
        http_port="$2"; shift 2 ;;
      --router-mode)              _require_value "$1" "${2-}"; router_mode="$2";              shift 2 ;;
      --input-tokens)             _require_value "$1" "${2-}"; input_tokens="$2";             shift 2 ;;
      --output-tokens)            _require_value "$1" "${2-}"; output_tokens="$2";            shift 2 ;;
      --request-count)            _require_value "$1" "${2-}"; request_count="$2";            shift 2 ;;
      --replay-concurrency)       _require_value "$1" "${2-}"; replay_concurrency="$2";       shift 2 ;;
      --replay-mode)              _require_value "$1" "${2-}"; replay_mode="$2";              shift 2 ;;
      # Disaggregated mode
      --disaggregation-mode)      _require_value "$1" "${2-}"; disaggregation_mode="$2";    shift 2 ;;
      --prefill-num-nodes)        _require_value "$1" "${2-}"; prefill_num_nodes="$2";      shift 2 ;;
      --decode-num-nodes)         _require_value "$1" "${2-}"; decode_num_nodes="$2";       shift 2 ;;
      --prefill-cmd)              _require_value "$1" "${2-}"; prefill_cmd="$2";            shift 2 ;;
      --decode-cmd)               _require_value "$1" "${2-}"; decode_cmd="$2";             shift 2 ;;
      --kv-transfer-bandwidth)    _require_value "$1" "${2-}"; kv_transfer_bandwidth="$2";  shift 2 ;;
      --prefill-initialized-regex)   _require_value "$1" "${2-}"; prefill_initialized_regex="$2";   shift 2 ;;
      --decode-initialized-regex)    _require_value "$1" "${2-}"; decode_initialized_regex="$2";    shift 2 ;;
      --benchmark-tool)              _require_value "$1" "${2-}"; benchmark_tool="$2";              shift 2 ;;
      --engine-*)
        # Extra dynamo.mocker engine flags forwarded from [cmd_args.engine] extras.
        engine_extra_args["--${1#--engine-}"]="$2"
        shift 2 ;;
      --mocker-*)
        # Extra dynamo.mocker topology flags forwarded from [cmd_args.worker] extras.
        # "mocker" prefix avoids any ambiguity with --worker-initialized-regex.
        mocker_extra_args["--${1#--mocker-}"]="$2"
        shift 2 ;;
      --frontend-*)
        # Extra dynamo.frontend flags forwarded from [cmd_args.frontend] extras.
        frontend_extra_args["--${1#--frontend-}"]="$2"
        shift 2 ;;
      --prefill-extra-args)
        # Raw CLI string appended verbatim to prefill mocker invocations.
        prefill_extra_args_raw="$2"
        shift 2 ;;
      --prefill-args-*)
        # Named flags from [cmd_args.worker.prefill_worker.args] — applied to prefill mockers only.
        prefill_args_extra["--${1#--prefill-args-}"]="$2"
        shift 2 ;;
      --decode-extra-args)
        # Raw CLI string appended verbatim to decode mocker invocations.
        decode_extra_args_raw="$2"
        shift 2 ;;
      --decode-args-*)
        # Named flags from [cmd_args.worker.decode_worker.args] — applied to decode mockers only.
        decode_args_extra["--${1#--decode-args-}"]="$2"
        shift 2 ;;
      --aiperf-cmd)
        # Override aiperf launch command (forwarded from [cmd_args.aiperf] cmd field).
        aiperf_cmd="$2"
        shift 2 ;;
      --aiperf-extra-args)
        # Raw CLI string appended verbatim to the aiperf profile command.
        aiperf_extra_args_raw="$2"
        shift 2 ;;
      --aiperf-*)
        # Extra aiperf-specific flags forwarded from the TOML [cmd_args.aiperf] section.
        # Stored as --<flag-name> → value and appended to the aiperf profile command.
        aiperf_extra_args["--${1#--aiperf-}"]="$2"
        shift 2 ;;
      --genai-perf-cmd)
        # Override genai-perf launch command (forwarded from [cmd_args.genai_perf] cmd field).
        genai_perf_cmd="$2"
        shift 2 ;;
      --genai-perf-extra-args)
        # Raw CLI string appended verbatim to the genai-perf profile command.
        genai_perf_extra_args_raw="$2"
        shift 2 ;;
      --genai-perf-*)
        # Extra genai-perf-specific flags forwarded from the TOML [cmd_args.genai_perf] section.
        # Stored as --<flag-name> → value and appended to the genai-perf profile command.
        genai_perf_extra_args["--${1#--genai-perf-}"]="$2"
        shift 2 ;;
      *)
        echo "ERROR: Unknown argument: $1" >&2
        echo "Usage: bash dynamo_mocker.sh --result-dir <dir> --model-path <path> [options...]" >&2
        return 1
        ;;
    esac
  done
}

# ── Python resolution ────────────────────────────────────────────────────
# Called in main() after parse_args so --venv-python is available.
# Prepends venv bin to PATH so cmd fields in the TOML (e.g.
# "python3 -m dynamo.mocker", "aiperf profile", "genai-perf profile")
# resolve to venv binaries without needing full paths.
resolve_python() {
  if [[ -z "$venv_python" ]]; then
    echo "ERROR: --venv-python is required" >&2; exit 1
  fi
  if [[ ! -x "$venv_python" ]]; then
    echo "ERROR: --venv-python '$venv_python' does not exist or is not executable" >&2; exit 1
  fi
  PYTHON="$venv_python"
  export PATH="$(dirname "$venv_python"):${PATH}"
}

# ── Wait helpers ──────────────────────────────────────────────────────────
wait_for_service() {
  local url="$1"
  local max_wait="${2:-120}"
  local waited=0
  log "Waiting for service at $url (timeout: ${max_wait}s)..."
  while ! curl -sf "$url" > /dev/null 2>&1; do
    if [[ $waited -ge $max_wait ]]; then
      log "ERROR: Service at $url did not become ready after ${max_wait}s"
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
  log "Service ready at $url (waited ${waited}s)"
}

wait_for_model() {
  # Wait until /v1/models lists at least one model (a decode worker has registered).
  local base_url="$1"
  local max_wait="${2:-300}"
  local waited=0
  log "Waiting for mocker worker to register a model at $base_url/v1/models (timeout: ${max_wait}s)..."
  while true; do
    local count
    count=$(curl -sf "$base_url/v1/models" 2>/dev/null \
      | "$PYTHON" -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('data', [])))" \
      2>/dev/null || echo "0")
    if [[ "$count" -gt 0 ]]; then
      log "Mocker worker registered ($count model(s) listed, waited ${waited}s)"
      return 0
    fi
    if [[ $waited -ge $max_wait ]]; then
      log "ERROR: No mocker worker registered after ${max_wait}s"
      log "Current /v1/models: $(curl -sf "$base_url/v1/models" 2>/dev/null || echo 'no response')"
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

# Wait for worker log files to contain the initialization regex.
# Mirrors ai_dynamo.sh's wait_for_dynamo_frontend / _count_initialized_*.
wait_for_workers_initialized() {
  local role="$1"               # prefill | decode
  local expected="$2"
  local initialized_regex="$3"  # per-role init regex (replaces global worker_initialized_regex)
  local max_wait="${4:-300}"
  local waited=0
  log "Waiting for $expected $role mocker(s) to initialize (timeout: ${max_wait}s)..."
  while true; do
    local ready=0
    # || true: with set -o pipefail, grep non-zero exit (no files) would otherwise
    # cause both wc -l's "0" AND "echo 0" to land in the variable giving "0\n0".
    ready=$(grep -i -l -E "$initialized_regex" \
      "$result_dir/dynamo_${role}_"*.log 2>/dev/null | wc -l) || true
    ready="${ready//[[:space:]]/}"   # strip whitespace from wc output
    if [[ "$ready" -ge "$expected" ]]; then
      log "$role workers ready: $ready/$expected (waited ${waited}s)"
      return 0
    fi
    log "  $role initialized: $ready/$expected (waited ${waited}s)..."
    if [[ $waited -ge $max_wait ]]; then
      log "ERROR: Only $ready/$expected $role workers initialized after ${max_wait}s"
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

# Scan worker logs for fatal errors. Mirrors ai_dynamo.sh's exit_on_error.
detect_fatal_errors() {
  local error_pattern="Exception:|FATAL|RuntimeError:|Failed to connect|Address already in use|Error.*initialization"
  local -a files=()
  [[ -f "$result_dir/dynamo_mocker_combined.log" ]] && files+=("$result_dir/dynamo_mocker_combined.log")
  for f in "$result_dir"/dynamo_prefill_*.log "$result_dir"/dynamo_decode_*.log; do
    [[ -f "$f" ]] && files+=("$f")
  done
  [[ ${#files[@]} -eq 0 ]] && return 0

  local n
  n=$(grep -c -E "$error_pattern" "${files[@]}" 2>/dev/null \
    | awk -F: '{sum += $NF} END {print sum+0}')
  if [[ "$n" -gt 0 ]]; then
    log "ERROR: Detected $n fatal error line(s) in worker logs"
    grep -h -E "$error_pattern" "${files[@]}" 2>/dev/null \
      >> "$result_dir/failure-marker.txt" || true
    return 1
  fi
  return 0
}

# ── Node roles log (mirrors ai_dynamo.sh NODE_ROLES_FILE) ─────────────────
write_node_roles() {
  local roles_file="$result_dir/node_roles.log"
  {
    if [[ "$disaggregation_mode" == "none" ]]; then
      echo "$(hostname),combined_mocker"
    else
      echo "$(hostname),prefill_mocker"
      echo "$(hostname),decode_mocker"
    fi
    echo "$(hostname),frontend"
    echo "$(hostname),${benchmark_tool}"
  } > "$roles_file"
  log "Node roles → $roles_file"
}

# ── NATS (mirrors ai_dynamo.sh launch_nats) ───────────────────────────────
launch_nats() {
  # First token of nats_cmd is the binary (e.g. "nats-server" or "/full/path/nats-server")
  local nats_bin="${nats_cmd%% *}"
  if ! command -v "$nats_bin" > /dev/null 2>&1 && [[ ! -x "$nats_bin" ]]; then
    write_failure_marker "nats-server not found: '$nats_bin'. Set nats_cmd in [cmd_args] to the full path."
    exit 1
  fi
  check_port_free "nats"             4222
  check_port_free "nats-monitoring"  8222

  log "Starting nats-server (cmd=$nats_cmd)..."
  # shellcheck disable=SC2086
  $nats_cmd -p 4222 -m 8222 > "$result_dir/nats.log" 2>&1 &
  NATS_PID=$!
  log "nats-server started (PID=$NATS_PID)"
  wait_for_service "http://localhost:8222/healthz" 30
  log "NATS ready"
}

# ── Extra-flags expansion helper ──────────────────────────────────────────
# Usage: _build_extra_flags <src-assoc-array-name> <dst-indexed-array-name>
# Expands an associative array of --flag → value pairs into a flat indexed
# array suitable for splicing into a command invocation.  Bool convention:
#   "true"  → bare flag (no value argument)
#   "false" → omitted entirely
#   other   → --flag value
# Requires bash 4.3+ (local -n nameref).
_build_extra_flags() {
  local -n _src="$1"
  local -n _dst="$2"
  local key val
  for key in "${!_src[@]}"; do
    val="${_src[$key]}"
    if   [[ "$val" == "true"  ]]; then _dst+=("$key")
    elif [[ "$val" != "false" ]]; then _dst+=("$key" "$val")
    fi
  done
}

# ── Frontend (mirrors ai_dynamo.sh launch_ingress) ────────────────────────
launch_frontend() {
  # Convert router_mode underscores to dashes (round_robin → round-robin)
  local frontend_router_mode="${router_mode//_/-}"
  # kv_router → kv  (frontend uses short form)
  [[ "$frontend_router_mode" == "kv-router" ]] && frontend_router_mode="kv"

  local extra_flags=()
  _build_extra_flags frontend_extra_args extra_flags

  # In kv router mode, mocker workers publish KV events via ZMQ (file-based
  # discovery backend), not via NATS.  ai-dynamo ≥1.2 changed kv router's
  # EventSubscriber to require NATS, which breaks chat-endpoint registration
  # when using file discovery.  --no-router-kv-events makes the router predict
  # cache state from routing decisions instead of subscribing to worker events,
  # which is the correct behaviour for a GPU-free mocker workload anyway.
  local kv_events_flag=()
  [[ "$frontend_router_mode" == "kv" ]] && kv_events_flag=("--no-router-kv-events")

  check_port_free "dynamo.frontend" "$http_port"
  log "Starting dynamo.frontend (port=$http_port, router=$frontend_router_mode)..."
  "$PYTHON" -m dynamo.frontend \
    --http-port "$http_port" \
    --router-mode "$frontend_router_mode" \
    --discovery-backend file \
    --request-plane tcp \
    "${kv_events_flag[@]+"${kv_events_flag[@]}"}" \
    "${extra_flags[@]+"${extra_flags[@]}"}" \
    > "$result_dir/dynamo_frontend.log" 2>&1 &
  FRONTEND_PID=$!
  log "dynamo.frontend started (PID=$FRONTEND_PID)"

  if ! wait_for_service "http://localhost:${http_port}/v1/models"; then
    write_failure_marker "dynamo.frontend did not become ready on port $http_port"
    exit 1
  fi
}

# ── Combined mocker (non-disaggregated) ───────────────────────────────────
launch_combined_mocker() {
  local prefix_caching_flag="--enable-prefix-caching"
  [[ "$enable_prefix_caching" == "false" ]] && prefix_caching_flag="--no-enable-prefix-caching"

  local extra_flags=()
  _build_extra_flags engine_extra_args extra_flags
  _build_extra_flags mocker_extra_args extra_flags

  log "Starting dynamo.mocker combined (num_workers=$num_workers, speedup=$speedup_ratio)..."
  # shellcheck disable=SC2086
  "$PYTHON" -m dynamo.mocker \
    --model-path "$model_path" \
    --num-workers "$num_workers" \
    --speedup-ratio "$speedup_ratio" \
    --block-size "$block_size" \
    --num-gpu-blocks-override "$num_gpu_blocks_override" \
    $prefix_caching_flag \
    --discovery-backend file \
    --request-plane tcp \
    "${extra_flags[@]+"${extra_flags[@]}"}" \
    > "$result_dir/dynamo_mocker_combined.log" 2>&1 &
  COMBINED_MOCKER_PID=$!
  log "dynamo.mocker (combined) started (PID=$COMBINED_MOCKER_PID)"
}

# ── Disaggregated: prefill mockers (mirrors ai_dynamo.sh launch_prefill) ──
launch_prefill_mockers() {
  local prefix_caching_flag="--enable-prefix-caching"
  [[ "$enable_prefix_caching" == "false" ]] && prefix_caching_flag="--no-enable-prefix-caching"

  local extra_flags=()
  _build_extra_flags engine_extra_args  extra_flags
  _build_extra_flags mocker_extra_args  extra_flags
  _build_extra_flags prefill_args_extra extra_flags
  # Append raw extra_args string if provided (word-split intentional for raw CLI string)
  # shellcheck disable=SC2206
  [[ -n "$prefill_extra_args_raw" ]] && extra_flags+=($prefill_extra_args_raw)

  # Split cmd string into array for safe word-splitting (no eval needed)
  local -a _prefill_cmd
  read -ra _prefill_cmd <<< "$prefill_cmd"

  log "Launching $prefill_num_nodes prefill mocker(s) via: $prefill_cmd"
  for i in $(seq 0 $(( prefill_num_nodes - 1 ))); do
    local log_file="$result_dir/dynamo_prefill_${i}.log"
    log "  prefill mocker $i → $log_file"
    # --disaggregation-mode prefill is owned by the cmd (set in TOML prefill_worker.cmd).
    # shellcheck disable=SC2086
    "${_prefill_cmd[@]}" \
      --model-path "$model_path" \
      --num-workers 1 \
      --speedup-ratio "$speedup_ratio" \
      --block-size "$block_size" \
      --num-gpu-blocks-override "$num_gpu_blocks_override" \
      $prefix_caching_flag \
      --kv-transfer-bandwidth "$kv_transfer_bandwidth" \
      --discovery-backend file \
      --request-plane tcp \
      "${extra_flags[@]+"${extra_flags[@]}"}" \
      > "$log_file" 2>&1 &
    PREFILL_PIDS+=($!)
    log "  prefill mocker $i started (PID=${PREFILL_PIDS[-1]})"
  done
}

# ── Disaggregated: decode mockers (mirrors ai_dynamo.sh launch_decode) ────
launch_decode_mockers() {
  local prefix_caching_flag="--enable-prefix-caching"
  [[ "$enable_prefix_caching" == "false" ]] && prefix_caching_flag="--no-enable-prefix-caching"

  local extra_flags=()
  _build_extra_flags engine_extra_args extra_flags
  _build_extra_flags mocker_extra_args extra_flags
  _build_extra_flags decode_args_extra extra_flags
  # Append raw extra_args string if provided (word-split intentional for raw CLI string)
  # shellcheck disable=SC2206
  [[ -n "$decode_extra_args_raw" ]] && extra_flags+=($decode_extra_args_raw)

  # Split cmd string into array for safe word-splitting (no eval needed)
  local -a _decode_cmd
  read -ra _decode_cmd <<< "$decode_cmd"

  log "Launching $decode_num_nodes decode mocker(s) via: $decode_cmd"
  for i in $(seq 0 $(( decode_num_nodes - 1 ))); do
    local log_file="$result_dir/dynamo_decode_${i}.log"
    log "  decode mocker $i → $log_file"
    # --disaggregation-mode decode is owned by the cmd (set in TOML decode_worker.cmd).
    # shellcheck disable=SC2086
    "${_decode_cmd[@]}" \
      --model-path "$model_path" \
      --num-workers 1 \
      --speedup-ratio "$speedup_ratio" \
      --block-size "$block_size" \
      --num-gpu-blocks-override "$num_gpu_blocks_override" \
      $prefix_caching_flag \
      --kv-transfer-bandwidth "$kv_transfer_bandwidth" \
      --discovery-backend file \
      --request-plane tcp \
      "${extra_flags[@]+"${extra_flags[@]}"}" \
      > "$log_file" 2>&1 &
    DECODE_PIDS+=($!)
    log "  decode mocker $i started (PID=${DECODE_PIDS[-1]})"
  done
}

# ── genai_perf (mirrors ai_dynamo.sh launch_workload / genai_perf.sh) ─────
resolve_genai_perf_sh() {
  local local_copy="$SCRIPT_DIR/genai_perf.sh"
  if [[ -f "$local_copy" ]]; then echo "$local_copy"; return; fi
  "$PYTHON" - <<'EOF'
import sys
try:
    from pathlib import Path
    import cloudai.workloads.ai_dynamo as m
    p = Path(m.__file__).parent / "genai_perf.sh"
    if p.exists():
        print(p)
        sys.exit(0)
except Exception:
    pass
sys.exit(1)
EOF
}

launch_genai_perf() {
  local GENAI_PERF_SH
  if ! GENAI_PERF_SH="$(resolve_genai_perf_sh)"; then
    write_failure_marker "genai_perf.sh not found — install cloudai or place genai_perf.sh alongside dynamo_mocker.sh"
    exit 1
  fi
  log "Using genai_perf.sh at: $GENAI_PERF_SH"

  # replay_mode: offline → --concurrency (throughput test)
  #              online  → --request-rate (latency-under-load test)
  local load_flag
  if [[ "$replay_mode" == "online" ]]; then
    load_flag="--request-rate $replay_concurrency"
  else
    load_flag="--concurrency $replay_concurrency"
  fi

  # Collect extra genai-perf flags from --genai-perf-* passthrough args.
  # Bool extras: "true" → bare flag, "false" → omitted.
  local extra_flags=()
  for key in "${!genai_perf_extra_args[@]}"; do
    local val="${genai_perf_extra_args[$key]}"
    if [[ "$val" == "true" ]]; then
      extra_flags+=("$key")
    elif [[ "$val" != "false" ]]; then
      extra_flags+=("$key" "$val")
    fi
  done
  # Append raw extra_args string if provided (word-split intentional for raw CLI string)
  # shellcheck disable=SC2206
  [[ -n "$genai_perf_extra_args_raw" ]] && extra_flags+=($genai_perf_extra_args_raw)

  # Build context args array; only pass --cmd when explicitly overridden.
  local context_args=(--result-dir "$result_dir" --model "$model_path" --port "$http_port")
  [[ -n "$genai_perf_cmd" ]] && context_args+=(--cmd "$genai_perf_cmd")

  # shellcheck disable=SC2086
  bash "$GENAI_PERF_SH" \
    "${context_args[@]}" \
    -- \
    $load_flag \
    --request-count "$request_count" \
    --synthetic-input-tokens-mean "$input_tokens" \
    --output-tokens-mean "$output_tokens" \
    --endpoint-type chat \
    "${extra_flags[@]}"

  log "genai_perf.sh completed"
}

# ── aiperf (alternative to genai-perf, mirrors launch_genai_perf) ────────
resolve_aiperf_sh() {
  local local_copy="$SCRIPT_DIR/aiperf.sh"
  if [[ -f "$local_copy" ]]; then echo "$local_copy"; return; fi
  return 1
}

launch_aiperf() {
  local AIPERF_SH
  if ! AIPERF_SH="$(resolve_aiperf_sh)"; then
    write_failure_marker "aiperf.sh not found — place aiperf.sh alongside dynamo_mocker.sh"
    exit 1
  fi
  log "Using aiperf.sh at: $AIPERF_SH"

  local load_flag
  if [[ "$replay_mode" == "online" ]]; then
    load_flag="--request-rate $replay_concurrency"
  else
    load_flag="--concurrency $replay_concurrency"
  fi

  # Collect extra aiperf flags from --aiperf-* passthrough args.
  # Bool extras: "true" → bare flag, "false" → omitted.
  local extra_flags=()
  for key in "${!aiperf_extra_args[@]}"; do
    local val="${aiperf_extra_args[$key]}"
    if [[ "$val" == "true" ]]; then
      extra_flags+=("$key")
    elif [[ "$val" != "false" ]]; then
      extra_flags+=("$key" "$val")
    fi
  done
  # Append raw extra_args string if provided (word-split intentional for raw CLI string)
  # shellcheck disable=SC2206
  [[ -n "$aiperf_extra_args_raw" ]] && extra_flags+=($aiperf_extra_args_raw)

  # Build context args array; only pass --cmd when explicitly overridden.
  local context_args=(--result-dir "$result_dir" --model "$model_path" --port "$http_port")
  [[ -n "$aiperf_cmd" ]] && context_args+=(--cmd "$aiperf_cmd")

  # shellcheck disable=SC2086
  bash "$AIPERF_SH" \
    "${context_args[@]}" \
    -- \
    --synthetic-input-tokens-mean "$input_tokens" \
    --output-tokens-mean          "$output_tokens" \
    --request-count               "$request_count" \
    $load_flag \
    "${extra_flags[@]}"

  log "aiperf.sh completed"
}

# ── Main ──────────────────────────────────────────────────────────────────
main() {
  parse_args "$@" || exit 1
  resolve_python
  # Set defaults that depend on $PYTHON (after venv is resolved).
  # User-provided --prefill-cmd / --decode-cmd take precedence.
  [[ -z "$prefill_cmd" ]] && prefill_cmd="$PYTHON -m dynamo.mocker --disaggregation-mode prefill"
  [[ -z "$decode_cmd" ]]  && decode_cmd="$PYTHON -m dynamo.mocker --disaggregation-mode decode"

  if [[ -z "$result_dir" ]]; then
    echo "ERROR: --result-dir is required" >&2
    exit 1
  fi

  mkdir -p "$result_dir"
  trap on_exit EXIT

  log "disaggregation_mode=$disaggregation_mode  router_mode=$router_mode  replay_mode=$replay_mode"
  write_node_roles

  # ── 1. NATS — always required (DistributedRuntime event plane) ──────────
  launch_nats

  if [[ "$disaggregation_mode" == "none" ]]; then
    # ── Combined mode: one mocker process, combined prefill+decode ─────────
    log "=== Combined mode: num_workers=$num_workers ==="
    launch_combined_mocker
    launch_frontend

    if ! wait_for_model "http://localhost:${http_port}"; then
      write_failure_marker "dynamo.mocker did not register after 300s — check dynamo_mocker_combined.log"
      exit 1
    fi

  elif [[ "$disaggregation_mode" == "prefill_decode" ]]; then
    # ── Disaggregated mode: separate prefill + decode mocker processes ─────
    # Mirrors ai_dynamo.sh: launch workers first, wait for init, then frontend.
    log "=== Disaggregated mode: prefill=$prefill_num_nodes, decode=$decode_num_nodes ==="
    launch_prefill_mockers
    launch_decode_mockers

    # Wait for prefill workers (log-based — prefill doesn't expose HTTP)
    if ! wait_for_workers_initialized "prefill" "$prefill_num_nodes" "$prefill_initialized_regex" 300; then
      detect_fatal_errors || true
      write_failure_marker "Prefill mocker(s) did not initialize — check dynamo_prefill_*.log"
      exit 1
    fi

    # Wait for decode workers (log-based)
    if ! wait_for_workers_initialized "decode" "$decode_num_nodes" "$decode_initialized_regex" 300; then
      detect_fatal_errors || true
      write_failure_marker "Decode mocker(s) did not initialize — check dynamo_decode_*.log"
      exit 1
    fi

    launch_frontend

    # Confirm at least one decode worker registered with the frontend
    if ! wait_for_model "http://localhost:${http_port}"; then
      write_failure_marker "No decode workers registered with frontend after 300s"
      exit 1
    fi

  else
    write_failure_marker "Unknown disaggregation_mode='$disaggregation_mode' (must be 'none' or 'prefill_decode')"
    exit 1
  fi

  # ── 2. Check for startup errors before running the benchmark ────────────
  if ! detect_fatal_errors; then
    write_failure_marker "Fatal errors detected in worker logs before benchmark — aborting"
    exit 1
  fi

  # ── 3. Run benchmark ─────────────────────────────────────────────────────
  if [[ "$benchmark_tool" == "aiperf" ]]; then
    launch_aiperf
  elif [[ "$benchmark_tool" == "genai_perf" ]]; then
    launch_genai_perf
  else
    write_failure_marker "Unknown benchmark_tool='$benchmark_tool' (must be 'aiperf' or 'genai_perf')"
    exit 1
  fi

  if [[ ! -f "$result_dir/benchmark_report.csv" ]]; then
    write_failure_marker "benchmark_report.csv not found in $result_dir — benchmark may not have completed"
    exit 1
  fi
  echo "SUCCESS" > "$result_dir/success-marker.txt"
  log "Run complete — SUCCESS"
  trap cleanup EXIT
}

main "$@"
