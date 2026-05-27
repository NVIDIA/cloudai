#!/bin/bash

result_dir=""
model=""
url="http://localhost"
port="8000"
module=""
eval_args=""
extra_args=""
log_file="semantic_eval.log"
passthrough_args=()

log() {
  echo "[$(date +%F\ %T) $(hostname)]: $*"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model)
        model="$2"; shift 2 ;;
      --url)
        url="${2%/}"; shift 2 ;;
      --port)
        port="$2"; shift 2 ;;
      --result-dir)
        result_dir="$2"; shift 2 ;;
      --module)
        module="$2"; shift 2 ;;
      --args)
        eval_args="$2"; shift 2 ;;
      --extra-args)
        extra_args="$2"; shift 2 ;;
      --log-file)
        log_file="$2"; shift 2 ;;
      --)
        shift
        passthrough_args=("$@")
        break
        ;;
      --*)
        shift 2 ;;
      *)
        shift ;;
    esac
  done
}

expand_args() {
  local raw_args="$1"
  raw_args="${raw_args//\{model\}/$model}"
  raw_args="${raw_args//\{host\}/$url}"
  raw_args="${raw_args//\{port\}/$port}"
  raw_args="${raw_args//\{output_path\}/$result_dir}"
  echo "$raw_args"
}

main() {
  parse_args "$@"

  if [[ -z "$result_dir" ]]; then
    log "ERROR: --result-dir is required"
    exit 1
  fi
  if [[ -z "$module" ]]; then
    log "ERROR: --module is required"
    exit 1
  fi
  mkdir -p "$result_dir"

  local expanded_args
  expanded_args="$(expand_args "$eval_args")"

  local command
  if [[ "$module" == *.py || "$module" == */* ]]; then
    command="python3 \"$module\""
  else
    command="python3 -m \"$module\""
  fi
  command="$command $expanded_args"

  if [[ -n "$extra_args" ]]; then
    command="$command $extra_args"
  fi
  if [[ ${#passthrough_args[@]} -gt 0 ]]; then
    command="$command ${passthrough_args[*]}"
  fi

  local output_log="$result_dir/$log_file"
  log "Launching semantic eval: $command"
  eval "$command" > "$output_log" 2>&1
}

main "$@"
