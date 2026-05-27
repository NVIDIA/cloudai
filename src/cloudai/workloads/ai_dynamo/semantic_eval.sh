#!/bin/bash

result_dir=""
cmd=""

log() {
  echo "[$(date +%F\ %T) $(hostname)]: $*"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --result-dir)
        result_dir="$2"; shift 2 ;;
      --cmd)
        cmd="$2"; shift 2 ;;
      --)
        shift
        break
        ;;
      --*)
        if [[ $# -ge 2 ]]; then
          shift 2
        else
          shift
        fi
        ;;
      *)
        shift ;;
    esac
  done
}

main() {
  parse_args "$@"

  if [[ -z "$result_dir" ]]; then
    log "ERROR: --result-dir is required"
    exit 1
  fi
  if [[ -z "$cmd" ]]; then
    log "ERROR: --cmd is required"
    exit 1
  fi
  mkdir -p "$result_dir"

  log "Launching semantic eval: $cmd"
  eval "$cmd"
}

main "$@"
