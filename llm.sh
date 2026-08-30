#!/usr/bin/env bash
# start/stop/status local LLM chain:
#   kompact (LAN :8000, context optimizer) -> toolup proxy 127.0.0.1:8001 -> llama-server 127.0.0.1:8002
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${LLM_PID_FILE:-$HERE/.server.pid}"
LOG="${LLM_LOG:-$HERE/server.log}"
KLOG="${LLM_KLOG:-$HERE/kompact.log}"
PORT="${LLM_PORT:-8000}"   # public (kompact)
TPORT="${LLM_TPORT:-8001}" # toolup proxy, loopback
BIND="${LLM_BIND:-0.0.0.0}"
CTX="${LLM_CTX:-16384}"
KCOMPACT="${KOMPACT:-$HOME/kompact/bin/kompact}"
KOMPACT_OPTS="${LLM_KOMPACT_OPTS:---no-otel --disable observation_masker --disable cache_aligner}"

running() { [ -f "$PID_FILE" ] && kill -0 "$(sed -n 1p "$PID_FILE")" 2>/dev/null; }
precv() { curl -fsS -m 3 "http://$1:$2/v1/models" >/dev/null 2>&1; }
kready() { curl -fsS -m 3 "http://$1:$2/health" >/dev/null 2>&1; }

case "${1:-}" in
  start)
    if running; then echo "llm already running (pid $(sed -n 1p "$PID_FILE")) -> http://$BIND:$PORT/v1"; exit 0; fi
    setsid "$HERE/manage_deepseek.py" serve --host 127.0.0.1 --port "$TPORT" --ctx "$CTX" >"$LOG" 2>&1 &
    pid=$!
    echo "$pid" >"$PID_FILE"
    for _ in $(seq 1 180); do
      precv 127.0.0.1 "$TPORT" && break
      sleep 1
    done
    precv 127.0.0.1 "$TPORT" || { echo "toolup proxy not ready; check $LOG"; exit 1; }
    setsid "$KCOMPACT" proxy --host "$BIND" --port "$PORT" \
      --openai-base-url "http://127.0.0.1:$TPORT" $KOMPACT_OPTS >"$KLOG" 2>&1 &
    echo $! >>"$PID_FILE"
    for _ in $(seq 1 30); do
      kready "$BIND" "$PORT" && { echo "llm running -> http://$BIND:$PORT/v1 (kompact, $KCOMPACT)"; exit 0; }
      sleep 1
    done
    echo "kompact not ready; check $KLOG"; exit 1
    ;;
  stop)
    p1="$(sed -n 1p "$PID_FILE" 2>/dev/null)"
    p2="$(sed -n 2p "$PID_FILE" 2>/dev/null)"
    for p in "$p1" "$p2"; do
      [ -n "$p" ] && { kill -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null; }
    done
    [ -n "$p1" ] && for _ in $(seq 1 30); do
      kill -0 "$p1" 2>/dev/null || break
      sleep 1
    done
    rm -f "$PID_FILE"
    echo "llm stopped"
    ;;
  status)
    if running; then echo "running (pid $(sed -n 1p "$PID_FILE")) -> http://$BIND:$PORT/v1"; else echo "not running"; fi
    ;;
  *) echo "usage: $0 start|stop|status"; exit 1 ;;
esac