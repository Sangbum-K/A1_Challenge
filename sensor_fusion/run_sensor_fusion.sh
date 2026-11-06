#!/usr/bin/env bash
set -eo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-start}" in
  lidar-check|topic-audit)
    mode="$1"; shift
    exec bash "$PROJECT/run.sh" "$mode" "$@"
    ;;
  stop|status)
    mode="$1"
    pid="$(cat "$PROJECT/outputs/ui.pid" 2>/dev/null || true)"
    if [[ ! "$pid" =~ ^[0-9]+$ ]] || [[ ! -r "/proc/$pid/cmdline" ]] ||
       ! tr '\0' '\n' <"/proc/$pid/cmdline" | grep -Fxq "$PROJECT/run_ui.sh"; then
      echo "UI is not running."; exit 0
    fi
    if [[ "$mode" == stop ]]; then kill -TERM "$pid"; echo "UI shutdown requested."
    else echo "UI running (PID $pid)."; cat "$PROJECT/outputs/ui_latest_path.txt"; fi
    ;;
  *)
    if [[ "${1:-}" == start ]]; then shift; fi
    exec bash "$PROJECT/run_ui.sh" "$@"
    ;;
esac
