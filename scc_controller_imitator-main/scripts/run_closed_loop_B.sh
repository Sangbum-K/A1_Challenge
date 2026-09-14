#!/usr/bin/env bash
# v25h-B closed-loop wrapper: Professor main (steer 정통) + Side SCC 4L (ax 보강).
# MODEL_VARIANT=B 만 export 하고 본 launcher 로 위임.
#
# 사용:
#   bash scripts/run_closed_loop_B.sh            # debug viz off
#   bash scripts/run_closed_loop_B.sh --viz      # debug viz on
#   bash scripts/run_closed_loop_B.sh --viz-all  # seg + a_2 overlay
set -e
export MODEL_VARIANT=B
exec bash "$(dirname "$0")/run_closed_loop.sh" "$@"
