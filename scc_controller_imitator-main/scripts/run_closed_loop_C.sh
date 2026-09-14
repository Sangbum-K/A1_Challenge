#!/usr/bin/env bash
# v25c2 closed-loop: f_1-only (main_no_image, side/image 없음). Professor #654 → v0301 단독 finetune.
# steer/waypoint tracking 검증용. image/a_2 미사용.
set -e
export MODEL_VARIANT=C
exec bash "$(dirname "$0")/run_closed_loop.sh" "$@"
