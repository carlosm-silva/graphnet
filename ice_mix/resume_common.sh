#!/bin/bash

find_wandb_run_id() {
    local run_dir="$1"
    local wandb_meta wandb_dir wandb_run_id base

    wandb_meta="$(
        find "$run_dir" -type f -name 'wandb-metadata.json' 2>/dev/null | sort | tail -n 1
    )"
    if [ -n "$wandb_meta" ]; then
        wandb_run_id="$(
            python - "$wandb_meta" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
except Exception:
    raise SystemExit(0)

for key in ("id", "run_id", "runId"):
    value = data.get(key)
    if value:
        print(value)
        raise SystemExit(0)
raise SystemExit(0)
PY
        )"
        if [ -n "$wandb_run_id" ]; then
            printf '%s\n' "$wandb_run_id"
            return 0
        fi
    fi

    wandb_dir="$(
        find "$run_dir" -type d \( -name 'run-*' -o -name 'offline-run-*' \) 2>/dev/null | sort | tail -n 1
    )"
    if [ -n "$wandb_dir" ]; then
        base="$(basename "$wandb_dir")"
        printf '%s\n' "${base##*-}"
        return 0
    fi

    return 1
}

configure_resume_from_latest_run() {
    local project_name="$1"
    local label="$2"
    local latest_ckpt run_dir run_name wandb_run_id

    latest_ckpt="$(
        ls -1dt ice_mix/outputs/${project_name}_*/checkpoints/last.ckpt 2>/dev/null | head -n 1 || true
    )"
    if [ -z "$latest_ckpt" ]; then
        echo "No previous checkpoint found for ${project_name}; starting from scratch."
        return 0
    fi

    run_dir="$(dirname "$(dirname "$latest_ckpt")")"
    run_name="$(basename "$run_dir")"
    wandb_run_id="$(find_wandb_run_id "$run_dir" || true)"

    echo "Resuming ${label} from checkpoint: $latest_ckpt"
    EXTRA_ARGS+=( "ckpt_path=$latest_ckpt" )
    EXTRA_ARGS+=( "run_name=$run_name" )

    if [ -n "$wandb_run_id" ]; then
        echo "Using WandB run id from previous run: $wandb_run_id"
        export WANDB_RUN_ID="$wandb_run_id"
    else
        echo "No WandB metadata found; falling back to run directory name: $run_name"
        export WANDB_RUN_ID="$run_name"
    fi
    export WANDB_RESUME="allow"
}
