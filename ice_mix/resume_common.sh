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

configure_fine_tune_from_latest_run() {
    local source_project_name="$1"
    local label="$2"
    local latest_ckpt run_dir run_name

    latest_ckpt="$(
        ls -1dt ice_mix/outputs/${source_project_name}_*/checkpoints/last.ckpt 2>/dev/null | head -n 1 || true
    )"
    if [ -z "$latest_ckpt" ]; then
        echo "[FATAL] No previous checkpoint found for ${source_project_name}; cannot fine-tune."
        return 44
    fi

    run_dir="$(dirname "$(dirname "$latest_ckpt")")"
    run_name="$(basename "$run_dir")"

    echo "Fine-tuning ${label} from checkpoint weights: $latest_ckpt"
    echo "Source run: $run_name"
    EXTRA_ARGS+=( "fine_tune_from_ckpt=$latest_ckpt" )
}

configure_fine_tune_resume_from_latest_run() {
    local project_name="$1"
    local label="$2"
    local latest_ckpt run_dir run_name wandb_run_id

    latest_ckpt="$(
        ls -1dt ice_mix/outputs/${project_name}_*/checkpoints/last.ckpt 2>/dev/null | head -n 1 || true
    )"
    if [ -z "$latest_ckpt" ]; then
        echo "[FATAL] No fine-tune checkpoint found for ${project_name}; cannot resume."
        return 44
    fi

    run_dir="$(dirname "$(dirname "$latest_ckpt")")"
    run_name="$(basename "$run_dir")"
    wandb_run_id="$(find_wandb_run_id "$run_dir" || true)"

    echo "Resuming ${label} from checkpoint: $latest_ckpt"
    EXTRA_ARGS+=( "ckpt_path=$latest_ckpt" )
    EXTRA_ARGS+=( "run_name=$run_name" )

    if [ -n "$wandb_run_id" ]; then
        echo "Resuming WandB run: $wandb_run_id"
        export WANDB_RUN_ID="$wandb_run_id"
    else
        echo "[FATAL] No WandB run id found in ${run_dir}; refusing to create a duplicate run."
        return 45
    fi
    export WANDB_RESUME="must"
}

load_ice_mix_env() {
    if [ -f ice_mix/.env ]; then
        set -a
        # shellcheck disable=SC1091
        source <(grep -v '^#' ice_mix/.env)
        set +a
    fi
}

copy_standard_data_to_local_tmp() {
    echo "Copying database files to local NVMe storage..."

    if [ -z "${TMPDIR:-}" ] || [ ! -d "$TMPDIR" ] || [ ! -w "$TMPDIR" ]; then
        echo "[FATAL] Slurm TMPDIR is missing or not writable: ${TMPDIR:-<unset>}"
        return 46
    fi

    if [ -z "${DATA_ROOT:-}" ]; then
        echo "[FATAL] DATA_ROOT not set; export it in ice_mix/.env before submitting"
        return 43
    fi

    local db_file source_path destination_path copy_pid
    local copy_pids=""
    for db_file in \
        "my_numu_database_part_1 (1).db" \
        "my_nue_database_part_1 (1).db" \
        "my_nutau_database_part_1 (1).db"
    do
        source_path="$DATA_ROOT/$db_file"
        if [ ! -r "$source_path" ] || [ ! -s "$source_path" ]; then
            echo "[FATAL] Required database is missing, unreadable, or empty: $source_path"
            return 47
        fi
    done

    for db_file in \
        "my_numu_database_part_1 (1).db" \
        "my_nue_database_part_1 (1).db" \
        "my_nutau_database_part_1 (1).db"
    do
        source_path="$DATA_ROOT/$db_file"
        echo "Copying $source_path..."
        cp "$source_path" "${TMPDIR}/" &
        copy_pid=$!
        copy_pids="$copy_pids $copy_pid"
    done

    for copy_pid in $copy_pids; do
        if ! wait "$copy_pid"; then
            echo "[FATAL] A database copy to $TMPDIR failed."
            return 48
        fi
    done

    for db_file in \
        "my_numu_database_part_1 (1).db" \
        "my_nue_database_part_1 (1).db" \
        "my_nutau_database_part_1 (1).db"
    do
        destination_path="$TMPDIR/$db_file"
        if [ ! -r "$destination_path" ] || [ ! -s "$destination_path" ]; then
            echo "[FATAL] Staged database is missing, unreadable, or empty: $destination_path"
            return 49
        fi
        ls -lh "$destination_path"
    done

    echo "Data copy completed. Using local data at ${TMPDIR}"
    export LOCAL_DATA_DIR="${TMPDIR}"
}

configure_healthy_gpus() {
    GOOD=$(
python - <<'PY'
import subprocess, torch, re
good=[]
n=torch.cuda.device_count()
for i in range(n):
    try:
        out=subprocess.check_output(["nvidia-smi","-i",str(i),"-q","-d","ECC"], text=True, stderr=subprocess.DEVNULL)
        m=re.search(r"Volatile Uncorr\. ECC.*?:\s+(\d+)", out)
        if m and m.group(1) not in ("0","N/A"):
            continue
    except Exception:
        pass
    try:
        torch.cuda.set_device(i)
        a=torch.randn(512,512, device=f"cuda:{i}")
        b=torch.randn(512,512, device=f"cuda:{i}")
        (a@b).sum().item(); torch.cuda.synchronize(i)
        good.append(str(i))
    except Exception:
        pass
print(",".join(good))
PY
    )
    if [ -z "$GOOD" ]; then echo "[FATAL] no healthy GPUs"; return 42; fi
    export CUDA_VISIBLE_DEVICES="${GOOD}"
    NPROC=$(echo "$GOOD" | awk -F',' '{print NF}')
    export NPROC
    echo "Using GPUs (visible): ${CUDA_VISIBLE_DEVICES}"
    echo "Launching torchrun with nproc_per_node=${NPROC}"
}

configure_training_environment() {
    local wandb_root
    wandb_root="${ICE_MIX_WANDB_ROOT:-$PWD/ice_mix/outputs/wandb_runtime}"
    mkdir -p "$wandb_root/cache" "$wandb_root/runs" "$wandb_root/data" || return 50
    export WANDB_CACHE_DIR="$wandb_root/cache"
    export WANDB_DIR="$wandb_root/runs"
    export WANDB_DATA_DIR="$wandb_root/data"

    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export CUDA_LAUNCH_BLOCKING=0
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export TORCH_CUDNN_V8_API_ENABLED=1
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    export CUDA_DEVICE_MAX_CONNECTIONS=32
}
