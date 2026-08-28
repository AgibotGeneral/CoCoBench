#!/bin/bash
# Unified CoCoBench VLM evaluation runner.
#
# Examples:
#   bash run_eval.sh --model <model-id> --conditions all
#   bash run_eval.sh --model <model-id> --conditions C0_image,D0_image,D1_image
#   bash run_eval.sh --model <model-id> --conditions image --eval-set rep240
#   bash run_eval.sh --model <model-id> --conditions C0_blind --jobs 8
#
# Condition shortcuts:
#   all       = C0_image,C0_blind,D0_image,D0_blind,D1_image,D1_blind
#   image     = C0_image,D0_image,D1_image
#   blind     = C0_blind,D0_blind,D1_blind
#   C0_image, C0_blind, D0_image, D0_blind, D1_image, D1_blind (individual)

set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python}"

MODELS=""
CONDITIONS=""
EVAL_SET=""
JOBS=16
CONFIG="config.yaml"

usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|--models)   MODELS="$2"; shift 2 ;;
        --conditions|-c)    CONDITIONS="$2"; shift 2 ;;
        --eval-set)         EVAL_SET="$2"; shift 2 ;;
        --jobs|-j)          JOBS="$2"; shift 2 ;;
        --config)           CONFIG="$2"; shift 2 ;;
        -h|--help)          usage ;;
        *)                  echo "Unknown option: $1"; usage ;;
    esac
done

[[ -z "$MODELS" ]] && { echo "Error: --model required"; usage; }
[[ -z "$CONDITIONS" ]] && { echo "Error: --conditions required"; usage; }

# Expand condition shortcuts
expand_conditions() {
    local input="$1"
    case "$input" in
        all)   echo "C0_image,C0_blind,D0_image,D0_blind,D1_image,D1_blind" ;;
        image) echo "C0_image,D0_image,D1_image" ;;
        blind) echo "C0_blind,D0_blind,D1_blind" ;;
        *)     echo "$input" ;;
    esac
}

# Map condition name to (policy, comm, obs_mode)
condition_params() {
    case "$1" in
        C0_image) echo "centralized none image" ;;
        C0_blind) echo "centralized none blind" ;;
        D0_image) echo "distributed none image" ;;
        D0_blind) echo "distributed none blind" ;;
        D1_image) echo "distributed broadcast image" ;;
        D1_blind) echo "distributed broadcast blind" ;;
        *) echo "Unknown condition: $1" >&2; exit 1 ;;
    esac
}

CONDITIONS=$(expand_conditions "$CONDITIONS")

IFS=',' read -ra MODEL_LIST <<< "$MODELS"
IFS=',' read -ra COND_LIST <<< "$CONDITIONS"

TOTAL=$(( ${#MODEL_LIST[@]} * ${#COND_LIST[@]} ))
COUNT=0

for MODEL in "${MODEL_LIST[@]}"; do
    for COND in "${COND_LIST[@]}"; do
        COUNT=$((COUNT + 1))
        read -r policy comm obs_mode <<< "$(condition_params "$COND")"
        EXP_NAME="${MODEL}_${COND}"

        echo "======================================================================"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ($COUNT/$TOTAL) $EXP_NAME"
        echo "  model=$MODEL policy=$policy comm=$comm obs=$obs_mode jobs=$JOBS"
        echo "======================================================================"

        eval_args=(
            --config "$CONFIG"
            --model-name "$MODEL"
            --policy "$policy"
            --comm "$comm"
            --obs-mode "$obs_mode"
            --exp-name "$EXP_NAME"
            --jobs "$JOBS"
            --resume
        )
        [[ -n "$EVAL_SET" ]] && eval_args+=(--eval-set "$EVAL_SET")

        $PY eval/evaluate_vlm.py "${eval_args[@]}"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE: $EXP_NAME"
        echo ""
    done
done

echo "======================================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All $TOTAL conditions complete."
echo "======================================================================"
