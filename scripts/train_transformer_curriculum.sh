#!/usr/bin/env bash
set -euo pipefail

INIT_MODEL="${INIT_MODEL:-}"
V2_MODEL="${V2_MODEL:-models/naruto_actor_critic_v2.pt}"
OUT_DIR="${OUT_DIR:-models}"
OUT_PREFIX="${OUT_PREFIX:-naruto_actor_critic_transformer_skill_features_curriculum}"
SEED="${SEED:-7}"
MAX_ACTIONS="${MAX_ACTIONS:-300}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"

HEURISTIC_EPISODES_1="${HEURISTIC_EPISODES_1:-10000}"
HEURISTIC_EPISODES_2="${HEURISTIC_EPISODES_2:-15000}"
V2_EPISODES_1="${V2_EPISODES_1:-10000}"
V2_EPISODES_2="${V2_EPISODES_2:-10000}"

if [[ -n "$INIT_MODEL" && ! -f "$INIT_MODEL" ]]; then
  echo "missing INIT_MODEL=$INIT_MODEL" >&2
  exit 1
fi

if [[ ! -f "$V2_MODEL" ]]; then
  echo "missing V2_MODEL=$V2_MODEL" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" reports

HEURISTIC_30K="$OUT_DIR/${OUT_PREFIX}_heuristic_30k.pt"
HEURISTIC_45K="$OUT_DIR/${OUT_PREFIX}_heuristic_45k.pt"
VS_V2_55K="$OUT_DIR/${OUT_PREFIX}_vs_v2_55k.pt"
VS_V2_65K="$OUT_DIR/${OUT_PREFIX}_vs_v2_65k.pt"

echo "stage=1 opponent=heuristic init=${INIT_MODEL:-from_scratch} save=$HEURISTIC_30K"
STAGE_1_INIT_ARGS=()
if [[ -n "$INIT_MODEL" ]]; then
  STAGE_1_INIT_ARGS=(--init-model-path "$INIT_MODEL")
fi
uv run --extra rl python scripts/train_rl_pytorch.py \
  --model-arch transformer \
  "${STAGE_1_INIT_ARGS[@]}" \
  --episodes "$HEURISTIC_EPISODES_1" \
  --batch-episodes 10 \
  --opponent heuristic \
  --learning-rate 1e-4 \
  --entropy-coef 0.008 \
  --value-coef 0.5 \
  --seed "$SEED" \
  --max-actions "$MAX_ACTIONS" \
  --log-interval "$LOG_INTERVAL" \
  --save-path "$HEURISTIC_30K"

echo "stage=2 opponent=heuristic init=$HEURISTIC_30K save=$HEURISTIC_45K"
uv run --extra rl python scripts/train_rl_pytorch.py \
  --model-arch transformer \
  --init-model-path "$HEURISTIC_30K" \
  --episodes "$HEURISTIC_EPISODES_2" \
  --batch-episodes 10 \
  --opponent heuristic \
  --learning-rate 5e-5 \
  --entropy-coef 0.005 \
  --value-coef 0.5 \
  --seed "$((SEED + 1))" \
  --max-actions "$MAX_ACTIONS" \
  --log-interval "$LOG_INTERVAL" \
  --save-path "$HEURISTIC_45K"

echo "stage=3 opponent=rl_v2 init=$HEURISTIC_45K save=$VS_V2_55K"
uv run --extra rl python scripts/train_rl_pytorch.py \
  --model-arch transformer \
  --init-model-path "$HEURISTIC_45K" \
  --episodes "$V2_EPISODES_1" \
  --batch-episodes 10 \
  --opponent rl \
  --opponent-model-path "$V2_MODEL" \
  --learning-rate 3e-5 \
  --entropy-coef 0.005 \
  --value-coef 0.5 \
  --seed "$((SEED + 2))" \
  --max-actions "$MAX_ACTIONS" \
  --log-interval "$LOG_INTERVAL" \
  --save-path "$VS_V2_55K"

echo "stage=4 opponent=rl_v2 init=$VS_V2_55K save=$VS_V2_65K"
uv run --extra rl python scripts/train_rl_pytorch.py \
  --model-arch transformer \
  --init-model-path "$VS_V2_55K" \
  --episodes "$V2_EPISODES_2" \
  --batch-episodes 10 \
  --opponent rl \
  --opponent-model-path "$V2_MODEL" \
  --learning-rate 1e-5 \
  --entropy-coef 0.003 \
  --value-coef 0.5 \
  --seed "$((SEED + 3))" \
  --max-actions "$MAX_ACTIONS" \
  --log-interval "$LOG_INTERVAL" \
  --save-path "$VS_V2_65K"

echo "compare deterministic transformer_vs_v2"
uv run --extra rl python scripts/compare_rl_models.py \
  --model-a "$V2_MODEL" \
  --label-a mlp_v2 \
  --model-b "$VS_V2_65K" \
  --label-b transformer_curriculum \
  --matches-per-pair 3 \
  --seed "$SEED" \
  --max-actions 500 \
  --output reports/rl_compare_mlp_v2_vs_transformer_curriculum.json

echo "done final_model=$VS_V2_65K"
