#!/usr/bin/env bash
# Verification smoke-test for the Apple-π eval pipeline.
#
# Runs perception_text on:
#   - sim_subset/circular/0 with gemini-3.1-flash-image-preview (image-gen)
#   - realworld_subset/Gravity/Freefall/1 with veo3.1 (video-gen)
#
# Required env vars (set these before running):
#   API_KEY            — bearer token for the chat-completions endpoint
#   LLM_CHAT_URL       — chat-completions URL (OpenAI-compatible)
#   LLM_CHAT_MODEL     — model name passed in the chat payload
#                        (e.g. gemini-2.5-flash, gpt-4o)
#
# What this proves: GT loading, inference-output discovery, multimodal
# request construction, judge response parsing, score JSON output.
# It does NOT exercise programmatic metrics (PSNR / SAM3 IoU / MoGe-2),
# which require additional optional dependencies — see main README.

set -euo pipefail

# Resolve repo root from this script's location (examples/verify.sh).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${API_KEY:-}" || -z "${LLM_CHAT_URL:-}" || -z "${LLM_CHAT_MODEL:-}" ]]; then
    echo "ERROR: set API_KEY, LLM_CHAT_URL, LLM_CHAT_MODEL before running." >&2
    echo "Example:" >&2
    echo '    export API_KEY=<your_token>' >&2
    echo '    export LLM_CHAT_URL=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions' >&2
    echo '    export LLM_CHAT_MODEL=gemini-2.5-flash' >&2
    exit 1
fi

# ── Step 1: download both GT cases from HF ────────────────────────────
echo "=== [1/3] downloading GT for circular/0 (sim) + Gravity/Freefall/1 (realworld) ==="
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='AnonymousSubmitt/apple-pi',
    repo_type='dataset',
    local_dir='./data',
    allow_patterns=[
        'sim_subset/circular/0/**',
        'realworld_subset/Gravity/Freefall/1/**',
    ],
)
print('GT download complete.')
"

# ── Step 2: env vars for the eval drivers ─────────────────────────────
export SIM_GT_ROOT="$REPO_ROOT/data/sim_subset"
export APPLE_PI_ROOT="$REPO_ROOT/examples/inference_outputs_sim"
export SIM_RESULTS_ROOT="$REPO_ROOT/results/sim"
export SIM_STAGE_ROOT="$REPO_ROOT/stage/sim"

export REALWORLD_GT_ROOT="$REPO_ROOT/data/realworld_subset"
export APPLE_PI_REALWORLD_ROOT="$REPO_ROOT/examples/inference_outputs_realworld"
export REALWORLD_RESULTS_ROOT="$REPO_ROOT/results/realworld"
export REALWORLD_STAGE_ROOT="$REPO_ROOT/stage/realworld"

# ── Step 3: run two smoke evals ───────────────────────────────────────
echo ""
echo "=== [2/3] sim eval — gemini-3.1-flash-image-preview / circular/0 / perception_text ==="
python eval/run_eval_apple_pi.py \
    --model gemini-3.1-flash-image-preview \
    --track perception_text \
    --ptype circular --case-id 0

echo ""
echo "=== [3/3] realworld eval — veo3.1 / Gravity_Freefall/1 / perception_text ==="
python eval/run_eval_realworld.py \
    --model veo3.1 \
    --track perception_text \
    --bucket Gravity_Freefall --case-id 1

# ── Show results ──────────────────────────────────────────────────────
echo ""
echo "=== Results ==="
for f in \
    "$SIM_RESULTS_ROOT/gemini-3.1-flash-image-preview/perception_text/circular_0.json" \
    "$REALWORLD_RESULTS_ROOT/veo3.1/perception_text/Gravity_Freefall_1.json"; do
    if [[ -f "$f" ]]; then
        score=$(python -c "import json; print(json.load(open('$f'))['score'])")
        echo "  $f : score=$score"
    else
        echo "  MISSING: $f"
        exit 1
    fi
done

echo ""
echo "Verification PASSED. Both eval drivers produce score JSONs end-to-end."
