# Apple-π — Physics Infographics Benchmark

Code release for **Apple-π**, a benchmark that verifies whether video and
image-generation models can reason about the physical world in a
law-grounded manner. Evaluation is structured around the three-stage
reasoning blueprint **Perception → Formulation → Deduction**.

The dataset (simulator + real-world subsets) is hosted separately on
HuggingFace:
**https://huggingface.co/datasets/AnonymousSubmitt/apple-pi**

The five tracks (paper labels in parentheses):

| Track | Subtrack | Code identifier |
|---|---|---|
| Perception | text | `perception_text` (P-T) |
| Perception | graphic | `perception_graphic` (P-G) |
| Formulation | text | `comprehension_text` (F-T) |
| Formulation | graphic | `comprehension_graphic` (F-G) |
| Deduction | — | `generation` (Ded.) |

---

## Repository layout

```
apple-pi/
├── README.md / LICENSE / .gitignore
├── demo.py                  # one-case end-to-end demo (start here)
│
├── api_client.py            # shared API client (chat / image-gen / video-gen)
├── config.py                # APIConfig, OptimizationConfig, CaseData, TrackType
├── utils.py                 # JSON parsing, retry helpers
├── Formula/                 # frozen ground-truth physics formulas per task
│                            #   type (used by the F-T track to score the model's
│                            #   formula choice against the canonical answer)
│
├── examples/                # ── verification kit (see "Quick verification")
│   ├── README.md
│   ├── verify.sh                 # end-to-end smoke test against bundled outputs
│   ├── inference_outputs_sim/<model>/circular_0_*.{png,mp4}
│   └── inference_outputs_realworld/<model>/Gravity_Freefall_1_*.{png,mp4}
│
├── inference/               # ── inference / generation
│   ├── build_bench.py            # per-(case, track) prompt builder → bench/*.json
│   ├── prompt_templates.py       # per-(model_family, track) inference prompts
│   ├── generate_nano_banana.py   # reference impl: image-gen (Nano-Banana style)
│   └── generate_veo3_single.py   # reference impl: video-gen (Veo3 style)
│
└── eval/                    # ── evaluation pipeline
    ├── evaluator.py              # frozen LLM-judge prompts + per-track rubric (TRACK_FIELDS)
    ├── eval_image.py             # image-track + generation-keyframe evaluator
    ├── eval_video.py             # video-track evaluator (last-frame for non-gen tracks)
    ├── seg_evaluator.py          # SAM3 segmentation IoU for graphic tracks
    ├── programmatic_metric.py    # PSNR / IoU / MoGe-2 velocity error for generation
    ├── run_eval_apple_pi.py      # main eval driver — simulator subset
    └── run_eval_realworld.py     # main eval driver — real-world subset
```

Subdirectory scripts auto-resolve the repo root on `sys.path` so they can
import the shared modules. **Run all scripts from the repository root.**

---

## Quickstart

Download one full case, build prompts, see what your model would
receive. `demo.py` does NOT call any inference API — it just shows
what the pipeline would feed to one model on one (case, track). To
also run the judge against the GT image as a sanity check, pass
`--eval` (needs more deps + API credentials, see below).

```bash
pip install huggingface_hub Pillow requests python-dotenv

# Simulator demo — sim_subset/circular/0  (~280 MB)
python demo.py

# Real-world demo — realworld_subset/Gravity/Freefall/1  (~5 MB)
python demo.py --split realworld

# Optional: also run the judge on the GT image
pip install numpy opencv-python                      # extra deps for eval
export API_KEY=<your_token>
export LLM_CHAT_URL=<openai-compatible chat endpoint>
export LLM_CHAT_MODEL=<model name>
python demo.py --eval                                # GT-as-output should score ≈ 1.0
```

## Quick verification (no inference needed)

The repo bundles inference outputs from two models on two cases under
[`examples/`](examples/), so you can run the eval pipeline end-to-end
without producing any model outputs yourself.

```bash
# Install the eval-driver deps (heavier than the demo deps above):
pip install huggingface_hub Pillow requests python-dotenv numpy opencv-python

export API_KEY=<your_token>
export LLM_CHAT_URL=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
export LLM_CHAT_MODEL=gemini-2.5-flash
./examples/verify.sh
```

This downloads two GT cases (~285 MB), runs the simulator and real-world
eval drivers on the bundled outputs, and writes score JSONs to
`./results/`. Each JSON has `score`, per-dimension `details`, and judge
`feedback`. See [examples/README.md](examples/README.md) for what's in
the verification kit.

---

## Full setup (for the actual benchmark)

### Python environment
Tested with Python 3.10. Minimal eval-only deps:
```bash
conda create -n apple-pi python=3.10
conda activate apple-pi
pip install huggingface_hub Pillow requests python-dotenv numpy opencv-python
```

Add the following only if you need the corresponding programmatic metrics
(LLM-judge tracks alone do not require them):
```bash
pip install torch torchvision transformers   # for SAM3 segmentation_iou (graphic tracks)
huggingface-cli login                        # SAM3 weights are gated
pip install git+https://github.com/microsoft/MoGe.git   # MoGe-2 velocity error (gen track)
```

### Download the dataset
```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="AnonymousSubmitt/apple-pi",
    repo_type="dataset",
    local_dir="./data",
)
```

After download, `./data/` contains `sim_subset/<ptype>/<case>/...` and
`realworld_subset/<pillar>/<task>/<case>/...`.

Each case directory:
```
<case_id>/
  initial_state/
    rgb_0000.png                  # the annotated first frame (model input)
    rgb_0000_white_bg.png         # reference for Perception-Text
    rgb_0000_white_bg_obj.png     # reference for Perception-Graphic
  rgb/                            # ground-truth RGB sequence (sim) or video.mp4 (real-world)
  instantaneous_velocity/
    velocity_annotated.png        # reference for Formulation-Graphic
    velocity.json                 # numerical velocity + target_time
  formula_info.json               # for Formulation-Text track
  physics_duration.txt            # simulation duration
  annotation.txt                  # physics parameters as text
  mask/, instance_segmentation/   # for the SAM3-based programmatic metrics
```

### Environment variables
The pipeline reads **all** paths and API endpoints from environment
variables — there are no hardcoded fallbacks. Set them in `.env` or via
`export`:

```bash
# ── Data roots ──
export SIM_GT_ROOT=./data/sim_subset
export REALWORLD_GT_ROOT=./data/realworld_subset

# ── Inference output roots ──
export APPLE_PI_ROOT=./inference_outputs/sim
export APPLE_PI_REALWORLD_ROOT=./inference_outputs/realworld

# ── Eval output roots ──
export SIM_RESULTS_ROOT=./results/sim
export SIM_STAGE_ROOT=./stage/sim
export REALWORLD_RESULTS_ROOT=./results/realworld
export REALWORLD_STAGE_ROOT=./stage/realworld

# ── API config (see "API providers" below) ──
export API_KEY=<your_token>
export LLM_CHAT_URL=<chat completions endpoint>           # for evaluation
export LLM_CHAT_MODEL=<model name in chat payload>        # e.g. gemini-2.5-flash
export IMAGE_GEN_BASE=<image-gen base URL>                # optional, for inference
export VEO3_BASE=<video-gen base URL>                     # optional, for inference
export VEO3_FAST_BASE=<video-gen-fast base URL>           # optional
```

### API providers

There are **two independent API surfaces**: one for **inference** (model
generation) and one for **evaluation** (LLM judge). Use any combination.

**Evaluation — `LLM_CHAT_URL` + `LLM_CHAT_MODEL`**: any OpenAI-compatible
chat-completions endpoint (POST `messages` with images, get back
`choices[0].message.content`). The judge prompts target a strong
multimodal model.

| Provider | URL | Example `LLM_CHAT_MODEL` |
|---|---|---|
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions` | `gemini-2.5-flash` |
| OpenAI | `https://api.openai.com/v1/chat/completions` | `gpt-4o` |
| OpenRouter | `https://openrouter.ai/api/v1/chat/completions` | (any) |
| Self-hosted vLLM | `http://localhost:8000/v1/chat/completions` | (your model) |

**Inference — `IMAGE_GEN_BASE` / `VEO3_BASE`**: needed only if you run
the bundled `inference/generate_*.py` reference scripts. They target an
async two-step protocol (POST submit → GET poll for the result). If
your provider's protocol differs, you have two options:

1. **Skip the bundled scripts** and run inference with your own model
   wrapper. Feed each entry from the `bench/*.json` files (built by
   `inference/build_bench.py`) into your pipeline, then save outputs
   under `$APPLE_PI_ROOT/<your_model>/` following the naming convention
   in Stage 1 below. Evaluation reads from disk and is independent of
   how files were produced.
2. **Fork `inference/api_client.py`** (the `generate_image` /
   `generate_video` methods) to match your provider's contract.

---

## End-to-end flow

### Stage 1 — Inference

Build the per-(case, track) prompt manifest:
```bash
python inference/build_bench.py --root ./data --out-dir ./bench
```
This produces `bench/videogen.json` (one entry per case per track for
video-gen models) and `bench/unified.json` (one entry per non-gen track
plus a 2 fps keyframe sequence over the generation track for image-gen
models).

Run inference. The bundled scripts work if your provider matches the
async protocol; otherwise see "API providers" above.
```bash
# Image-gen — 1 PNG per non-gen track + a keyframe sequence for generation
python inference/generate_nano_banana.py \
    --case 0 --demo-dir $SIM_GT_ROOT/circular \
    --output-dir $APPLE_PI_ROOT/your_model_name

# Video-gen — 1 MP4 per track
python inference/generate_veo3_single.py \
    --case 0 --demo-dir $SIM_GT_ROOT/circular \
    --output-dir $APPLE_PI_ROOT/your_model_name
```

Inference outputs use this flat naming convention (the evaluators
discover them by pattern):

```
$APPLE_PI_ROOT/<model_name>/
  <ptype>_<case_id>_perception_text.{png,mp4}
  <ptype>_<case_id>_perception_graphic.{png,mp4}
  <ptype>_<case_id>_comprehension_text.{png,mp4}
  <ptype>_<case_id>_comprehension_graphic.{png,mp4}
  <ptype>_<case_id>_generation.mp4                  # video-gen models
  <ptype>_<case_id>_generation_<idx>_t<t>.png       # image-gen models (multi-keyframe)
```

For the real-world subset, replace `<ptype>` with the bucket name
`<Pillar>_<Task>` (e.g. `ConservationofMomentum_InelasticCollision`),
or just `Composition` for the multi-law task.

### Stage 2 — Evaluation

```bash
# Smoke test: one case, all tracks
python eval/run_eval_apple_pi.py --model your_model --track all \
    --ptype circular --case-id 0

# One model on one track
python eval/run_eval_apple_pi.py --model your_model --track perception_text
python eval/run_eval_realworld.py --model your_model --track perception_text

# Full sweep (long; supports sharding across GPUs)
python eval/run_eval_apple_pi.py --model all --track all --workers 4
```

Per-(model, track, case) JSON outputs are written under
`$SIM_RESULTS_ROOT/<model>/<track>/<case>.json` (or `$REALWORLD_RESULTS_ROOT/...`).
Each JSON contains:
- `score` — overall track score in [0, 1]
- `details` — per-dimension judge scores (see `evaluator.TRACK_FIELDS`)
- `programmatic` — PSNR / IoU / velocity-error breakdown (generation only)
- `feedback` — the judge's free-text rationale

### Stage 3 — Aggregation

Average the `score` field across all `<ptype>_<case_id>.json` files for
each (model, track). The paper reports per-track means and per-pillar
means (Universal Gravitation, Conservation of Momentum, Newton's First
Law, Multi-Law).

---

## Programmatic metrics

Some tracks combine the LLM-judge scores with non-LLM metrics:

| Track | Programmatic metric |
|---|---|
| `perception_graphic` | SAM3-aligned `segmentation_iou` (object-mask IoU) |
| `comprehension_graphic` | SAM3-aligned `comp_graphic_segmentation_iou` |
| `generation` | PSNR, masked-PSNR, spatial / spatiotemporal / weighted-spatial IoU, MoGe-2 velocity error |

These appear under each JSON's `details` field alongside the LLM-judge
scores. `perception_text` and `comprehension_text` are pure LLM-judge.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
