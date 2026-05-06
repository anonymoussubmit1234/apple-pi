# Verification kit

Bundled inference outputs from two reference models on two cases — one
per split — so reviewers can run the full eval pipeline end-to-end
without producing any model outputs themselves.

## Cases

| Split | Case | GT source |
|---|---|---|
| `sim` | `sim_subset/circular/0` | HF download (~280 MB) |
| `realworld` | `realworld_subset/Gravity/Freefall/1` | HF download (~5 MB) |

## Bundled inference outputs

For each case, two reference models are included — one image-gen and one
video-gen — so both eval drivers (image + video paths) are exercised:

- `gemini-3.1-flash-image-preview` — image-gen (PNG outputs, including
  a 2 fps keyframe sequence for the generation track).
- `veo3.1` — video-gen (MP4 outputs, one per track).

Files follow the flat naming convention from the main README:

```
<ptype>_<case_id>_<track>.{png,mp4}                        # non-gen tracks
<ptype>_<case_id>_generation.mp4                           # video-gen
<ptype>_<case_id>_generation_<idx>_t<t>.png                # image-gen, multi-keyframe
```

For real-world cases, `<ptype>` is replaced by `<Pillar>_<Task>`
(e.g. `Gravity_Freefall_1_perception_text.png`).

## verify.sh

Runs two smoke-test evals end-to-end against the bundled outputs:
- sim — `gemini-3.1-flash-image-preview` on `circular/0`, `perception_text` track
- realworld — `veo3.1` on `Gravity_Freefall/1`, `perception_text` track

```bash
export API_KEY=<your_token>
export LLM_CHAT_URL=<chat_completions_endpoint>
export LLM_CHAT_MODEL=<model_name_for_chat_payload>
./examples/verify.sh
```

Two score JSONs are written under `./results/` — one per (model, case).
Each contains:
- `score` in [0, 1]
- per-dimension judge scores under `details`
- judge `feedback` (free text)

A correct setup produces non-zero scores. Image-gen models on the
perception / comprehension tracks should score noticeably higher than
video-gen models, since the image-gen training objective is to reproduce
annotations faithfully.
