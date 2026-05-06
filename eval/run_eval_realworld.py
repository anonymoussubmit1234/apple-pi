#!/usr/bin/env python3
"""Real-world subset eval driver.

Wrapper over eval_image.py / eval_video.py / evaluator.py. Same idea as
run_eval_apple_pi.py but with two adjustments:

  1. Inference flat naming uses physics-taxonomy buckets that flatten the
     2-level GT path. e.g.
       Inference filename:   ConservationofMomentum_InelasticCollision_3_perception_text.mp4
       GT path:              <REALWORLD_GT_ROOT>/ConservationofMomentum/InelasticCollision/3/...
       Special case:         Composition_<n>_*  →  Composition/<n>/...   (1-level)

  2. GT video is `<case>/rgb/video.mp4` (no per-frame PNG sequence).
     CaseData.load handles this via its `gt_video_path` field.

All paths are read from environment variables (no defaults). Required:
  APPLE_PI_REALWORLD_ROOT   — root of inference outputs
  REALWORLD_GT_ROOT         — root of real-world ground truth
  REALWORLD_RESULTS_ROOT    — where eval JSONs are written
  REALWORLD_STAGE_ROOT      — keyframe staging area for image-gen generation track

Usage:
    python run_eval_realworld.py --model all --track perception_text --workers 4
    # Multi-GPU sharded:
    CUDA_VISIBLE_DEVICES=0 ... --shard 0/3 ...
    CUDA_VISIBLE_DEVICES=1 ... --shard 1/3 ...
    CUDA_VISIBLE_DEVICES=2 ... --shard 2/3 ...
"""


# Make root-level shared modules (api_client, config, utils) importable.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

from api_client import APIClient
from config import APIConfig, CaseData, OptimizationConfig, TrackType
from evaluator import VideoEvaluator
from eval_image import evaluate_single_image, evaluate_generation_keyframes
from eval_video import evaluate_video

# Paths resolved from env vars — no fallback defaults. Read lazily so
# `--help` works without setup; `_require_env()` validates before main work.
APPLE_PI_REALWORLD_ROOT = os.environ.get("APPLE_PI_REALWORLD_ROOT")
GT_ROOT = os.environ.get("REALWORLD_GT_ROOT")
RESULTS_ROOT = os.environ.get("REALWORLD_RESULTS_ROOT")
STAGE_ROOT = os.environ.get("REALWORLD_STAGE_ROOT")


def _require_env():
    missing = [n for n, v in [
        ("APPLE_PI_REALWORLD_ROOT", APPLE_PI_REALWORLD_ROOT),
        ("REALWORLD_GT_ROOT", GT_ROOT),
        ("REALWORLD_RESULTS_ROOT", RESULTS_ROOT),
        ("REALWORLD_STAGE_ROOT", STAGE_ROOT),
    ] if not v]
    if missing:
        sys.exit(f"[error] missing required env vars: {missing} (see README)")

# 7 models — same 6 as sim plus doubao-seedance-2-0-260128
VIDEO_MODELS = {"veo3.1", "VBVR-Wan2.2", "doubao-seedance-2-0-260128"}
IMAGE_MODELS = {
    "gemini-3.1-flash-image-preview",
    "gpt-image-2",
    "SenseNova-U1-8B-MoT",
    "SenseNova-U1-8B-MoT-think",
}
ALL_MODELS = sorted(VIDEO_MODELS | IMAGE_MODELS)
ALL_TRACKS = [
    "perception_text",
    "perception_graphic",
    "comprehension_text",
    "comprehension_graphic",
    "generation",
]

# Buckets that nest 2 levels in GT (Composition is flat)
TWO_LEVEL_TOPS = ("ConservationofMomentum", "Gravity", "NewtonsFirstLaw")

logger = logging.getLogger(__name__)


# ─── Path / naming helpers ──────────────────────────────────

def bucket_to_gt_subpath(bucket: str) -> str:
    """`ConservationofMomentum_InelasticCollision` → `ConservationofMomentum/InelasticCollision`.

    Composition stays flat. The 3 other top-levels each have a single
    underscore separating top from subtype.
    """
    for top in TWO_LEVEL_TOPS:
        prefix = top + "_"
        if bucket.startswith(prefix):
            return f"{top}/{bucket[len(prefix):]}"
    return bucket  # Composition


def parse_inference_filename(fname: str):
    """Return (bucket, case_id, track, gen_meta_or_none) or None.

    Examples:
      "Composition_0_perception_text.mp4"  →  ("Composition", 0, "perception_text", None)
      "ConservationofMomentum_InelasticCollision_3_generation.mp4"
        →  ("ConservationofMomentum_InelasticCollision", 3, "generation", None)
      "gpt-image-2/Gravity_Freefall_5_generation_2_t1.5.png"
        →  ("Gravity_Freefall", 5, "generation", {"idx": 2, "t": 1.5})
    """
    base = fname.rsplit(".", 1)[0]
    m = re.match(r"^(.+)_(\d+)_generation_(\d+)_t([\d.]+)$", base)
    if m:
        return m.group(1), int(m.group(2)), "generation", {
            "idx": int(m.group(3)), "t": float(m.group(4))
        }
    for tr in ALL_TRACKS:
        suff = "_" + tr
        if base.endswith(suff):
            stem = base[: -len(suff)]
            m2 = re.match(r"^(.+)_(\d+)$", stem)
            if m2:
                return m2.group(1), int(m2.group(2)), tr, None
    return None


# ─── Discovery ──────────────────────────────────────────────

def discover_cases() -> List[Tuple[str, int]]:
    """Cases (bucket, case_id) actually present in the inference set —
    not all GT cases (157) but only the inference subset (100)."""
    seen = set()
    # Use any model dir that's available; image-gen has more files but any works
    sample_model = sorted(os.listdir(APPLE_PI_REALWORLD_ROOT))[0]
    sample_dir = os.path.join(APPLE_PI_REALWORLD_ROOT, sample_model)
    for f in os.listdir(sample_dir):
        if not (f.endswith(".png") or f.endswith(".mp4")):
            continue
        r = parse_inference_filename(f)
        if r:
            seen.add((r[0], r[1]))
    # Try a video-gen too in case that one was the cleaner set; union them
    for model in os.listdir(APPLE_PI_REALWORLD_ROOT):
        if model == sample_model:
            continue
        d = os.path.join(APPLE_PI_REALWORLD_ROOT, model)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if not (f.endswith(".png") or f.endswith(".mp4")):
                continue
            r = parse_inference_filename(f)
            if r:
                seen.add((r[0], r[1]))
    return sorted(seen, key=lambda x: (x[0], x[1]))


def find_model_output(model: str, bucket: str, case_id: int, track: str
                      ) -> Optional[List[str]]:
    """Resolve realworld model output(s)."""
    base = os.path.join(APPLE_PI_REALWORLD_ROOT, model)
    if not os.path.isdir(base):
        return None
    is_video = model in VIDEO_MODELS
    ext = "mp4" if is_video else "png"

    if track != "generation":
        f = f"{bucket}_{case_id}_{track}.{ext}"
        p = os.path.join(base, f)
        return [p] if os.path.isfile(p) else None

    if is_video:
        f = f"{bucket}_{case_id}_generation.{ext}"
        p = os.path.join(base, f)
        return [p] if os.path.isfile(p) else None

    # Image-gen generation: globbed keyframes
    prefix = f"{bucket}_{case_id}_generation_"
    cands: List[Tuple[int, float, str]] = []
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)_t([\d.]+)\.png$")
    for fn in os.listdir(base):
        m = pat.match(fn)
        if m:
            cands.append((int(m.group(1)), float(m.group(2)), os.path.join(base, fn)))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0])
    return [c[2] for c in cands]


def stage_keyframes(model: str, bucket: str, case_id: int,
                    paths: List[str]) -> str:
    """Symlink image-gen generation frames into eval_image-friendly layout."""
    stage_dir = os.path.join(STAGE_ROOT, model, f"{bucket}_{case_id}")
    os.makedirs(stage_dir, exist_ok=True)
    pat = re.compile(r"_generation_(\d+)_t([\d.]+)\.(png|jpg)$")
    for src in paths:
        m = pat.search(src)
        if not m:
            continue
        idx = int(m.group(1))
        t = float(m.group(2))
        ext = m.group(3)
        dst = os.path.join(stage_dir, f"frame_{idx:02d}_t{t}s.{ext}")
        if not os.path.exists(dst):
            try:
                os.symlink(os.path.abspath(src), dst)
            except FileExistsError:
                pass
    return stage_dir


def gt_demo_dir_for_bucket(bucket: str) -> str:
    """Realworld GT root for `CaseData.load(case_id, gt_demo_dir)` lookups."""
    return os.path.join(GT_ROOT, bucket_to_gt_subpath(bucket))


# ─── Per-(model, bucket, case_id, track) eval ───────────────

def run_one(
    model: str,
    bucket: str,
    case_id: int,
    track: str,
    evaluator: VideoEvaluator,
    force: bool = False,
) -> dict:
    out_dir = os.path.join(RESULTS_ROOT, model, track)
    os.makedirs(out_dir, exist_ok=True)
    # Use same flat naming on disk as model side: bucket_caseid.json
    out_path = os.path.join(out_dir, f"{bucket}_{case_id}.json")
    if os.path.isfile(out_path) and not force:
        with open(out_path) as f:
            return json.load(f)

    paths = find_model_output(model, bucket, case_id, track)
    if not paths:
        result = {
            "model": model, "bucket": bucket, "case_id": case_id, "track": track,
            "error": "model_output_missing",
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        return result

    gt_demo_dir = gt_demo_dir_for_bucket(bucket)
    case = CaseData.load(case_id, gt_demo_dir)
    track_type = TrackType(track)
    is_video = model in VIDEO_MODELS

    try:
        if is_video:
            scores = evaluate_video(
                video_path=paths[0],
                case=case,
                track=track_type,
                evaluator=evaluator,
                num_eval_frames=-1,
            )
            track_score = scores.track_score(track_type)
            result = {
                "model": model, "bucket": bucket, "case_id": case_id,
                "track": track, "video": paths[0],
                "score": round(track_score, 3),
                "details": scores.to_dict()[track],
                "feedback": scores.feedback,
            }
        else:
            if track_type == TrackType.GENERATION:
                stage_dir = stage_keyframes(model, bucket, case_id, paths)
                r = evaluate_generation_keyframes(
                    keyframe_dir=stage_dir, case=case, evaluator=evaluator,
                )
                result = {
                    "model": model, "bucket": bucket, "case_id": case_id,
                    "track": track, "keyframe_dir": stage_dir,
                    "n_keyframes": len(paths),
                    "generation_score": r["generation_score"],
                    "gemini_avg": r["gemini_avg"],
                    "programmatic": r.get("programmatic"),
                    "details": r["track_scores"],
                    "per_frame_count": len(r["per_frame"]),
                }
            else:
                scores = evaluate_single_image(
                    image_path=paths[0], case=case, track=track_type,
                    evaluator=evaluator,
                )
                track_score = scores.track_score(track_type)
                result = {
                    "model": model, "bucket": bucket, "case_id": case_id,
                    "track": track, "image": paths[0],
                    "score": round(track_score, 3),
                    "details": scores.to_dict()[track],
                    "feedback": scores.feedback,
                }
    except Exception as e:
        logger.exception(f"FAIL {model} {bucket}_{case_id} {track}")
        result = {
            "model": model, "bucket": bucket, "case_id": case_id, "track": track,
            "error": str(e),
        }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


# ─── CLI ────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   help=f"single model or 'all'. Valid: {ALL_MODELS}")
    p.add_argument("--track", required=True,
                   help=f"single track, 'all', or 'all_no_gen'. "
                        f"Valid singles: {ALL_TRACKS}")
    p.add_argument("--bucket", default=None,
                   help="filter to one realworld bucket (e.g. Composition or "
                        "ConservationofMomentum_InelasticCollision)")
    p.add_argument("--case-id", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel ThreadPool workers within each track")
    p.add_argument("--shard", default=None,
                   help="case-level shard 'K/N' for multi-GPU")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    _require_env()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    api = APIConfig()
    if not api.api_key:
        logger.error("API_KEY not set")
        sys.exit(1)
    client = APIClient(api)
    evaluator = VideoEvaluator(client, OptimizationConfig())

    models = ALL_MODELS if args.model == "all" else [args.model]
    if args.track == "all":
        tracks = ALL_TRACKS
    elif args.track == "all_no_gen":
        tracks = [t for t in ALL_TRACKS if t != "generation"]
    else:
        tracks = [args.track]

    cases = discover_cases()
    if args.bucket:
        cases = [c for c in cases if c[0] == args.bucket]
    if args.case_id is not None:
        cases = [c for c in cases if c[1] == args.case_id]
    if args.shard:
        try:
            k, n = (int(x) for x in args.shard.split("/"))
            assert 0 <= k < n
        except Exception as e:
            logger.error(f"--shard must be 'K/N' with 0<=K<N: {e}")
            sys.exit(1)
        before = len(cases)
        cases = [c for i, c in enumerate(cases) if i % n == k]
        logger.info(f"Shard {k}/{n}: {len(cases)} of {before} cases")
    if not cases:
        logger.error(f"No cases match filter")
        sys.exit(1)

    logger.info(f"Models ({len(models)}): {models}")
    logger.info(f"Tracks ({len(tracks)}): {tracks}")
    logger.info(f"Cases ({len(cases)}): first 3 = {cases[:3]}")
    logger.info(f"Total work units: {len(models) * len(tracks) * len(cases)}")

    n_done = n_fail = 0
    counter_lock = threading.Lock()
    t_total = time.time()

    def _exec_one(model, bucket, cid, track):
        nonlocal n_done, n_fail
        t0 = time.time()
        try:
            r = run_one(model, bucket, cid, track, evaluator, force=args.force)
        except Exception as e:
            r = {"error": f"unexpected: {e}"}
        elapsed = time.time() - t0
        tag = f"{track} | {model} | {bucket}/{cid}"
        with counter_lock:
            if "error" in r:
                n_fail += 1
                logger.warning(f"FAIL  {tag} ({elapsed:.1f}s): {str(r.get('error',''))[:100]}")
            else:
                n_done += 1
                score = r.get("score", r.get("generation_score", "?"))
                logger.info(f"OK    {tag} ({elapsed:.1f}s) score={score}")
        return r

    for track in tracks:
        track_t0 = time.time()
        worklist = [(m, b, c, track) for m in models for b, c in cases]
        logger.info(f"\n>>> Track {track}: {len(worklist)} units, workers={args.workers}")
        if args.workers <= 1:
            for w in worklist:
                _exec_one(*w)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(_exec_one, *w) for w in worklist]
                for _ in as_completed(futures):
                    pass
        logger.info(f">>> Track {track} done in {(time.time()-track_t0)/60:.1f} min")

    logger.info(f"\nDone: {n_done}  Failed: {n_fail}  "
                f"Total: {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
