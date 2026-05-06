#!/usr/bin/env python3
"""Simulator-subset eval driver.

Wrapper over eval_image.py / eval_video.py / evaluator.py. Resolves the
flat inference-output naming convention:

  Non-gen tracks:       <ptype>_<case_id>_<track>.{mp4,png}
  Gen track, video-gen: <ptype>_<case_id>_generation.mp4
  Gen track, image-gen: <ptype>_<case_id>_generation_<idx>_t<float>.png  (multi)

For image-gen generation, frames are symlinked into a staging dir named
'frame_<idx:02d>_t<t>s.png' so eval_image.evaluate_generation_keyframes
can discover them.

All paths are read from environment variables (no defaults). Required:
  APPLE_PI_ROOT      — root of inference outputs
  SIM_GT_ROOT        — root of simulator ground truth
  SIM_RESULTS_ROOT   — where eval JSONs are written
  SIM_STAGE_ROOT     — keyframe staging area for image-gen generation track

Usage:
    # smoke test: one case
    python run_eval_apple_pi.py --model all --track all \\
        --ptype at_rest --case-id 0

    # full run, one (model, track) at a time
    python run_eval_apple_pi.py --model <model> --track perception_text

    # everything
    python run_eval_apple_pi.py --model all --track all
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
APPLE_PI_ROOT = os.environ.get("APPLE_PI_ROOT")
GT_ROOT = os.environ.get("SIM_GT_ROOT")
RESULTS_ROOT = os.environ.get("SIM_RESULTS_ROOT")
STAGE_ROOT = os.environ.get("SIM_STAGE_ROOT")


def _require_env():
    missing = [n for n, v in [
        ("APPLE_PI_ROOT", APPLE_PI_ROOT),
        ("SIM_GT_ROOT", GT_ROOT),
        ("SIM_RESULTS_ROOT", RESULTS_ROOT),
        ("SIM_STAGE_ROOT", STAGE_ROOT),
    ] if not v]
    if missing:
        sys.exit(f"[error] missing required env vars: {missing} (see README)")

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

logger = logging.getLogger(__name__)


# ─── Discovery ──────────────────────────────────────────────

def discover_cases() -> List[Tuple[str, int]]:
    """List (ptype, case_id) for all 100 cases under GT_ROOT."""
    out = []
    for ptype in sorted(os.listdir(GT_ROOT)):
        p = os.path.join(GT_ROOT, ptype)
        if not os.path.isdir(p) or ptype.startswith("."):
            continue
        for case in sorted(os.listdir(p)):
            if case.isdigit() and os.path.isdir(os.path.join(p, case)):
                out.append((ptype, int(case)))
    return out


def find_model_output(
    model: str, ptype: str, case_id: int, track: str
) -> Optional[List[str]]:
    """Resolve apple-pi output file(s) for one (model, ptype, case_id, track).

    Returns:
      - non-gen: [<single path>]
      - generation video-gen: [<single mp4>]
      - generation image-gen: [<frames sorted by idx>]
      - None if any required file missing
    """
    base = os.path.join(APPLE_PI_ROOT, model)
    if not os.path.isdir(base):
        return None
    is_video = model in VIDEO_MODELS
    ext = "mp4" if is_video else "png"

    if track != "generation":
        f = f"{ptype}_{case_id}_{track}.{ext}"
        p = os.path.join(base, f)
        return [p] if os.path.isfile(p) else None

    # Generation
    if is_video:
        f = f"{ptype}_{case_id}_generation.{ext}"
        p = os.path.join(base, f)
        return [p] if os.path.isfile(p) else None

    # Image-gen generation: pattern <ptype>_<id>_generation_<idx>_t<t>.png
    prefix = f"{ptype}_{case_id}_generation_"
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


def stage_keyframes(model: str, ptype: str, case_id: int, paths: List[str]) -> str:
    r"""Symlink image-gen generation frames into eval_image-friendly layout.

    eval_image._find_keyframe_files matches: frame_(\d+)_t([\d.]+)s\.png
    We rename to satisfy that regex.
    """
    stage_dir = os.path.join(STAGE_ROOT, model, f"{ptype}_{case_id}")
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


# ─── Per-(model, ptype, case_id, track) eval ───────────────

def run_one(
    model: str,
    ptype: str,
    case_id: int,
    track: str,
    evaluator: VideoEvaluator,
    force: bool = False,
) -> dict:
    out_dir = os.path.join(RESULTS_ROOT, model, track)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ptype}_{case_id}.json")
    if os.path.isfile(out_path) and not force:
        logger.info(f"  skip {out_path} (exists)")
        with open(out_path) as f:
            return json.load(f)

    paths = find_model_output(model, ptype, case_id, track)
    if not paths:
        result = {
            "model": model, "ptype": ptype, "case_id": case_id, "track": track,
            "error": "model_output_missing",
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        return result

    case = CaseData.load(case_id, os.path.join(GT_ROOT, ptype))
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
                "model": model, "ptype": ptype, "case_id": case_id,
                "track": track, "video": paths[0],
                "score": round(track_score, 3),
                "details": scores.to_dict()[track],
                "feedback": scores.feedback,
            }
        else:
            if track_type == TrackType.GENERATION:
                stage_dir = stage_keyframes(model, ptype, case_id, paths)
                r = evaluate_generation_keyframes(
                    keyframe_dir=stage_dir, case=case, evaluator=evaluator,
                )
                result = {
                    "model": model, "ptype": ptype, "case_id": case_id,
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
                    "model": model, "ptype": ptype, "case_id": case_id,
                    "track": track, "image": paths[0],
                    "score": round(track_score, 3),
                    "details": scores.to_dict()[track],
                    "feedback": scores.feedback,
                }
    except Exception as e:
        logger.exception(f"FAIL {model} {ptype}_{case_id} {track}")
        result = {
            "model": model, "ptype": ptype, "case_id": case_id, "track": track,
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
    p.add_argument("--ptype", default=None,
                   help="filter to one physics_type bucket (smoke test)")
    p.add_argument("--case-id", type=int, default=None,
                   help="filter to one case_id")
    p.add_argument("--force", action="store_true",
                   help="re-eval even if result json exists")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel ThreadPool workers within each track "
                        "(1=serial, recommended start). API client retries on 429.")
    p.add_argument("--shard", default=None,
                   help="case-level shard 'K/N' (0-indexed). Split 100 cases "
                        "across N processes — each handles cases where "
                        "(case_index %% N) == K. For multi-GPU runs.")
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
        logger.error("API_KEY not set in env / .env")
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
    if args.ptype:
        cases = [c for c in cases if c[0] == args.ptype]
    if args.case_id is not None:
        cases = [c for c in cases if c[1] == args.case_id]
    if args.shard:
        try:
            k, n = (int(x) for x in args.shard.split("/"))
            assert 0 <= k < n, f"K={k} must satisfy 0<=K<N={n}"
        except Exception as e:
            logger.error(f"--shard must be 'K/N' with 0<=K<N: {e}")
            sys.exit(1)
        before = len(cases)
        cases = [c for i, c in enumerate(cases) if i % n == k]
        logger.info(f"Shard {k}/{n}: {len(cases)} of {before} cases")
    if not cases:
        logger.error(f"No cases match filter: ptype={args.ptype} case_id={args.case_id}")
        sys.exit(1)

    logger.info(f"Models ({len(models)}): {models}")
    logger.info(f"Tracks ({len(tracks)}): {tracks}")
    logger.info(f"Cases ({len(cases)}): {cases[:5]}{'...' if len(cases)>5 else ''}")
    logger.info(f"Total work units: {len(models) * len(tracks) * len(cases)}")

    n_done = n_fail = 0
    counter_lock = threading.Lock()
    t_total = time.time()

    def _exec_one(model, ptype, case_id, track):
        nonlocal n_done, n_fail
        t0 = time.time()
        try:
            r = run_one(model, ptype, case_id, track, evaluator, force=args.force)
        except Exception as e:
            r = {"error": f"unexpected: {e}"}
        elapsed = time.time() - t0
        tag = f"{track} | {model} | {ptype}/{case_id}"
        with counter_lock:
            if "error" in r:
                n_fail += 1
                logger.warning(f"FAIL  {tag} ({elapsed:.1f}s): {str(r.get('error',''))[:100]}")
            else:
                n_done += 1
                score = r.get("score", r.get("generation_score", "?"))
                logger.info(f"OK    {tag} ({elapsed:.1f}s) score={score}")
        return r

    for track in tracks:  # outer loop = track per user spec
        track_t0 = time.time()
        worklist = [(m, pt, cid, track) for m in models for pt, cid in cases]
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

    logger.info("\n========================================")
    logger.info(f"Done: {n_done}  Failed: {n_fail}  "
                f"Total elapsed: {(time.time()-t_total)/60:.1f} min")
    logger.info("========================================")


if __name__ == "__main__":
    main()
