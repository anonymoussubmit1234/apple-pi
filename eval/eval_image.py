#!/usr/bin/env python3
"""Evaluate image-generation model outputs against the physics infographics benchmark.

Works with any image-gen model (Nano Banana Pro, etc.) that produces:
  - Tracks 1-4: a single PNG per (case, track)
  - Generation:  a directory of keyframe PNGs (timestamps parsed from filenames)

For tracks 1-4, the Gemini eval is identical to video-gen eval (evaluator.py).
For generation, we do pairwise per-frame Gemini eval + programmatic metrics.

Usage:
    # Single track
    python eval_image.py --image results/nb/case_0/perception_text_variant_A.png \
        --case 0 --track perception_text

    # All non-generation tracks for one case
    python eval_image.py --results-dir results/nb/case_0 --case 0 --track all

    # Generation track (keyframes)
    python eval_image.py --keyframe-dir results/nb/case_0/generation_frames_variant_B \
        --case 0 --track generation

    # Full eval: all tracks including generation
    python eval_image.py --results-dir results/nb/case_0 --case 0 --track all \
        --variant B
"""


# Make root-level shared modules (api_client, config, utils) importable.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np

from api_client import APIClient
from config import (
    APIConfig,
    BASE_DIR,
    CaseData,
    OptimizationConfig,
    TrackType,
    load_case_formula_info,
    parse_cases,
    parse_tracks,
)
from evaluator import TrackScores, VideoEvaluator
from utils import (
    bytes_to_data_url,
    frame_to_jpeg_bytes,
    image_to_data_url,
    parse_llm_json,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tracks 1-4: single image evaluation (shared with video-gen pipeline)
# ---------------------------------------------------------------------------

def evaluate_single_image(
    image_path: str,
    case: CaseData,
    track: TrackType,
    evaluator: VideoEvaluator,
) -> TrackScores:
    """Evaluate a single generated PNG against a non-generation track.

    This calls the same evaluator.evaluate() used by eval_video.py —
    the only difference is we load from a PNG file instead of extracting
    the last frame of an MP4.
    """
    assert track != TrackType.GENERATION, "Use evaluate_generation_keyframes for generation"

    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    gen_frames = [frame]

    # Build all the reference data URLs that evaluator.evaluate() needs
    first_frame_data_url = image_to_data_url(case.first_frame_path)

    first_frame_white_bg_data_url = None
    if track in [TrackType.PERCEPTION_TEXT, TrackType.PERCEPTION_GRAPHIC] and case.white_bg_first_frame_path:
        first_frame_white_bg_data_url = image_to_data_url(case.white_bg_first_frame_path)

    first_frame_white_bg_obj_data_url = None
    if track == TrackType.PERCEPTION_GRAPHIC and case.white_bg_obj_first_frame_path:
        first_frame_white_bg_obj_data_url = image_to_data_url(case.white_bg_obj_first_frame_path)

    formula_info = None
    if track == TrackType.COMPREHENSION_TEXT:
        cached = load_case_formula_info(case.case_dir)
        if not cached:
            raise FileNotFoundError(
                f"Missing formula_info.json for case {case.case_id}"
            )
        formula_info = {
            "correct_letter": str(cached["correct_letter"]),
            "correct_formula": str(cached["correct_formula"]),
            "annotation": str(cached.get("annotation", case.annotation_text)),
        }

    track_ref_data_url = None
    if track == TrackType.COMPREHENSION_GRAPHIC and case.track_image_path:
        track_ref_data_url = image_to_data_url(case.track_image_path)

    scores = evaluator.evaluate(
        track=track,
        first_frame_data_url=first_frame_data_url,
        gen_frames=gen_frames,
        gen_video_bytes=None,
        gt_frames=None,
        track_ref_data_url=track_ref_data_url,
        formula_info=formula_info,
        first_frame_white_bg_data_url=first_frame_white_bg_data_url,
        first_frame_white_bg_obj_data_url=first_frame_white_bg_obj_data_url,
    )

    # SAM3 segmentation IoU for perception_graphic
    if track == TrackType.PERCEPTION_GRAPHIC:
        try:
            from seg_evaluator import compute_segmentation_iou
            gen_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            iou = compute_segmentation_iou(
                gen_image=gen_rgb,
                case_dir=case.case_dir,
                is_realworld=case.is_realworld,
            )
            scores.segmentation_iou = iou
        except Exception as e:
            logger.warning(f"SAM3 segmentation IoU failed: {e}")
            scores.segmentation_iou = 0.0

    return scores


# ---------------------------------------------------------------------------
# Generation track: pairwise per-frame Gemini eval + programmatic metrics
# ---------------------------------------------------------------------------

EVAL_PROMPT_GENERATION_KEYFRAME = """\
You are evaluating a single keyframe from a physics simulation generation model.

You are provided with:
1. INITIAL ANNOTATED PHYSICS SCENE (the first frame with coordinate axes, text labels, etc.)
2. GROUND TRUTH FRAME at t = {time_point:.2f}s (the correct physical state at this moment)
3. GENERATED FRAME at t = {time_point:.2f}s (the model's output for this moment)

Compare the GENERATED FRAME against the GROUND TRUTH FRAME. Score each from 0.0 to 1.0:

- gen_annotations_removed: Are all human annotations from the INITIAL SCENE \
(text labels, coordinate axes, initial velocity arrows) removed in the generated frame? \
(0 = annotations clearly visible, 1 = perfectly clean)
- object_consistency: Do the physical objects match the GT in identity, count, and shape? \
(0 = objects missing/hallucinated/severely deformed, 1 = exact match)
- visual_quality: Overall visual quality — sharpness, clarity, absence of artifacts. \
(0 = severe artifacts/blur, 1 = high quality crisp rendering)
- object_position_match: Are the objects at the same positions as in the GT frame? \
(0 = completely wrong positions, 1 = exact position match)
- physics_accuracy: Does the overall physical state (positions, deformations, interactions) \
match the GT? (0 = completely wrong, 1 = physically accurate)

Respond with ONLY JSON:
{{
  "gen_annotations_removed": <float>,
  "object_consistency": <float>,
  "visual_quality": <float>,
  "object_position_match": <float>,
  "physics_accuracy": <float>,
  "feedback": "<1-2 sentences on the biggest issues>"
}}"""

GENERATION_KEYFRAME_FIELDS = [
    "gen_annotations_removed",
    "object_consistency",
    "visual_quality",
    "object_position_match",
    "physics_accuracy",
]


def _find_keyframe_files(keyframe_dir: str) -> List[dict]:
    """Discover keyframe PNGs and parse their timestamps.

    Expects filenames like: frame_00_t0.5s.png, frame_01_t1.0s.png, ...
    Returns sorted list of {"path": str, "time": float, "index": int}.
    """
    import re
    pattern = re.compile(r"frame_(\d+)_t([\d.]+)s\.png")
    entries = []
    for fname in sorted(os.listdir(keyframe_dir)):
        m = pattern.match(fname)
        if m:
            entries.append({
                "path": os.path.join(keyframe_dir, fname),
                "time": float(m.group(2)),
                "index": int(m.group(1)),
            })
    entries.sort(key=lambda e: e["time"])
    return entries


def _get_gt_frame(
    case: CaseData, time_sec: float
) -> Optional[np.ndarray]:
    """Get the GT frame for a given timestamp.

    Both simulator and real-world use the same mapping:
        frame_idx = round(time_sec * gt_fps)

    Simulator:  reads rgb/{frame_idx:04d}.png
    Real-world: seeks to frame_idx in rgb/video.mp4

    Frame 0 is the conditional image (t=0) in both cases.
    Since image-gen models generate from t >= 1/gen_fps (e.g. t=0.5),
    frame 0 is never accessed.
    """
    frame_idx = round(time_sec * case.gt_fps)

    if case.gt_video_path:
        # Real-world: extract frame from video
        cap = cv2.VideoCapture(case.gt_video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_idx >= total:
            cap.release()
            logger.warning(f"GT frame {frame_idx} out of range (video has {total} frames)")
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None
    else:
        # Simulator: read PNG file
        gt_path = os.path.join(case.gt_frames_dir, f"{frame_idx:04d}.png")
        if not os.path.exists(gt_path):
            return None
        return cv2.imread(gt_path, cv2.IMREAD_COLOR)


def evaluate_generation_keyframes(
    keyframe_dir: str,
    case: CaseData,
    evaluator: VideoEvaluator,
    skip_gemini: bool = False,
    skip_programmatic: bool = False,
) -> dict:
    """Evaluate generation track keyframes: pairwise Gemini + programmatic metrics.

    For each keyframe at timestamp t, finds the GT frame at the same t via:
        frame_idx = round(t * gt_fps)
    This works for both simulator (PNG files) and real-world (video.mp4).

    Returns:
        {
            "per_frame": [{"time": float, "gemini_scores": {...}, ...}, ...],
            "gemini_avg": {...},        # averaged Gemini scores across frames
            "programmatic": {...},      # PSNR, IoU from programmatic_metric.py
            "track_scores": TrackScores.to_dict()["generation"],
        }
    """
    keyframes = _find_keyframe_files(keyframe_dir)
    if not keyframes:
        raise FileNotFoundError(f"No keyframe PNGs found in {keyframe_dir}")

    gt_source = case.gt_video_path or case.gt_frames_dir
    logger.info(
        f"Found {len(keyframes)} keyframes in {keyframe_dir}, "
        f"GT source: {gt_source} ({case.gt_fps}fps)"
    )

    first_frame_data_url = image_to_data_url(case.first_frame_path)

    # --- Pairwise Gemini eval per frame ---
    per_frame_results = []
    all_gemini_scores = {f: [] for f in GENERATION_KEYFRAME_FIELDS}

    for kf in keyframes:
        result = {
            "time": kf["time"],
            "index": kf["index"],
            "path": kf["path"],
        }

        gt_frame_idx = round(kf["time"] * case.gt_fps)
        result["gt_frame_idx"] = gt_frame_idx

        gt_frame = _get_gt_frame(case, kf["time"])

        if not skip_gemini and gt_frame is not None:
            try:
                scores = _evaluate_single_keyframe_gemini(
                    evaluator=evaluator,
                    first_frame_data_url=first_frame_data_url,
                    gen_frame_path=kf["path"],
                    gt_frame=gt_frame,
                    time_point=kf["time"],
                )
                result["gemini_scores"] = scores
                for field in GENERATION_KEYFRAME_FIELDS:
                    val = scores.get(field, 0.0)
                    if val >= 0.0:
                        all_gemini_scores[field].append(val)
            except Exception as e:
                logger.warning(f"Gemini eval failed for frame {kf['index']}: {e}")
                result["gemini_error"] = str(e)
        elif gt_frame is None:
            logger.warning(f"No GT frame for t={kf['time']:.2f}s (idx={gt_frame_idx})")

        per_frame_results.append(result)

    # Average Gemini scores across frames
    gemini_avg = {}
    for field in GENERATION_KEYFRAME_FIELDS:
        vals = all_gemini_scores[field]
        gemini_avg[field] = float(np.mean(vals)) if vals else 0.0

    # --- Programmatic metrics ---
    prog_results = None
    if not skip_programmatic:
        try:
            from programmatic_metric import (
                Evaluator as ProgrammaticEvaluator,
                PSNRMetric,
                MaskedPSNRMetric,
                SpatialIoUMetric,
                SpatiotemporalIoUMetric,
                WeightedSpatialIoUMetric,
            )
            prog_evaluator = ProgrammaticEvaluator(
                metrics=[PSNRMetric(), MaskedPSNRMetric()],
                video_mask_metrics=[
                    SpatialIoUMetric(),
                    SpatiotemporalIoUMetric(),
                    WeightedSpatialIoUMetric(),
                ],
                velocity_metric=None,
            )
            prog_results = prog_evaluator.evaluate_keyframes(
                keyframe_dir=keyframe_dir,
                case_dir=case.case_dir,
                physics_duration=case.physics_duration,
                is_realworld=case.is_realworld,
            )
        except ImportError as e:
            logger.warning(f"Programmatic metrics unavailable: {e}")
        except Exception as e:
            logger.warning(f"Programmatic metrics failed: {e}")

    # --- Build TrackScores ---
    scores = TrackScores()
    scores.gen_annotations_removed = gemini_avg.get("gen_annotations_removed", 0.0)
    scores.object_consistency = gemini_avg.get("object_consistency", 0.0)
    scores.visual_quality = gemini_avg.get("visual_quality", 0.0)
    scores.motion_smoothness = gemini_avg.get("object_position_match", 0.0)  # map position→motion proxy
    scores.physics_accuracy = gemini_avg.get("physics_accuracy", 0.0)

    if prog_results:
        scores.gen_psnr = prog_results.get("psnr", -1.0)
        scores.gen_masked_psnr = prog_results.get("masked_psnr", -1.0)
        scores.gen_spatial_iou = prog_results.get("spatial_iou", -1.0)
        scores.gen_spatiotemporal_iou = prog_results.get("spatiotemporal_iou", -1.0)
        scores.gen_weighted_spatial_iou = prog_results.get("weighted_spatial_iou", -1.0)
        scores.gen_velocity_error = prog_results.get("velocity_error", -1.0)

    feedbacks = [r.get("gemini_scores", {}).get("feedback", "") for r in per_frame_results]
    scores.feedback = " | ".join(f for f in feedbacks if f)

    return {
        "per_frame": per_frame_results,
        "gemini_avg": gemini_avg,
        "programmatic": prog_results,
        "track_scores": scores.to_dict()["generation"],
        "generation_score": round(scores.generation_score(), 3),
    }


def _evaluate_single_keyframe_gemini(
    evaluator: VideoEvaluator,
    first_frame_data_url: str,
    gen_frame_path: str,
    gt_frame: np.ndarray,
    time_point: float,
) -> dict:
    """Send one (gen_frame, gt_frame) pair to Gemini for comparison.

    Args:
        gt_frame: GT frame as BGR ndarray (from PNG or video extraction).
    Returns raw score dict.
    """
    prompt = EVAL_PROMPT_GENERATION_KEYFRAME.format(time_point=time_point)

    gen_frame = cv2.imread(gen_frame_path, cv2.IMREAD_COLOR)
    if gen_frame is None:
        raise FileNotFoundError(f"Cannot read generated frame: {gen_frame_path}")

    gen_jpeg = frame_to_jpeg_bytes(gen_frame)
    gt_jpeg = frame_to_jpeg_bytes(gt_frame)

    content = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": "INITIAL ANNOTATED PHYSICS SCENE:"},
        {"type": "image_url", "image_url": {"url": first_frame_data_url}},
        {"type": "text", "text": f"GROUND TRUTH FRAME at t = {time_point:.2f}s:"},
        {"type": "image_url", "image_url": {"url": bytes_to_data_url(gt_jpeg)}},
        {"type": "text", "text": f"GENERATED FRAME at t = {time_point:.2f}s:"},
        {"type": "image_url", "image_url": {"url": bytes_to_data_url(gen_jpeg)}},
    ]

    messages = [{"role": "user", "content": content}]
    raw = evaluator.client.chat_completion(messages)

    # Parse JSON response
    data = parse_llm_json(raw)
    if data is None:
        logger.error(f"Failed to parse Gemini response: {raw[:300]}")
        return {f: 0.0 for f in GENERATION_KEYFRAME_FIELDS}

    result = {}
    for field in GENERATION_KEYFRAME_FIELDS + ["feedback"]:
        v = data.get(field, 0.0)
        if field == "feedback":
            result[field] = str(v) if v else ""
        else:
            try:
                val = float(v)
                result[field] = max(0.0, min(1.0, val)) if val >= 0 else -1.0
            except (ValueError, TypeError):
                result[field] = 0.0
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_image_path(results_dir: str, track: TrackType, variant: str) -> Optional[str]:
    """Resolve the expected image path for a (track, variant) in a case results dir."""
    # Try variant naming first, then fallback to bare name
    candidates = [
        os.path.join(results_dir, f"{track.value}_variant_{variant}.png"),
        os.path.join(results_dir, f"{track.value}.png"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def resolve_keyframe_dir(results_dir: str, variant: str) -> Optional[str]:
    """Resolve the keyframe directory for a generation variant."""
    candidates = [
        os.path.join(results_dir, f"generation_frames_variant_{variant}"),
        os.path.join(results_dir, "generation_frames"),
    ]
    for d in candidates:
        if os.path.isdir(d):
            return d
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate image-gen model outputs against physics infographics benchmark"
    )
    # Input modes
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--image", type=str,
        help="Path to a single image file (for one track)",
    )
    input_group.add_argument(
        "--keyframe-dir", type=str,
        help="Path to keyframe directory (for generation track)",
    )
    input_group.add_argument(
        "--results-dir", type=str,
        help="Results directory containing {track}_variant_{V}.png files",
    )

    parser.add_argument("--case", type=str, default="0",
                        help="Case ID(s): 0, 1, all, etc.")
    parser.add_argument("--track", type=str, required=True,
                        help="Track: perception_text, ..., generation, all, all_no_gen")
    parser.add_argument("--variant", type=str, default="A",
                        help="Prompt variant letter (default: A)")
    parser.add_argument("--demo-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="JSON output path (default: print to stdout)")
    parser.add_argument("--skip-gemini", action="store_true",
                        help="Skip Gemini eval for generation track (programmatic only)")
    parser.add_argument("--skip-programmatic", action="store_true",
                        help="Skip programmatic metrics for generation track")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # API client
    api_config = APIConfig()
    if not api_config.api_key:
        logger.error("API_KEY not set. Add to .env file.")
        sys.exit(1)

    if not args.demo_dir:
        sys.exit("--demo-dir is required (path to a case-parent dir, e.g. data/sim_subset/circular)")
    demo_dir = args.demo_dir
    opt_config = OptimizationConfig()
    client = APIClient(api_config)
    evaluator = VideoEvaluator(client, opt_config)

    # Parse cases and tracks
    case_ids = parse_cases(args.case)
    tracks = parse_tracks(args.track)

    all_results = {}

    for case_id in case_ids:
        try:
            case = CaseData.load(case_id, demo_dir)
        except FileNotFoundError as e:
            logger.error(str(e))
            continue

        for track in tracks:
            key = f"case_{case_id}_{track.value}"
            logger.info(f"Evaluating case {case_id}, track {track.value}")

            try:
                if track == TrackType.GENERATION:
                    # --- Generation track: keyframe eval ---
                    if args.keyframe_dir:
                        kf_dir = args.keyframe_dir
                    elif args.results_dir:
                        kf_dir = resolve_keyframe_dir(args.results_dir, args.variant)
                        if not kf_dir:
                            raise FileNotFoundError(
                                f"No keyframe dir found in {args.results_dir} for variant {args.variant}"
                            )
                    else:
                        raise ValueError("Generation track needs --keyframe-dir or --results-dir")

                    result = evaluate_generation_keyframes(
                        keyframe_dir=kf_dir,
                        case=case,
                        evaluator=evaluator,
                        skip_gemini=args.skip_gemini,
                        skip_programmatic=args.skip_programmatic,
                    )
                    all_results[key] = {
                        "case_id": case_id,
                        "track": track.value,
                        "keyframe_dir": kf_dir,
                        "generation_score": result["generation_score"],
                        "gemini_avg": result["gemini_avg"],
                        "programmatic": result["programmatic"],
                        "details": result["track_scores"],
                        "per_frame_count": len(result["per_frame"]),
                    }
                else:
                    # --- Tracks 1-4: single image eval ---
                    if args.image:
                        img_path = args.image
                    elif args.results_dir:
                        img_path = resolve_image_path(args.results_dir, track, args.variant)
                        if not img_path:
                            raise FileNotFoundError(
                                f"No image found for {track.value} variant {args.variant} in {args.results_dir}"
                            )
                    else:
                        raise ValueError("Need --image or --results-dir")

                    scores = evaluate_single_image(
                        image_path=img_path,
                        case=case,
                        track=track,
                        evaluator=evaluator,
                    )
                    track_score = scores.track_score(track)
                    all_results[key] = {
                        "case_id": case_id,
                        "track": track.value,
                        "image": img_path,
                        "score": round(track_score, 3),
                        "details": scores.to_dict()[track.value],
                        "feedback": scores.feedback,
                    }
                    logger.info(f"  Score: {track_score:.3f}")
                    logger.info(f"  Feedback: {scores.feedback[:200]}")

            except Exception as e:
                logger.exception(f"  Failed: {e}")
                all_results[key] = {
                    "case_id": case_id,
                    "track": track.value,
                    "error": str(e),
                }

    # Output
    results_json = json.dumps(all_results, indent=2, ensure_ascii=False)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(results_json)
        print(f"Results saved to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print("IMAGE EVALUATION RESULTS")
        print("=" * 60)
        print(results_json)


if __name__ == "__main__":
    main()
