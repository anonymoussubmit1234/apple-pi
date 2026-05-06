#!/usr/bin/env python3
"""Standalone evaluation script for generated videos.

Evaluate any video (from any model) against the physics infographics
benchmark ground truth. Supports all 5 subtracks.

Usage examples:
    # Evaluate a single video on a specific track
    python eval_video.py --video output.mp4 --case 0 --track perception_text

    # Evaluate on comprehension_graphic (uses target_time + track ref)
    python eval_video.py --video output.mp4 --case 0 --track comprehension_graphic

    # Evaluate all tracks for a video
    python eval_video.py --video output.mp4 --case 0 --track all

    # Batch: evaluate videos in a directory (one per case)
    python eval_video.py --video-dir outputs/ --track generation
    # expects outputs/case_0.mp4, outputs/case_1.mp4, etc.
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
# seg_evaluator (which pulls in torch + SAM3) is imported lazily inside the
# graphic-track branches below — keeps perception_text / comprehension_text
# runs lightweight.
from utils import (
    extract_last_frame_from_file,
    image_to_data_url,
)

logger = logging.getLogger(__name__)


def evaluate_video(
    video_path: str,
    case: CaseData,
    track: TrackType,
    evaluator: VideoEvaluator,
    num_eval_frames: int = -1,
) -> TrackScores:
    """Evaluate a single video file against a track.

    Args:
        video_path: path to the .mp4 video to evaluate.
        case: CaseData with ground truth info.
        track: which subtrack to evaluate.
        evaluator: VideoEvaluator instance.
        num_eval_frames: number of frames to sample from the video.

    Returns:
        TrackScores with per-dimension scores and feedback.
    """
    gen_video_bytes: Optional[bytes] = None
    if track != TrackType.GENERATION:
        gen_frames = extract_last_frame_from_file(video_path)
    else:
        # Generation track: feed the whole mp4 instead of per-frame sampling.
        with open(video_path, "rb") as f:
            gen_video_bytes = f.read()
        gen_frames = []

    # Load first frame (annotated)
    first_frame_data_url = image_to_data_url(case.first_frame_path)
    first_frame_white_bg_data_url = None
    if track in [TrackType.PERCEPTION_TEXT, TrackType.PERCEPTION_GRAPHIC] and case.white_bg_first_frame_path:
        first_frame_white_bg_data_url = image_to_data_url(case.white_bg_first_frame_path)
        
    first_frame_white_bg_obj_data_url = None
    if track == TrackType.PERCEPTION_GRAPHIC and case.white_bg_obj_first_frame_path:
        first_frame_white_bg_obj_data_url = image_to_data_url(case.white_bg_obj_first_frame_path)

    # Load GT frames
    gt_frames = None

    # Formula info for comprehension_text
    formula_info = None
    if track == TrackType.COMPREHENSION_TEXT:
        try:
            cached = load_case_formula_info(case.case_dir)
            if not cached:
                raise FileNotFoundError(
                    f"Missing cached formula info for case {case.case_id}: "
                    f"{os.path.join(case.case_dir, 'formula_info.json')}"
                )
            formula_info = {
                "correct_letter": str(cached["correct_letter"]),
                "correct_formula": str(cached["correct_formula"]),
                "annotation": str(cached.get("annotation", case.annotation_text)),
            }
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
            raise RuntimeError(
                f"Failed to load cached formula info for case {case.case_id} "
                f"from {os.path.join(case.case_dir, 'formula_info.json')}: {e}"
            ) from e
    
    # Track reference for comprehension_graphic
    track_ref_data_url = None
    if track == TrackType.COMPREHENSION_GRAPHIC and case.track_image_path:
        track_ref_data_url = image_to_data_url(case.track_image_path)

    scores = evaluator.evaluate(
        track=track,
        first_frame_data_url=first_frame_data_url,
        gen_frames=gen_frames,
        gen_video_bytes=gen_video_bytes,
        gt_frames=gt_frames,
        track_ref_data_url=track_ref_data_url,
        formula_info=formula_info,
        first_frame_white_bg_data_url=first_frame_white_bg_data_url,
        first_frame_white_bg_obj_data_url=first_frame_white_bg_obj_data_url,
    )

    if track == TrackType.PERCEPTION_GRAPHIC:
        try:
            import cv2 as _cv2
            from seg_evaluator import compute_segmentation_iou
            gen_image_rgb = _cv2.cvtColor(gen_frames[0], _cv2.COLOR_BGR2RGB)
            iou = compute_segmentation_iou(
                gen_image=gen_image_rgb,
                case_dir=case.case_dir,
                is_realworld=case.is_realworld,
            )
            # from seg_evaluator import visualize_segmentation_debug
            # visualize_segmentation_debug(gen_image_rgb, case.case_dir, case.is_realworld, "segmentation_debug.png")
            scores.segmentation_iou = iou
        except Exception as e:
            logger.warning(f"SAM3 segmentation IoU failed: {e}")
            scores.segmentation_iou = 0.0

    if track == TrackType.COMPREHENSION_GRAPHIC:
        try:
            import cv2 as _cv2
            from seg_evaluator import compute_segmentation_iou
            gen_image_rgb = _cv2.cvtColor(gen_frames[0], _cv2.COLOR_BGR2RGB)
            iou = compute_segmentation_iou(
                gen_image=gen_image_rgb,
                case_dir=case.case_dir,
                is_realworld=case.is_realworld,
                mask_subdir="instantaneous_velocity",
            )
            scores.comp_graphic_segmentation_iou = iou
        except Exception as e:
            logger.warning(f"SAM3 segmentation IoU for comprehension_graphic failed: {e}")
            scores.comp_graphic_segmentation_iou = 0.0

    if track == TrackType.GENERATION:
        prog_results = compute_programmatic_metrics(
            video_path=video_path,
            case=case,
        )
        if prog_results:
            # Fidelity metrics: pixel-level reconstruction quality
            scores.gen_psnr = prog_results.get("psnr", -1.0)
            scores.gen_masked_psnr = prog_results.get("masked_psnr", -1.0)
            # Physics metrics: object spatial accuracy and velocity correctness
            scores.gen_spatial_iou = prog_results.get("spatial_iou", -1.0)
            scores.gen_spatiotemporal_iou = prog_results.get("spatiotemporal_iou", -1.0)
            scores.gen_weighted_spatial_iou = prog_results.get("weighted_spatial_iou", -1.0)
            scores.gen_velocity_error = prog_results.get("velocity_error", -1.0)

    return scores


def compute_programmatic_metrics(
    video_path: str,
    case: CaseData,
    save_mask_path: Optional[str] = None,
    save_depth_dir: Optional[str] = None,
    save_velocity_dir: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """Compute programmatic metrics (PSNR, IoU, velocity) for generation track.

    Lazily imports the Evaluator and metric classes from programmatic_metric.py.
    Requires SAM3 and MoGe dependencies for mask/velocity metrics.

    Args:
        video_path: Path to the generated video file.
        case: CaseData instance for this case.

    Returns dict of metric_name -> score, or None if dependencies unavailable.
    """
    try:
        from programmatic_metric import (
            Evaluator as ProgrammaticEvaluator,
            PSNRMetric,
            MaskedPSNRMetric,
            SpatialIoUMetric,
            SpatiotemporalIoUMetric,
            WeightedSpatialIoUMetric,
            VelocityMetric,
        )
    except ImportError as e:
        logger.warning(
            f"Cannot import programmatic metrics (missing dependencies?): {e}"
        )
        return None

    frame_metrics = [PSNRMetric(), MaskedPSNRMetric()]
    mask_metrics = [
        SpatialIoUMetric(),
        SpatiotemporalIoUMetric(),
        WeightedSpatialIoUMetric(),
    ]
    velocity_metric = VelocityMetric() if not case.is_realworld else None

    prog_evaluator = ProgrammaticEvaluator(
        frame_metrics,
        mask_metrics,
        velocity_metric=velocity_metric,
    )

    try:
        results = prog_evaluator.evaluate_video(
            video_path=video_path,
            case_dir=case.case_dir,
            physics_duration=case.physics_duration,
            is_realworld=case.is_realworld,
            save_mask_path=save_mask_path,
            save_depth_dir=save_depth_dir,
            save_velocity_dir=save_velocity_dir,
        )
        return results
    except Exception as e:
        logger.exception(f"Programmatic metric computation failed: {e}")
        return None



def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated videos against physics infographics benchmark"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--video",
        type=str,
        help="Path to a single video file to evaluate",
    )
    group.add_argument(
        "--video-dir",
        type=str,
        help="Directory with videos named case_0.mp4, case_1.mp4, etc.",
    )
    parser.add_argument(
        "--case",
        type=str,
        default="0",
        help="Case ID(s): 0, 1, 2, 3, 0,1,2, or all (default: 0)",
    )
    parser.add_argument(
        "--track",
        type=str,
        required=True,
        help="Track: perception_text, perception_graphic, "
        "comprehension_text, comprehension_graphic, generation, "
        "or groups: perception, comprehension, all",
    )
    parser.add_argument(
        "--demo-dir",
        type=str,
        default=None,
        help="Path to demo data directory (default: ./demo)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=-1,
        help="Number of frames to sample from video (default: -1, all frames)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results (default: print to stdout)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Check API key
    api_config = APIConfig()
    if not api_config.api_key:
        logger.error(
            "API_KEY not set. "
            "Add it to .env file: API_KEY=your_key_here"
        )
        sys.exit(1)

    if not args.demo_dir:
        sys.exit("--demo-dir is required (path to a case-parent dir, e.g. data/sim_subset/circular)")
    demo_dir = args.demo_dir
    opt_config = OptimizationConfig(num_eval_frames=args.num_frames)
    client = APIClient(api_config)
    evaluator = VideoEvaluator(client, opt_config)

    # Parse cases and tracks
    case_ids = parse_cases(args.case)
    tracks = parse_tracks(args.track)

    # Run evaluations
    all_results = {}

    for case_id in case_ids:
        try:
            case = CaseData.load(case_id, demo_dir)
        except FileNotFoundError as e:
            logger.error(str(e))
            continue

        # Determine video path
        if args.video:
            video_path = args.video
        else:
            video_path = os.path.join(args.video_dir, f"case_{case_id}.mp4")

        if not os.path.exists(video_path):
            logger.error(f"Video not found: {video_path}")
            continue

        logger.info(f"Evaluating case {case_id}: {video_path}")

        for track in tracks:
            logger.info(f"  Track: {track.value}")
            try:
                scores = evaluate_video(
                    video_path=video_path,
                    case=case,
                    track=track,
                    evaluator=evaluator,
                    num_eval_frames=args.num_frames,
                )
                track_score = scores.track_score(track)
                key = f"case_{case_id}_{track.value}"
                all_results[key] = {
                    "case_id": case_id,
                    "track": track.value,
                    "video": video_path,
                    "score": round(track_score, 3),
                    "details": {
                        k: v
                        for k, v in scores.to_dict()[track.value].items()
                    },
                    "feedback": scores.feedback,
                }
                logger.info(f"    Score: {track_score:.3f}")
                logger.info(f"    Feedback: {scores.feedback[:200]}")
            except Exception as e:
                logger.exception(f"    Failed: {e}")
                all_results[f"case_{case_id}_{track.value}"] = {
                    "case_id": case_id,
                    "track": track.value,
                    "video": video_path,
                    "error": str(e),
                }

    # Output results
    results_json = json.dumps(all_results, indent=2, ensure_ascii=False)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(results_json)
        print(f"Results saved to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(results_json)


if __name__ == "__main__":
    main()
