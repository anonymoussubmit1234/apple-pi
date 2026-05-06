#!/usr/bin/env python3
"""Generate images with Nano Banana Pro for the physics infographics benchmark.

For each case × track, generates image(s) and saves as .png + wraps into
single-frame .mp4 for compatibility with eval_video.py.

Usage:
    # Single case + track
    python generate_nano_banana.py --case 0 --track perception_text

    # All 4 non-generation tracks
    python generate_nano_banana.py --case all --track all

    # Generation track: generates 8 images (1fps) for each case
    python generate_nano_banana.py --case 0 --track generation

    # Prompts come from prompt_templates.py (UM_PROMPT_TEMPLATES)
    # No variant selection needed
"""


# Make root-level shared modules (api_client, config, utils) importable.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional

from api_client import APIClient
from config import (
    APIConfig,
    BASE_DIR,
    CaseData,
    TrackType,
    load_case_formula_info,
    parse_cases,
    parse_tracks,
)
from prompt_templates import UM_PROMPT_TEMPLATES
from utils import image_to_base64

logger = logging.getLogger(__name__)

LETTERS = ["A", "B", "C", "D"]

# Prompts come from prompt_templates.py (single source of truth).
# No variants — always use UM_PROMPT_TEMPLATES directly.


# ── Helpers ──────────────────────────────────────────────────────────

def build_prompt(
    track: TrackType,
    case: CaseData,
    time_point: Optional[float] = None,
) -> str:
    """Build prompt for a given track + case from UM_PROMPT_TEMPLATES."""
    template = UM_PROMPT_TEMPLATES[track]

    kwargs = {}

    if track == TrackType.COMPREHENSION_TEXT:
        formula_info = load_case_formula_info(case.case_dir)
        if formula_info and "choices" in formula_info:
            choices = formula_info["choices"]
        else:
            raise RuntimeError(
                f"Missing formula_info.json with 'choices' for case {case.case_id}: "
                f"{case.case_dir}"
            )
        lines = [f"{LETTERS[i]}) {formula}" for i, formula in enumerate(choices)]
        kwargs["formula_choices"] = "\n".join(lines)

    if track == TrackType.COMPREHENSION_GRAPHIC:
        kwargs["target_time"] = case.target_time

    if track == TrackType.GENERATION:
        kwargs["time_point"] = time_point

    return template.format(**kwargs)


def save_image(img_bytes: bytes, path: str) -> str:
    """Save image bytes to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(img_bytes)
    return os.path.abspath(path)


# ── Main generation logic ────────────────────────────────────────────

def generate_single_image_track(
    client: APIClient,
    case: CaseData,
    track: TrackType,
    output_dir: str,
) -> dict:
    """Generate a single image for perception/comprehension tracks."""
    prompt = build_prompt(track, case)
    first_frame_b64, first_frame_mime = image_to_base64(case.first_frame_path)

    logger.info(f"Generating {track.value} | case {case.case_id}")
    logger.info(f"Prompt ({len(prompt.split())} words): {prompt[:120]}...")

    img_bytes, mime = client.generate_image(
        prompt=prompt,
        input_image_b64=first_frame_b64,
        input_image_mime=first_frame_mime,
    )

    # Save PNG
    ext = "jpg" if "jpeg" in mime else "png"
    png_path = os.path.join(output_dir, f"{track.value}.{ext}")
    save_image(img_bytes, png_path)

    logger.info(f"Saved: {png_path}")
    return {
        "case_id": case.case_id,
        "track": track.value,
        "prompt": prompt,
        "png_path": png_path,
        "timestamp": datetime.now().isoformat(),
    }


def generate_generation_track(
    client: APIClient,
    case: CaseData,
    output_dir: str,
    fps: int = 2,
) -> dict:
    """Generate multiple images for the generation track at 2fps.

    Sampling rate is 2fps over the physics duration:
      2s  → 4 frames  (t=0.5, 1.0, 1.5, 2.0)
      10s → 20 frames (t=0.5, 1.0, 1.5, ..., 10.0)

    Each frame is generated with a separate API call (one prompt per time point).
    Saved as individual PNGs for per-frame evaluation.
    Also combined into mp4 for backward compatibility.
    """
    first_frame_b64, first_frame_mime = image_to_base64(case.first_frame_path)

    # Use physics duration (not GT video duration which may be longer)
    duration = case.physics_duration
    num_frames = int(duration * fps)

    # Time points at 2fps: 0.5, 1.0, 1.5, ..., duration
    time_points = [
        round((i + 1) / fps, 2)
        for i in range(num_frames)
    ]

    logger.info(
        f"Generating generation track | case {case.case_id} | "
        f"GT duration={duration}s | {num_frames} frames at t = {time_points}"
    )

    frame_dir = os.path.join(output_dir, "generation_frames")
    os.makedirs(frame_dir, exist_ok=True)
    frame_paths = []

    for i, t in enumerate(time_points):
        prompt = build_prompt(TrackType.GENERATION, case, time_point=t)
        logger.info(f"  Frame {i+1}/{num_frames} (t={t}s): {prompt[:80]}...")

        img_bytes, mime = client.generate_image(
            prompt=prompt,
            input_image_b64=first_frame_b64,
            input_image_mime=first_frame_mime,
        )

        ext = "jpg" if "jpeg" in mime else "png"
        frame_path = os.path.join(frame_dir, f"frame_{i:02d}_t{t}s.{ext}")
        save_image(img_bytes, frame_path)
        frame_paths.append(frame_path)

    logger.info(f"Saved: {len(frame_paths)} frames under {frame_dir}")
    return {
        "case_id": case.case_id,
        "track": "generation",
        "gt_duration": duration,
        "fps": fps,
        "num_frames": num_frames,
        "time_points": time_points,
        "frame_paths": frame_paths,
        "frame_dir": frame_dir,
        "timestamp": datetime.now().isoformat(),
    }


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate images with Nano Banana Pro for physics benchmark"
    )
    parser.add_argument("--case", type=str, default="0",
                        help="Case ID(s): 0,1,2,3 or all")
    parser.add_argument("--track", type=str, default="perception_text",
                        help="Track: perception_text, perception_graphic, "
                             "comprehension_text, comprehension_graphic, "
                             "generation, or groups: perception, comprehension, all")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (required)")
    parser.add_argument("--gen-fps", type=int, default=2,
                        help="FPS for generation track (default: 2)")
    parser.add_argument("--demo-dir", type=str, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Setup
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    api_config = APIConfig()
    if not api_config.api_key:
        logger.error("API_KEY not set")
        sys.exit(1)

    client = APIClient(api_config)
    if not args.demo_dir:
        sys.exit("--demo-dir is required (path to a case-parent dir, e.g. data/sim_subset/circular)")
    demo_dir = args.demo_dir
    if not args.output_dir:
        sys.exit("--output-dir is required")
    output_base = args.output_dir

    case_ids = parse_cases(args.case)
    tracks = parse_tracks(args.track, include_generation=False)
    results = []

    for case_id in case_ids:
        try:
            case = CaseData.load(case_id, demo_dir)
        except FileNotFoundError as e:
            logger.error(str(e))
            continue

        case_dir = os.path.join(output_base, f"case_{case_id}")

        for track in tracks:
            try:
                if track == TrackType.GENERATION:
                    result = generate_generation_track(
                        client, case, case_dir,
                        fps=args.gen_fps,
                    )
                else:
                    result = generate_single_image_track(
                        client, case, track, case_dir,
                    )
                results.append(result)
                print(f"OK: case {case_id} | {track.value}")
            except Exception as e:
                logger.exception(f"Failed: case {case_id} | {track.value}")
                results.append({
                    "case_id": case_id,
                    "track": track.value,
                    "error": str(e),
                })

    # Save summary
    summary_path = os.path.join(output_base, "generation_log.json")
    os.makedirs(output_base, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
