#!/usr/bin/env python3
"""Single-pass Veo3 video generation for one or more cases (no optimization loop).

Usage:
    python generate_veo3_single.py --case 0 --track all \
        --demo-dir <case-parent-dir> --output-dir <out-dir>
"""


# Make root-level shared modules (api_client, config, utils) importable.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import os
import sys

from api_client import APIClient
from config import (
    APIConfig,
    BASE_DIR,
    CaseData,
    TrackType,
    build_formula_choices_seeded,
    load_case_formula_info,
    parse_cases,
    parse_tracks,
)
from prompt_templates import VM_PROMPT_TEMPLATES
from utils import image_to_base64, save_video

logger = logging.getLogger(__name__)


def build_veo3_prompt(track: TrackType, case: CaseData) -> str:
    """Build Veo3 prompt for a given track + case."""
    template = VM_PROMPT_TEMPLATES[track]

    if track == TrackType.COMPREHENSION_TEXT:
        # Use formula choices from formula_info.json (same as NB)
        formula_info = load_case_formula_info(case.case_dir)
        if formula_info and "choices" in formula_info:
            choices = formula_info["choices"]
        else:
            raise RuntimeError(
                f"Missing formula_info.json with 'choices' for case {case.case_id}"
            )
        letters = ["A", "B", "C", "D"]
        lines = [f"  {letters[i]}) {f}" for i, f in enumerate(choices)]
        choices_str = "\n".join(lines)
        return template.format(formula_choices=choices_str)

    elif track == TrackType.COMPREHENSION_GRAPHIC:
        return template.format(target_time=case.target_time)

    elif track == TrackType.GENERATION:
        return template.format(physics_duration=int(case.physics_duration))

    return template


def main():
    parser = argparse.ArgumentParser(
        description="Single-pass Veo3 video generation (no optimization)"
    )
    parser.add_argument("--case", type=str, default="0")
    parser.add_argument("--track", type=str, default="generation")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--demo-dir", type=str, default=None)
    parser.add_argument("--fast", action="store_true", default=True)
    parser.add_argument("--no-fast", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")

    api_config = APIConfig()
    if not api_config.api_key:
        print("API_KEY not set")
        sys.exit(1)

    client = APIClient(api_config)
    if not args.demo_dir:
        sys.exit("--demo-dir is required (path to a case-parent dir, e.g. data/sim_subset/circular)")
    demo_dir = args.demo_dir
    if not args.output_dir:
        sys.exit("--output-dir is required")
    output_dir = args.output_dir
    use_fast = not args.no_fast

    case_ids = parse_cases(args.case)
    tracks = parse_tracks(args.track)

    results = {}

    for case_id in case_ids:
        try:
            case = CaseData.load(case_id, demo_dir)
        except Exception as e:
            logger.error(f"Failed to load case {case_id}: {e}")
            continue

        for track in tracks:
            case_out = os.path.join(output_dir, f"case_{case_id}")
            os.makedirs(case_out, exist_ok=True)

            mp4_path = os.path.join(case_out, f"{track.value}.mp4")
            if os.path.exists(mp4_path):
                logger.info(f"Skipping case {case_id} | {track.value} (already exists)")
                continue

            prompt = build_veo3_prompt(track, case)
            logger.info(f"Case {case_id} | {track.value} | prompt: {prompt[:100]}...")

            # Use first frame as conditioning image
            img_b64, img_mime = image_to_base64(case.first_frame_path)

            try:
                task_id = client.submit_video_generation(
                    prompt=prompt,
                    image_b64=img_b64,
                    image_mime=img_mime,
                    fast=use_fast,
                )
                video_bytes = client.poll_video_task(task_id, fast=use_fast)
                save_video(video_bytes, mp4_path)
                logger.info(f"OK: case {case_id} | {track.value} → {mp4_path}")
                print(f"OK: case {case_id} | {track.value}")
                results[f"case_{case_id}_{track.value}"] = {
                    "status": "ok",
                    "path": mp4_path,
                }
            except Exception as e:
                logger.error(f"FAILED: case {case_id} | {track.value}: {e}")
                print(f"FAILED: case {case_id} | {track.value}: {e}")
                results[f"case_{case_id}_{track.value}"] = {
                    "status": "failed",
                    "error": str(e),
                }

    # Save results log
    log_path = os.path.join(output_dir, "generation_log.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {log_path}")


if __name__ == "__main__":
    main()
