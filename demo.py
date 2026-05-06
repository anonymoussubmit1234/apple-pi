#!/usr/bin/env python3
"""End-to-end demo for the Apple-π benchmark.

Steps:
  1. Download the full case folder for one demo case from the anonymous
     HuggingFace repo `AnonymousSubmitt/apple-pi`.
  2. Build inference prompts for that case (calls inference/build_bench.py).
  3. Print one (case, track) entry — what the model would receive.
  4. (Optional, --eval) If API endpoints are configured, run the LLM
     judge on the GT image as a sanity check.

Two preset cases — pick via --split:
    sim         sim_subset/circular/0                 (~280 MB)
    realworld   realworld_subset/Gravity/Freefall/1   (~5 MB)

Usage:
    python demo.py                      # sim demo, steps 1-3
    python demo.py --split realworld    # realworld demo, steps 1-3
    python demo.py --eval               # also run judge sanity check
"""

import argparse
import json
import os
import subprocess
import sys


HF_REPO_ID = "AnonymousSubmitt/apple-pi"

# Preset cases per split.
DEMO_CASES = {
    "sim": {
        "case_path": "sim_subset/circular/0",
        "case_parent_rel": "sim_subset/circular",  # parent dir of <case_id>
        "case_id": "0",
        "size_mb": 280,
        "build_split": "sim",
    },
    "realworld": {
        "case_path": "realworld_subset/Gravity/Freefall/1",
        "case_parent_rel": "realworld_subset/Gravity/Freefall",
        "case_id": "1",
        "size_mb": 5,
        "build_split": "real",
    },
}


def download_one_case(local_dir: str, sample_case_path: str, size_mb: int):
    """Pull the full case folder for one demo case.

    The full folder is needed because the evaluation pipeline reads the
    GT rgb sequence (or video.mp4 for realworld), mask/, and
    instance_segmentation/ for programmatic metrics (PSNR, IoU,
    segmentation IoU, MoGe-2 velocity error).
    """
    from huggingface_hub import snapshot_download
    print(f"[1/4] downloading 1 full case from {HF_REPO_ID} → {local_dir}")
    print(f"      (~{size_mb} MB — includes GT video/frames, masks, segmentation)")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=[
            f"{sample_case_path}/**",
            "README.md",
        ],
    )


def build_demo_prompts(local_dir: str, build_split: str):
    """Run inference/build_bench.py limited to the demo case."""
    print(f"[2/4] building prompts → {local_dir}/bench/")
    here = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        sys.executable,
        os.path.join(here, "inference", "build_bench.py"),
        "--root", local_dir,
        "--split", build_split,
        "--limit", "1",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print("[error] build_bench failed:")
        print(r.stderr)
        sys.exit(1)


def show_one_entry(local_dir: str, track: str, kind: str = "unified"):
    """Print one (case, track) entry from the generated bench JSON."""
    bench_file = os.path.join(local_dir, "bench", f"{kind}.json")
    if not os.path.exists(bench_file):
        print(f"[error] missing {bench_file}")
        return None
    with open(bench_file) as f:
        entries = json.load(f)
    matching = [e for e in entries if e["track"] == track]
    if not matching:
        print(f"[error] no {track} entries in {bench_file}")
        return None
    e = matching[0]
    print(f"[3/4] sample entry from {bench_file}")
    print("─" * 70)
    print(f"  case:     {e['case']}")
    print(f"  track:    {e['track']}")
    print(f"  image:    {e['image']}    (relative to dataset root)")
    print(f"  duration: {e['duration']}")
    print(f"  size:     {e['size']}")
    if "time_point" in e:
        print(f"  time_pt:  {e['time_point']}  (keyframe {e['prompt_idx']})")
    print(f"\n  prompt (first 500 chars):")
    print("  " + e["prompt"][:500].replace("\n", "\n  ") + ("..." if len(e["prompt"]) > 500 else ""))
    print("─" * 70)
    return e


def maybe_run_eval(local_dir: str, case_parent_rel: str, case_id: str):
    """If env vars are set, run a sanity-check eval on the GT image.

    Pretends the white-bg GT image *is* the model output, then asks the
    judge to score it on perception_text. A correctly-configured judge
    should give a high score (~1.0) since the input matches the
    reference perfectly.
    """
    if not (os.environ.get("API_KEY") and os.environ.get("LLM_CHAT_URL")
            and os.environ.get("LLM_CHAT_MODEL")):
        print("[4/4] (skipped) eval requires API_KEY + LLM_CHAT_URL + LLM_CHAT_MODEL env vars.")
        print("      see README for provider-configuration examples.")
        return

    print("[4/4] running judge sanity check on the GT image…")
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    sys.path.insert(0, os.path.join(here, "eval"))
    from config import APIConfig, CaseData, TrackType, OptimizationConfig
    from api_client import APIClient
    from evaluator import VideoEvaluator
    from eval_image import evaluate_single_image

    case_parent = os.path.join(local_dir, case_parent_rel)
    case = CaseData.load(int(case_id), case_parent)
    api_cfg = APIConfig()
    opt_cfg = OptimizationConfig()
    client = APIClient(api_cfg)
    evaluator = VideoEvaluator(client, opt_cfg)

    gt_path = case.white_bg_first_frame_path
    if not gt_path:
        print("      (no white-bg reference for this case; skipping)")
        return
    scores = evaluate_single_image(
        image_path=gt_path,
        case=case,
        track=TrackType.PERCEPTION_TEXT,
        evaluator=evaluator,
    )
    track_score = scores.track_score(TrackType.PERCEPTION_TEXT)
    print(f"      track_score (GT-as-output, perception_text): {track_score:.3f}")
    print(f"      feedback: {scores.feedback[:200]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="sim", choices=["sim", "realworld"],
                        help="Which preset case to demo (default: sim)")
    parser.add_argument("--data-dir", default="./data",
                        help="Local cache for the HF dataset")
    parser.add_argument("--track", default="perception_text",
                        choices=["perception_text", "perception_graphic",
                                 "comprehension_text", "comprehension_graphic",
                                 "generation"],
                        help="Which track's prompt to display")
    parser.add_argument("--eval", action="store_true",
                        help="Also run a judge sanity check (requires API_KEY + LLM_CHAT_URL)")
    args = parser.parse_args()

    cfg = DEMO_CASES[args.split]
    print(f"[demo] split={args.split}, case={cfg['case_path']}")

    download_one_case(args.data_dir, cfg["case_path"], cfg["size_mb"])
    build_demo_prompts(args.data_dir, cfg["build_split"])
    show_one_entry(args.data_dir, args.track, kind="unified")

    if args.eval:
        maybe_run_eval(args.data_dir, cfg["case_parent_rel"], cfg["case_id"])
    else:
        print("[4/4] (skipped) pass --eval to run the judge sanity check.")

    print("\nDemo complete.")
    print(f"  See {args.data_dir}/bench/unified.json   — full unified prompts")
    print(f"      {args.data_dir}/bench/videogen.json — full video-gen prompts")
    print("To run inference: pipe each entry to your model and save outputs")
    print("  under $APPLE_PI_ROOT/<your_model>/ following the naming convention")
    print("  documented in README.md, then run:")
    print("    python eval/run_eval_apple_pi.py --model <your_model> --track all")


if __name__ == "__main__":
    main()
