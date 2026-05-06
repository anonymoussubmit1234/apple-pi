#!/usr/bin/env python3
"""Build inference-driver JSONs (videogen.json + unified.json) for the
Apple-π benchmark.

For each case, reads the case's metadata files (formula choices, physics
duration, target_time, first-frame image path) and emits one entry per
(case, track) for two model families:

  - VM (video generation: Veo3 / Sora style)        → bench/videogen.json
  - UM (unified image generation: Nano-Banana style) → bench/unified.json

Layout expected under <root>/ (matches the HuggingFace dataset layout
for AnonymousSubmitt/apple-pi):

    sim_subset/<ptype>/<case_id>/
        initial_state/rgb_0000.png
        formula_info.json                      # contains "choices"
        physics_duration.txt                   # float
        instantaneous_velocity/velocity.json   # contains "time_seconds"

    realworld_subset/<pillar>/<task>/<case_id>/      (2-level taxonomy)
        ...same case files as above
    realworld_subset/Composition/<case_id>/          (1-level)
        ...same case files as above

Usage:
    python inference/build_bench.py --root ./data --out-dir ./bench
    python inference/build_bench.py --root ./data --split sim     # sim only
    python inference/build_bench.py --root ./data --split real    # realworld only
"""

# Make root-level shared modules (api_client, config, utils) importable.
import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from prompt_templates import (
    VM_PROMPT_TEMPLATES,
    UM_PROMPT_TEMPLATES,
)
from config import TrackType

LETTERS = ["A", "B", "C", "D"]


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def format_choices(choices: List[str]) -> str:
    return "\n".join(f"  {LETTERS[i]}) {c}" for i, c in enumerate(choices))


def read_float(path: str) -> Optional[float]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return float(f.read().strip())
    except (ValueError, IOError):
        return None


def read_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (ValueError, IOError):
        return None


def case_files(case_dir: str) -> dict:
    """Resolve all metadata files for one case directory."""
    finfo = read_json(os.path.join(case_dir, "formula_info.json")) or {}
    velocity = read_json(os.path.join(case_dir, "instantaneous_velocity",
                                      "velocity.json")) or {}
    physics_duration = read_float(os.path.join(case_dir, "physics_duration.txt"))
    return {
        "image_rel": "initial_state/rgb_0000.png",
        "choices": finfo.get("choices"),
        "target_time": velocity.get("time_seconds"),
        "physics_duration": physics_duration,
    }


def enumerate_sim_cases(sim_root: str) -> List[Tuple[str, str, str]]:
    """Yield (id_prefix, ptype, case_id) for every sim case under <root>/sim_subset/."""
    out = []
    if not os.path.isdir(sim_root):
        return out
    for ptype in sorted(os.listdir(sim_root)):
        ptype_dir = os.path.join(sim_root, ptype)
        if not os.path.isdir(ptype_dir):
            continue
        for case in sorted(os.listdir(ptype_dir),
                           key=lambda x: int(x) if x.isdigit() else 10**9):
            if not case.isdigit():
                continue
            out.append(("sim", ptype, case))
    return out


def enumerate_realworld_cases(rw_root: str) -> List[Tuple[str, str, str, str]]:
    """Yield (id_prefix, pillar, task, case_id) for realworld_subset/."""
    out = []
    if not os.path.isdir(rw_root):
        return out
    for pillar in sorted(os.listdir(rw_root)):
        pillar_dir = os.path.join(rw_root, pillar)
        if not os.path.isdir(pillar_dir):
            continue
        # Composition: 1-level (case_id directly under pillar)
        # Other pillars: 2-level (pillar/task/case_id)
        first_children = sorted(os.listdir(pillar_dir))
        if all(c.isdigit() for c in first_children if not c.startswith(".")):
            # 1-level: pillar/<case_id>
            for case in first_children:
                if case.isdigit():
                    out.append(("realworld", pillar, "Composition", case))
        else:
            # 2-level: pillar/<task>/<case_id>
            for task in first_children:
                task_dir = os.path.join(pillar_dir, task)
                if not os.path.isdir(task_dir):
                    continue
                for case in sorted(os.listdir(task_dir),
                                   key=lambda x: int(x) if x.isdigit() else 10**9):
                    if case.isdigit():
                        out.append(("realworld", pillar, task, case))
    return out


# ─────────────────────────────────────────────────────────────────────────
# Entry builders
# ─────────────────────────────────────────────────────────────────────────

def build_entry(case_path: str, image_rel_path: str, track: str, prompt: str,
                duration: int, *, prompt_idx: Optional[int] = None,
                time_point: Optional[float] = None) -> Dict[str, Any]:
    e = {
        "case": case_path,
        "track": track,
        "prompt": prompt,
        "image": f"{case_path}/{image_rel_path}",
        "duration": duration,
        "size": "1280x720",
    }
    if prompt_idx is not None:
        e["prompt_idx"] = prompt_idx
        e["time_point"] = time_point
    return e


def emit_for_case(case_path: str, info: dict,
                  videogen: List[dict], unified: List[dict],
                  warnings: List[str]):
    """Append all (track, prompt) entries for this case to videogen + unified."""
    img_rel = info["image_rel"]

    # ── Video-gen (VM) ──
    videogen.append(build_entry(case_path, img_rel, "perception_text",
                                VM_PROMPT_TEMPLATES[TrackType.PERCEPTION_TEXT],
                                duration=8))
    videogen.append(build_entry(case_path, img_rel, "perception_graphic",
                                VM_PROMPT_TEMPLATES[TrackType.PERCEPTION_GRAPHIC],
                                duration=8))
    if info.get("choices"):
        prompt = VM_PROMPT_TEMPLATES[TrackType.COMPREHENSION_TEXT].format(
            formula_choices=format_choices(info["choices"])
        )
        videogen.append(build_entry(case_path, img_rel, "comprehension_text",
                                    prompt, duration=8))
    else:
        warnings.append(f"skip comprehension_text (VM) for {case_path}: no formula choices")
    if info.get("target_time") is not None:
        prompt = VM_PROMPT_TEMPLATES[TrackType.COMPREHENSION_GRAPHIC].format(
            target_time=info["target_time"]
        )
        videogen.append(build_entry(case_path, img_rel, "comprehension_graphic",
                                    prompt, duration=8))
    else:
        warnings.append(f"skip comprehension_graphic (VM) for {case_path}: no target_time")
    if info.get("physics_duration") is not None:
        pd_str = f"{info['physics_duration']:g}"
        prompt = VM_PROMPT_TEMPLATES[TrackType.GENERATION].format(physics_duration=pd_str)
        videogen.append(build_entry(case_path, img_rel, "generation",
                                    prompt, duration=8))
    else:
        warnings.append(f"skip generation (VM) for {case_path}: no physics_duration")

    # ── Unified image-gen (UM) ──
    unified.append(build_entry(case_path, img_rel, "perception_text",
                               UM_PROMPT_TEMPLATES[TrackType.PERCEPTION_TEXT],
                               duration=1))
    unified.append(build_entry(case_path, img_rel, "perception_graphic",
                               UM_PROMPT_TEMPLATES[TrackType.PERCEPTION_GRAPHIC],
                               duration=1))
    if info.get("choices"):
        prompt = UM_PROMPT_TEMPLATES[TrackType.COMPREHENSION_TEXT].format(
            formula_choices=format_choices(info["choices"])
        )
        unified.append(build_entry(case_path, img_rel, "comprehension_text",
                                   prompt, duration=1))
    if info.get("target_time") is not None:
        prompt = UM_PROMPT_TEMPLATES[TrackType.COMPREHENSION_GRAPHIC].format(
            target_time=info["target_time"]
        )
        unified.append(build_entry(case_path, img_rel, "comprehension_graphic",
                                   prompt, duration=1))
    # UM generation: 2 fps over [0.5, physics_duration]
    if info.get("physics_duration") is not None and info["physics_duration"] > 0:
        n_frames = int(info["physics_duration"] * 2)
        for i in range(n_frames):
            t = round((i + 1) * 0.5, 2)
            prompt = UM_PROMPT_TEMPLATES[TrackType.GENERATION].format(time_point=t)
            unified.append(build_entry(case_path, img_rel, "generation", prompt,
                                       duration=1, prompt_idx=i, time_point=t))


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".",
                        help="Dataset root (containing sim_subset/ and realworld_subset/)")
    parser.add_argument("--out-dir", default="bench",
                        help="Output folder (relative to --root)")
    parser.add_argument("--split", choices=["both", "sim", "real"], default="both",
                        help="Which split to include (default: both)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on number of cases (for demo / testing)")
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    out_dir = os.path.join(root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cases: List[Tuple] = []
    if args.split in ("both", "sim"):
        cases.extend(enumerate_sim_cases(os.path.join(root, "sim_subset")))
    if args.split in ("both", "real"):
        cases.extend(enumerate_realworld_cases(os.path.join(root, "realworld_subset")))
    if args.limit:
        cases = cases[: args.limit]

    print(f"[info] discovered {len(cases)} cases under {root}")
    if not cases:
        print(f"[error] no cases found. Did you point --root at the dataset folder?")
        return

    videogen: List[dict] = []
    unified: List[dict] = []
    warnings: List[str] = []

    for tup in cases:
        if tup[0] == "sim":
            _, ptype, case_id = tup
            case_path = f"sim_subset/{ptype}/{case_id}"
        else:
            _, pillar, task, case_id = tup
            if pillar == "Composition":
                case_path = f"realworld_subset/Composition/{case_id}"
            else:
                case_path = f"realworld_subset/{pillar}/{task}/{case_id}"

        full_dir = os.path.join(root, case_path)
        info = case_files(full_dir)
        emit_for_case(case_path, info, videogen, unified, warnings)

    vg_path = os.path.join(out_dir, "videogen.json")
    un_path = os.path.join(out_dir, "unified.json")
    indent = args.indent if args.indent > 0 else None
    with open(vg_path, "w") as f:
        json.dump(videogen, f, ensure_ascii=False, indent=indent)
    with open(un_path, "w") as f:
        json.dump(unified, f, ensure_ascii=False, indent=indent)

    print(f"\n[out] {vg_path}: {len(videogen)} entries")
    print(f"[out] {un_path}: {len(unified)} entries")
    print("\n[videogen by track]")
    for t, n in sorted(Counter(e["track"] for e in videogen).items()):
        print(f"  {t}: {n}")
    print("\n[unified by track]")
    for t, n in sorted(Counter(e["track"] for e in unified).items()):
        print(f"  {t}: {n}")
    if warnings:
        print(f"\n[warnings] {len(warnings)} skipped entries:")
        for w in warnings[:10]:
            print(f"  {w}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more")


if __name__ == "__main__":
    main()
