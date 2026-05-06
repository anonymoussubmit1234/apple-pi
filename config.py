"""Configuration for the Veo3 prompt optimization pipeline."""

import os
import random
import json
import hashlib
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FORMULA_DIR = os.path.join(BASE_DIR, "Formula")
FORMULA_INFO_FILENAME = "formula_info.json"


class TrackType(Enum):
    # Perception subtracks
    PERCEPTION_TEXT = "perception_text"  # OCR: can model read/preserve text labels
    PERCEPTION_GRAPHIC = "perception_graphic"  # can model preserve axes, arrows, lines

    # Comprehension subtracks
    COMPREHENSION_TEXT = "comprehension_text"  # formula selection + substitution (text output)
    COMPREHENSION_GRAPHIC = "comprehension_graphic"  # moment: position + velocity arrow (graphic output)

    # Generation
    GENERATION = "generation"


@dataclass
class APIConfig:
    """API configuration. All endpoints + the API key are read from env vars
    at runtime, no fallback defaults — set them in your `.env` file or
    `export` them before running.

    Required env vars:
      API_KEY              — bearer token for the API gateway
      LLM_CHAT_URL         — chat completions endpoint (LLM judge)
      LLM_CHAT_MODEL       — model name passed in the chat payload
                             (e.g. "gemini-2.5-flash", "gpt-4o")
      VEO3_BASE            — Veo3 video-gen base URL (optional, only for inference)
      VEO3_FAST_BASE       — Veo3-fast video-gen base URL (optional)
    """

    api_key: str = field(default_factory=lambda: os.environ["API_KEY"])

    # Video generation endpoints (Veo3-style) — only used for inference scripts
    veo3_base: str = field(
        default_factory=lambda: os.environ.get("VEO3_BASE", "")
    )
    veo3_fast_base: str = field(
        default_factory=lambda: os.environ.get("VEO3_FAST_BASE",
                                               os.environ.get("VEO3_BASE", ""))
    )

    # LLM judge chat endpoint + model name
    gemini_flash_url: str = field(default_factory=lambda: os.environ["LLM_CHAT_URL"])
    chat_model: str = field(default_factory=lambda: os.environ["LLM_CHAT_MODEL"])

    # Veo3 parameters
    video_duration_seconds: int = 8
    generate_audio: bool = True

    # Polling
    poll_interval_seconds: int = 10
    poll_max_attempts: int = 60  # 10 min max

    # LLM parameters
    llm_max_tokens: int = 4096
    reasoning_effort: str = "high"

    @property
    def headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def veo3_generate_url(self, fast: bool = False) -> str:
        base = self.veo3_fast_base if fast else self.veo3_base
        return f"{base}/veo/videos/generate"

    def veo3_task_url(self, fast: bool = False) -> str:
        base = self.veo3_fast_base if fast else self.veo3_base
        return f"{base}/veo/videos/task"


@dataclass
class CaseData:
    """Data for a single physics simulation case from the demo dataset.

    Demo structure per case:
      {case_dir}/
        annotation.txt          - physics params (g, e, d, h, v_0)
        prompt.txt              - pre-written prompt for Veo3
        initial_state/
          rgb_0000.png          - annotated first frame (conditioning image)
          rgb_0000_arrows.png   - first frame with velocity arrows (if v_0!=0)
          velocity_0000.npy     - initial velocity field
          mask_0000.npy         - object mask
          ...
        rgb/
          0000.png              - clean first frame (no annotations)
          0001.png ~ 0240.png   - GT video frames (240 frames, 24fps, 10s)
        instantaneous_velocity/
          0000.png              - moment tracking: velocity arrow at a time point
    """

    case_id: int
    case_dir: str
    annotation_text: str  # raw content of annotation.txt
    first_frame_path: str  # initial_state/rgb_0000.png (annotated)
    clean_first_frame_path: str  # rgb/0000.png (no annotations)
    gt_frames_dir: str  # rgb/ directory

    # Optional or default fields MUST follow non-default fields
    white_bg_first_frame_path: Optional[str] = None  # initial_state/rgb_0000_white_bg.png
    white_bg_obj_first_frame_path: Optional[str] = None  # initial_state/rgb_0000_white_bg_obj.png
    track_image_path: Optional[str] = None  # instantaneous_velocity/velocity_annotated.png (moment subtrack ref)
    arrows_image_path: Optional[str] = None  # initial_state/rgb_0000_arrows.png
    gt_video_path: Optional[str] = None  # rgb/video.mp4 (real-world only)
    target_time: float = 0.0  # time (seconds) for comprehension_graphic track
    generation_duration: float = 8.0  # video duration (seconds) for generation track
    physics_duration: float = 10.0  # real-world physics duration (seconds)
    gt_fps: int = 24
    gt_duration_seconds: float = 10.0
    gt_total_frames: int = 240  # 0001-0240

    # Physics type for formula lookup (derived from annotation)
    physics_type: str = "FreeFall"

    # Whether this case uses realworld data (vs simulator)
    is_realworld: bool = False

    @classmethod
    def load(cls, case_id: int, demo_dir: str) -> "CaseData":
        """Load a case from the demo directory."""
        case_dir = os.path.join(demo_dir, str(case_id))
        if not os.path.isdir(case_dir):
            raise FileNotFoundError(f"Case directory not found: {case_dir}")

        with open(os.path.join(case_dir, "annotation.txt")) as f:
            annotation_text = f.read().strip()

        first_frame = os.path.join(case_dir, "initial_state", "rgb_0000.png")
        clean_first = os.path.join(case_dir, "rgb", "0000.png")
        white_bg_img = os.path.join(case_dir, "initial_state", "rgb_0000_white_bg.png")
        white_bg_img = white_bg_img if os.path.exists(white_bg_img) else None
        
        white_bg_obj_img = os.path.join(case_dir, "initial_state", "rgb_0000_white_bg_obj.png")
        white_bg_obj_img = white_bg_obj_img if os.path.exists(white_bg_obj_img) else None
        gt_dir = os.path.join(case_dir, "rgb")

        track_img = os.path.join(case_dir, "instantaneous_velocity", "velocity_annotated.png")
        track_img = track_img if os.path.exists(track_img) else None

        arrows_img = os.path.join(
            case_dir, "initial_state", "rgb_0000_arrows.png"
        )
        arrows_img = arrows_img if os.path.exists(arrows_img) else None

        # Load target time for comprehension_graphic track.
        target_time = None
        velocity_json_path = os.path.join(
            case_dir, "instantaneous_velocity", "velocity.json"
        )
        if os.path.exists(velocity_json_path):
            with open(velocity_json_path, "r", encoding="utf-8") as f:
                velocity_data = json.load(f)
            if "time_seconds" in velocity_data:
                target_time = float(velocity_data["time_seconds"])
            else:
                print(f"Warning: 'time_seconds' not found in {velocity_json_path}, using default 0.0")
        else:
            raise FileNotFoundError(f"Missing required velocity info for Case {case_id}: {velocity_json_path}")

        # Load physics_duration from physics_duration.txt.
        # This is the authoritative source for how many seconds of physics
        # the generation model should produce. Every case (sim + real-world)
        # should have this file filled in.
        physics_duration_file = os.path.join(case_dir, "physics_duration.txt")
        physics_duration = None
        if os.path.exists(physics_duration_file):
            with open(physics_duration_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    try:
                        physics_duration = float(content)
                    except ValueError:
                        print(f"Warning: Invalid physics_duration in {physics_duration_file}")

        is_realworld = not os.path.exists(os.path.join(case_dir, "initial_state", "instance_segmentation_mapping_0000.json"))

        # Real-world cases: GT is rgb/video.mp4 instead of individual PNGs.
        # Read gt_fps from the video.
        gt_video_path = os.path.join(gt_dir, "video.mp4")
        gt_fps = 24
        gt_total_frames = 240
        gt_duration_seconds = 10.0

        if os.path.exists(gt_video_path):
            import cv2 as _cv2
            cap = _cv2.VideoCapture(gt_video_path)
            video_fps = cap.get(_cv2.CAP_PROP_FPS)
            video_total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if video_fps > 0 and video_total > 0:
                gt_fps = int(round(video_fps))
                gt_total_frames = video_total
                gt_duration_seconds = (video_total - 1) / video_fps
                # Fallback: if physics_duration.txt is empty, derive from video
                if physics_duration is None:
                    physics_duration = gt_duration_seconds
                    print(f"Warning: physics_duration.txt empty for case {case_id}, "
                          f"derived {physics_duration:.3f}s from video "
                          f"({video_total} frames, {video_fps}fps)")
        else:
            gt_video_path = None

        # Final fallback / validation
        if physics_duration is None:
            raise FileNotFoundError(
                f"Missing physics_duration for case {case_id}: "
                f"{physics_duration_file} is empty or missing, "
                f"and no video.mp4 to derive from"
            )

        return cls(
            case_id=case_id,
            case_dir=case_dir,
            annotation_text=annotation_text,
            first_frame_path=first_frame,
            clean_first_frame_path=clean_first,
            white_bg_first_frame_path=white_bg_img,
            white_bg_obj_first_frame_path=white_bg_obj_img,
            gt_frames_dir=gt_dir,
            gt_video_path=gt_video_path,
            track_image_path=track_img,
            arrows_image_path=arrows_img,
            target_time=target_time,
            physics_duration=physics_duration,
            gt_fps=gt_fps,
            gt_duration_seconds=gt_duration_seconds,
            gt_total_frames=gt_total_frames,
            is_realworld=is_realworld,
        )


def load_formulas(
    physics_type: str,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Load formulas for a physics type, grouped by category.

    The corresponding `{physics_type}.txt` file is expected to have four sections:

      right:
        <correct formulas for this specific scenario>
      wrong:
        <real formulas that misuse the annotated quantities for this scenario>
      unrelated:
        <real formulas from other physics domains>
      fictional:
        <plausible-looking but fabricated formulas>

    Returns (right_formulas, wrong_formulas, unrelated_formulas, fictional_formulas).
    """
    path = os.path.join(FORMULA_DIR, f"{physics_type}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Formula file not found: {path}")

    with open(path) as f:
        content = f.read()

    right: List[str] = []
    wrong: List[str] = []
    unrelated: List[str] = []
    fictional: List[str] = []

    current: Optional[List[str]] = None
    for line in content.strip().splitlines():
        line = line.strip()
        if line.lower().startswith("right"):
            current = right
        elif line.lower().startswith("wrong"):
            current = wrong
        elif line.lower().startswith("unrelated"):
            current = unrelated
        elif line.lower().startswith("fictional"):
            current = fictional
        elif line and current is not None:
            current.append(line)
    return right, wrong, unrelated, fictional


def build_formula_choices(
    physics_type: str, num_wrong: int = 3
) -> Tuple[List[str], int]:
    """Build a shuffled list of formula choices.

    From the four-category pools, build exactly four options:
      - 1 correct formula from `right`
      - 1 distractor from `wrong`
      - 1 distractor from `unrelated`
      - 1 distractor from `fictional`

    Returns (choices_list, correct_index_0based).
    """
    right, wrong, unrelated, fictional = load_formulas(physics_type)
    if not right:
        raise ValueError(f"No correct formula found for {physics_type}")

    # Ensure each category has at least one candidate
    categories = {
        "right": right,
        "wrong": wrong,
        "unrelated": unrelated,
        "fictional": fictional,
    }
    for name, pool in categories.items():
        if not pool:
            raise ValueError(f"No formulas found in '{name}' for {physics_type}")

    # Randomly pick one from each category
    correct = random.choice(right)
    wrong_choice = random.choice(wrong)
    unrelated_choice = random.choice(unrelated)
    fictional_choice = random.choice(fictional)

    choices = [correct, wrong_choice, unrelated_choice, fictional_choice]
    random.shuffle(choices)
    correct_idx = choices.index(correct)
    return choices, correct_idx


def build_formula_choices_seeded(
    physics_type: str,
    *,
    seed: Union[int, str],
) -> Tuple[List[str], int]:
    """Deterministically build formula choices for a given seed.

    This avoids global RNG side effects while keeping the same category logic
    as `build_formula_choices`.
    """
    right, wrong, unrelated, fictional = load_formulas(physics_type)
    if not right:
        raise ValueError(f"No correct formula found for {physics_type}")

    categories = {
        "right": right,
        "wrong": wrong,
        "unrelated": unrelated,
        "fictional": fictional,
    }
    for name, pool in categories.items():
        if not pool:
            raise ValueError(f"No formulas found in '{name}' for {physics_type}")

    if isinstance(seed, str):
        # stable across runs/processes
        seed_int = int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16], 16)
    else:
        seed_int = int(seed)

    rng = random.Random(seed_int)
    correct = rng.choice(right)
    wrong_choice = rng.choice(wrong)
    unrelated_choice = rng.choice(unrelated)
    fictional_choice = rng.choice(fictional)

    choices = [correct, wrong_choice, unrelated_choice, fictional_choice]
    rng.shuffle(choices)
    correct_idx = choices.index(correct)
    return choices, correct_idx


def load_case_formula_info(case_dir: str) -> Optional[Dict[str, object]]:
    """Load cached formula info from `{case_dir}/formula_info.json` if present."""
    path = os.path.join(case_dir, FORMULA_INFO_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid formula_info format: {path}")
    # soft validation of required keys
    for k in ("correct_letter", "correct_formula", "annotation"):
        if k not in data:
            raise ValueError(f"Missing key '{k}' in {path}")
    return data


def save_case_formula_info(case_dir: str, formula_info: Dict[str, object]) -> str:
    """Save formula info to `{case_dir}/formula_info.json` (UTF-8, pretty)."""
    os.makedirs(case_dir, exist_ok=True)
    path = os.path.join(case_dir, FORMULA_INFO_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(formula_info, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path


@dataclass
class OptimizationConfig:
    """Parameters for the prompt optimization loop."""

    max_iterations: int = 7
    target_score: float = 0.8
    use_fast_veo: bool = True
    num_eval_frames: int = 8

    # Track weights for overall scoring
    perception_weight: float = 0.3
    comprehension_weight: float = 0.35
    generation_weight: float = 0.35


def parse_tracks(value: str, include_generation: bool = True) -> List[TrackType]:
    """Parse --track argument into list of TrackType.

    Accepts: "all", "perception", "comprehension", "all_no_gen",
             "all_with_gen", or any TrackType value like "perception_text".

    Args:
        include_generation: whether "all" includes GENERATION track.
    """
    all_tracks = list(TrackType) if include_generation else [
        t for t in TrackType if t != TrackType.GENERATION
    ]
    groups = {
        "all": all_tracks,
        "all_with_gen": list(TrackType),
        "all_no_gen": [t for t in TrackType if t != TrackType.GENERATION],
        "perception": [TrackType.PERCEPTION_TEXT, TrackType.PERCEPTION_GRAPHIC],
        "comprehension": [TrackType.COMPREHENSION_TEXT, TrackType.COMPREHENSION_GRAPHIC],
    }
    if value in groups:
        return groups[value]
    try:
        return [TrackType(value)]
    except ValueError:
        valid = [t.value for t in TrackType] + list(groups.keys())
        print(f"Invalid track '{value}'. Valid: {valid}")
        sys.exit(1)


def parse_cases(value: str) -> List[int]:
    """Parse --case argument into list of case IDs.

    Accepts: single id "0", or comma-separated "0,1,2". Use the dataset
    structure to enumerate "all cases" — there is no single canonical
    case-id list (cases are partitioned by ptype / pillar / task).
    """
    try:
        return [int(x.strip()) for x in value.split(",")]
    except ValueError:
        print(f"Invalid case '{value}'. Use: 0, 1, 2, 3, 0,1,2, or all")
        sys.exit(1)
