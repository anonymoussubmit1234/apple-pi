"""Utility functions for image/video encoding and frame extraction."""

import base64
import glob
import json
import mimetypes
import os
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def parse_llm_json(raw: str) -> Optional[Dict]:
    """Parse JSON from LLM response, stripping markdown code fences.

    Handles common LLM output patterns like ```json ... ``` wrappers.
    Returns parsed dict on success, None on failure.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def image_to_base64(path: str) -> Tuple[str, str]:
    """Read image file → (base64_string, mime_type).

    Used for Veo3 payload: {"bytesBase64Encoded": b64, "mimeType": mime}.
    """
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        mime_type = "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return b64, mime_type


def image_to_data_url(path: str) -> str:
    """Convert image file → data URL for OpenAI-compatible chat endpoints."""
    b64, mime_type = image_to_base64(path)
    return f"data:{mime_type};base64,{b64}"


def bytes_to_data_url(img_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Convert raw image bytes → data URL (for in-memory frames)."""
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def extract_frames_from_video(
    video_bytes: bytes, num_frames: int = -1
) -> List[np.ndarray]:
    """Extract uniformly-spaced frames from video bytes.

    Returns list of BGR numpy arrays.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        tmp.write(video_bytes)
        tmp.flush()
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError("Could not read frames from video")

        if num_frames == -1:
            # Read every frame sequentially (no seeking per index).
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frames: List[np.ndarray] = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
        else:
            if num_frames <= 0:
                raise ValueError("num_frames must be -1 or a positive integer")
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
        cap.release()
        return frames
    finally:
        os.unlink(tmp.name)


def extract_frames_from_file(
    video_path: str, num_frames: int = -1
) -> List[np.ndarray]:
    """Extract uniformly-spaced frames from a video file on disk."""
    with open(video_path, "rb") as f:
        return extract_frames_from_video(f.read(), num_frames)


def extract_last_frame_from_video(video_bytes: bytes) -> List[np.ndarray]:
    """Extract only the last frame from video bytes.
    
    Returns list of one BGR numpy array.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        tmp.write(video_bytes)
        tmp.flush()
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError("Could not read frames from video")

        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return [frame]
        return []
    finally:
        os.unlink(tmp.name)


def extract_last_frame_from_file(video_path: str) -> List[np.ndarray]:
    """Extract only the last frame from a video file on disk."""
    with open(video_path, "rb") as f:
        return extract_last_frame_from_video(f.read())


def load_gt_frames_from_dir(
    rgb_dir: str, num_frames: int = 8, skip_first: bool = True
) -> List[np.ndarray]:
    """Load GT frames from a directory of numbered PNGs.

    The demo data has rgb/0000.png (clean first frame, no annotations)
    and rgb/0001.png ~ rgb/0240.png (240 GT frames, 24fps, 10s).

    Args:
        rgb_dir: directory containing numbered PNG frames
        num_frames: how many frames to uniformly sample
        skip_first: if True, skip 0000.png (clean first frame)
    """
    pattern = os.path.join(rgb_dir, "*.png")
    all_paths = sorted(glob.glob(pattern))

    if skip_first and len(all_paths) > 1:
        # Skip 0000.png (clean first frame without annotations)
        all_paths = all_paths[1:]

    if not all_paths:
        raise ValueError(f"No PNG frames found in {rgb_dir}")

    # Uniformly sample
    indices = np.linspace(0, len(all_paths) - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        frame = cv2.imread(all_paths[idx], cv2.IMREAD_COLOR)
        if frame is not None:
            frames.append(frame)
    return frames


def frame_to_jpeg_bytes(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode BGR numpy frame → JPEG bytes."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def save_video(video_bytes: bytes, path: str) -> str:
    """Save raw video bytes to disk. Returns absolute path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(video_bytes)
    return os.path.abspath(path)


def align_frame_counts(
    gen_frames: List[np.ndarray],
    gt_frames: List[np.ndarray],
    target_count: int = 8,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Align two frame lists to the same count via uniform subsampling.

    Handles different video lengths (e.g., Veo3 generates 8s but GT is 10s)
    by sampling target_count frames uniformly from each independently.
    This way gen frame i and gt frame i represent the same proportional
    time point in each video.
    """

    def subsample(frames: List[np.ndarray], n: int) -> List[np.ndarray]:
        if len(frames) <= n:
            return frames
        indices = np.linspace(0, len(frames) - 1, n, dtype=int)
        return [frames[i] for i in indices]

    return subsample(gen_frames, target_count), subsample(gt_frames, target_count)
