"""Generic API client for video generation, image generation, and LLM chat completions."""

import base64
import csv
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

from config import APIConfig, BASE_DIR

logger = logging.getLogger(__name__)

# ── API Usage Tracker ─────────────────────────────────────────────────
USAGE_LOG_PATH = os.path.join(BASE_DIR, "api_usage.csv")
API_COUNT_PATH = os.path.join(BASE_DIR, "api_call_count.txt")
USAGE_CSV_FIELDS = ["timestamp", "date", "model", "endpoint", "status_code", "success"]

# In-memory buffer, flushed periodically or at exit
_usage_buffer: List[dict] = []
_FLUSH_EVERY = 20  # flush to disk every N calls


def _increment_api_count() -> None:
    count = 0
    if os.path.exists(API_COUNT_PATH):
        try:
            with open(API_COUNT_PATH, "r") as f:
                content = f.read().strip()
                if content:
                    count = int(content)
        except (ValueError, IOError):
            pass
    count += 1
    try:
        with open(API_COUNT_PATH, "w") as f:
            f.write(str(count))
    except IOError as e:
        logger.error(f"Failed to write API count: {e}")

def _log_api_call(
    model: str, endpoint: str, status_code: int, success: bool
) -> None:
    """Buffer one API call record. Flushes to CSV periodically."""
    _increment_api_count()
    now = datetime.now()
    _usage_buffer.append({
        "timestamp": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "model": model,
        "endpoint": endpoint,
        "status_code": status_code,
        "success": success,
    })
    if len(_usage_buffer) >= _FLUSH_EVERY:
        flush_usage_log()


def flush_usage_log() -> None:
    """Write buffered records to CSV and clear the buffer."""
    if not _usage_buffer:
        return
    file_exists = os.path.exists(USAGE_LOG_PATH)
    with open(USAGE_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=USAGE_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(_usage_buffer)
    _usage_buffer.clear()


import atexit
atexit.register(flush_usage_log)


class APIClient:
    """Unified API client supporting two patterns:

    Two API patterns:
    1. Veo3: POST to submit → GET polling for result (video bytes).
    2. Chat completions (Gemini 3.0 Flash): single POST, OpenAI-compatible.
    """

    def __init__(self, config: APIConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(config.headers)

    # ── Veo3 Video Generation ──────────────────────────────────────────

    def submit_video_generation(
        self,
        prompt: str,
        image_b64: str,
        image_mime: str,
        fast: bool = True,
        duration_seconds: Optional[int] = None,
    ) -> str:
        """Submit Veo3 video generation request. Returns task_id."""
        url = self.config.veo3_generate_url(fast=fast)
        dur = duration_seconds or self.config.video_duration_seconds
        payload = {
            "instances": [
                {
                    "prompt": prompt,
                    "image": {
                        "bytesBase64Encoded": image_b64,
                        "mimeType": image_mime,
                    },
                }
            ],
            "parameters": {
                "durationSeconds": dur,
                "generateAudio": self.config.generate_audio,
            },
        }

        logger.info(
            f"Submitting video generation (fast={fast}), "
            f"prompt={prompt[:80]}..."
        )
        model_name = "veo3_fast" if fast else "veo3"
        resp = self.session.post(url, json=payload, timeout=120)
        _log_api_call(model_name, url, resp.status_code, resp.status_code == 200)
        if resp.status_code != 200:
            logger.error(
                f"Veo3 submit failed ({resp.status_code}): {resp.text[:500]}"
            )
        resp.raise_for_status()
        data = resp.json()
        logger.debug(f"Submit response keys: {list(data.keys())}")

        # Extract task ID — try common response patterns
        task_id = (
            data.get("name")
            or data.get("taskId")
            or data.get("operationId")
            or data.get("id")
        )
        if not task_id:
            logger.warning(
                f"Could not extract task_id, full response: {data}"
            )
            raise RuntimeError(f"No task_id in response: {data}")

        logger.info(f"Video generation submitted, task_id={task_id}")
        return str(task_id)

    def poll_video_task(
        self,
        task_id: str,
        fast: bool = True,
    ) -> bytes:
        """Poll until video generation completes. Returns video bytes (MP4)."""
        url = self.config.veo3_task_url(fast=fast)
        poll_payload = {"operationName": task_id}

        for attempt in range(self.config.poll_max_attempts):
            logger.info(
                f"Polling attempt {attempt + 1}/{self.config.poll_max_attempts}"
            )
            model_name = "veo3_fast_poll" if fast else "veo3_poll"
            resp = self.session.post(url, json=poll_payload, timeout=300)
            _log_api_call(model_name, url, resp.status_code, resp.status_code == 200)
            resp.raise_for_status()
            data = resp.json()
            logger.debug(f"Poll response keys: {list(data.keys())}")

            # Response may wrap result under "data" key
            inner = data.get("data", data)
            state = (
                inner.get("state")
                or inner.get("status")
                or inner.get("done")
            )

            if state in ("SUCCEEDED", "COMPLETED", True, "done"):
                video_b64 = self._extract_video_b64(inner)
                if video_b64:
                    logger.info("Video ready (base64), decoding...")
                    return base64.b64decode(video_b64)

                video_url = self._extract_video_url(inner)
                if video_url:
                    logger.info(f"Video ready (URL), downloading: {video_url}")
                    dl = self.session.get(video_url, timeout=120)
                    dl.raise_for_status()
                    return dl.content

                raise RuntimeError(
                    f"Task completed but no video data found. "
                    f"Response: {data}"
                )

            if state in ("FAILED", "CANCELLED", "ERROR"):
                error_msg = inner.get("error", {}).get("message", str(data))
                raise RuntimeError(f"Video generation failed: {error_msg}")

            time.sleep(self.config.poll_interval_seconds)

        raise TimeoutError(
            f"Video generation did not complete within "
            f"{self.config.poll_max_attempts * self.config.poll_interval_seconds}s"
        )

    def _extract_video_b64(self, data: dict) -> Optional[str]:
        """Try multiple JSON paths to find base64 video data."""
        paths = [
            lambda d: d["response"]["videos"][0]["bytesBase64Encoded"],
            lambda d: d["result"]["videos"][0]["bytesBase64Encoded"],
            lambda d: d["videos"][0]["bytesBase64Encoded"],
            lambda d: d["response"]["generatedSamples"][0]["video"][
                "bytesBase64Encoded"
            ],
        ]
        for path in paths:
            try:
                return path(data)
            except (KeyError, IndexError, TypeError):
                continue
        return None

    def _extract_video_url(self, data: dict) -> Optional[str]:
        """Try multiple JSON paths to find video download URL."""
        paths = [
            lambda d: d["response"]["videos"][0]["uri"],
            lambda d: d["result"]["videos"][0]["uri"],
            lambda d: d["videos"][0]["uri"],
            lambda d: d["response"]["videos"][0]["url"],
            lambda d: d["response"]["generatedSamples"][0]["video"]["uri"],
        ]
        for path in paths:
            try:
                return path(data)
            except (KeyError, IndexError, TypeError):
                continue
        return None

    # ── Chat Completions (Gemini 3.0 Flash) ────────────────────────────

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
    ) -> str:
        """Send chat completion request. Returns assistant message content."""
        url = self.config.gemini_flash_url
        payload = {
            "model": self.config.chat_model,
            "messages": messages,
            "reasoning_effort": self.config.reasoning_effort,
        }

        max_retries = 5
        for attempt in range(max_retries):
            logger.info(f"Sending chat completion request (attempt {attempt + 1})...")
            try:
                resp = self.session.post(url, json=payload, timeout=300)
            except requests.exceptions.ReadTimeout:
                logger.warning(f"Read timeout on attempt {attempt + 1}, retrying...")
                time.sleep(5)
                continue
            _log_api_call("gemini_3.0_flash", url, resp.status_code, resp.status_code == 200)

            if resp.status_code == 429:
                wait = min(30, 5 * (attempt + 1))
                logger.warning(f"Rate limited (429), waiting {wait}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        raise RuntimeError(f"Chat completion failed after {max_retries} retries")

    # ── Image Generation (Nano-Banana style) ──────────────────────────

    @property
    def NANO_BANANA_PRO_BASE(self) -> str:
        """Image-generation API base URL — required env var `IMAGE_GEN_BASE`."""
        return os.environ["IMAGE_GEN_BASE"]

    def generate_image(
        self,
        prompt: str,
        input_image_b64: Optional[str] = None,
        input_image_mime: str = "image/png",
        aspect_ratio: str = "16:9",
        image_size: str = "1k",
    ) -> Tuple[bytes, str]:
        """Nano Banana Pro: text (+image) → image.

        Returns (image_bytes, mime_type).
        Raises on failure.
        """
        url = f"{self.NANO_BANANA_PRO_BASE}/generateContent"

        parts: List[dict] = [{"text": prompt}]
        if input_image_b64:
            parts.append({
                "inlineData": {
                    "mimeType": input_image_mime,
                    "data": input_image_b64,
                }
            })

        payload = {
            "contents": [{"role": "USER", "parts": parts}],
            "generationConfig": {
                "response_modalities": ["TEXT", "IMAGE"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": image_size,
                },
                "temperature": 0.4,
                "maxOutputTokens": 2048,
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
            ],
        }

        max_retries = 3
        for attempt in range(max_retries):
            logger.info(
                f"Nano Banana image gen (attempt {attempt + 1}), "
                f"prompt={prompt[:80]}..."
            )
            try:
                resp = self.session.post(url, json=payload, timeout=300)
            except requests.exceptions.ReadTimeout:
                logger.warning(
                    f"Timeout on attempt {attempt + 1}, retrying..."
                )
                time.sleep(5)
                continue
            _log_api_call(
                "nano_banana_pro", url,
                resp.status_code, resp.status_code == 200,
            )

            if resp.status_code == 429:
                wait = min(60, 15 * (attempt + 1))
                logger.warning(f"Rate limited (429), waiting {wait}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            # Extract image from response parts
            for part in data["candidates"][0]["content"]["parts"]:
                if "inlineData" in part:
                    mime = part["inlineData"].get("mimeType", "image/png")
                    img_bytes = base64.b64decode(part["inlineData"]["data"])
                    logger.info(
                        f"Image generated: {len(img_bytes)} bytes ({mime})"
                    )
                    return img_bytes, mime

            raise RuntimeError(
                f"No image in response. Parts: "
                f"{[list(p.keys()) for p in data['candidates'][0]['content']['parts']]}"
            )

        raise RuntimeError(
            f"Nano Banana image gen failed after {max_retries} retries"
        )
