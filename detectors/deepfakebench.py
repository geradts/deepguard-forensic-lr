from __future__ import annotations
import json
import subprocess
from pathlib import Path
from typing import Any

from .base import DetectorAdapter, DetectorResult

class DeepfakeBenchAdapter(DetectorAdapter):
    """Adapter for an externally installed DeepfakeBench checkout.

    It does not vendor DeepfakeBench. Configure its inference command in a JSON
    template and require that the command writes one JSON object containing a
    video-level score. This keeps model weights and upstream licensing separate.
    """
    name = "deepfakebench"
    version = "adapter-0.1"

    def __init__(self, checkout: str, command_template: str):
        self.checkout = Path(checkout).resolve()
        self.command_template = command_template
        if not self.checkout.exists():
            raise FileNotFoundError(f"DeepfakeBench checkout not found: {self.checkout}")

    def score_video(self, path: str | Path, **kwargs) -> DetectorResult:
        video = Path(path).resolve()
        with open(self.checkout / ".deepguard_detector_output.json", "w", encoding="utf-8") as marker:
            marker.write("")
        command = self.command_template.format(video=str(video), output=str(self.checkout / ".deepguard_detector_output.json"))
        proc = subprocess.run(command, cwd=self.checkout, shell=True, text=True, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-4000:])
        output = self.checkout / ".deepguard_detector_output.json"
        if not output.exists() or not output.read_text(encoding="utf-8").strip():
            raise RuntimeError("Detector command did not produce the configured JSON output")
        data = json.loads(output.read_text(encoding="utf-8"))
        if "score" not in data:
            raise ValueError("Detector JSON must contain 'score'")
        features = data.get("features", {})
        return DetectorResult(
            video_id=video.stem,
            label=data.get("label"),
            score=float(data["score"]),
            features={k: float(v) for k,v in features.items()},
            metadata={"detector":self.name,"version":self.version,"stdout":proc.stdout[-2000:]},
        )
