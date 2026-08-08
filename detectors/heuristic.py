from __future__ import annotations
from pathlib import Path
import math
import cv2
import numpy as np
from .base import DetectorAdapter, DetectorResult

class HeuristicVideoDetector(DetectorAdapter):
    """Dependency-light baseline for pipeline testing, not a validated detector."""
    name = "heuristic_baseline"
    version = "0.1"

    def score_video(self, path: str | Path, max_frames: int = 96, **kwargs) -> DetectorResult:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        gray_vars=[]; lap_vars=[]; diffs=[]; frames=0; prev=None
        while frames < max_frames:
            ok, frame=cap.read()
            if not ok: break
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            gray_vars.append(float(np.std(gray)))
            lap_vars.append(float(cv2.Laplacian(gray,cv2.CV_64F).var()))
            if prev is not None:
                diffs.append(float(np.mean(cv2.absdiff(gray,prev))))
            prev=gray; frames+=1
        cap.release()
        if frames < 2:
            raise ValueError("Video contains fewer than two readable frames")
        temporal=float(np.mean(diffs)) if diffs else 0.0
        sharp=float(np.mean(lap_vars))
        contrast=float(np.mean(gray_vars))
        score=1.0/(1.0+math.exp(-(0.02*temporal-0.001*sharp-0.01*contrast)))
        return DetectorResult(
            video_id=Path(path).stem,
            label=None,
            score=float(score),
            features={
                "spatial_score": float(score),
                "temporal_score": float(temporal),
                "frequency_score": float(sharp),
                "compression_score": float(contrast),
            },
            metadata={"frames":frames,"detector":self.name,"version":self.version},
        )
