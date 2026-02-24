import json
import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional


TRAJECTORY_KEYS = [
    "trajectory",
    "legs_trajectory",
    "shoulder_trajectory",
    "back_trajectory",
    "knee_angle_trajectory",
    "arm_Trajectory",
    "core_",
]


class ClipDataset(Dataset):
    """Loads clip JSON files from `data/` and yields trajectory waves + metadata."""

    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 300,
        signal_keys: Optional[List[str]] = None,
        normalize: bool = True,
    ):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.signal_keys = signal_keys or TRAJECTORY_KEYS
        self.normalize = normalize
        self.clips = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        if not self.clips:
            raise FileNotFoundError(f"No JSON clips found in {data_dir}")

    def __len__(self) -> int:
        return len(self.clips)

    def _pad_or_truncate(self, arr: np.ndarray) -> np.ndarray:
        if len(arr) >= self.sequence_length:
            return arr[: self.sequence_length]
        out = np.zeros(self.sequence_length, dtype=np.float32)
        out[: len(arr)] = arr
        return out

    def _norm(self, arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-8:
            return np.zeros_like(arr)
        return ((arr - lo) / (hi - lo)).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict:
        with open(self.clips[idx], "r") as f:
            raw = json.load(f)

        video_id = raw.get("video_id", os.path.basename(self.clips[idx]).replace(".json", ""))
        exercise = raw.get("exercise", "unknown")
        fps = raw.get("fps", 30.0)
        n_frames = raw.get("n_frames", 0)
        expert = raw.get("expert", False)
        error_rate = raw.get("error_rate", [])

        waves = {}
        for key in self.signal_keys:
            signal = np.array(raw.get(key, []), dtype=np.float32)
            if len(signal) == 0:
                signal = np.zeros(self.sequence_length, dtype=np.float32)
            orig_len = len(signal)
            if self.normalize and signal.max() - signal.min() > 1e-8:
                signal = self._norm(signal)
            signal = self._pad_or_truncate(signal)
            waves[key] = torch.from_numpy(signal)

        mask = torch.zeros(self.sequence_length)
        mask[: min(n_frames, self.sequence_length)] = 1.0

        wave_features = raw.get("wave_features", {})
        language = raw.get("LANGUAGE", "")

        stacked = torch.stack([waves[k] for k in self.signal_keys], dim=0)  # (C, T)

        return {
            "video_id": video_id,
            "exercise": exercise,
            "fps": fps,
            "expert": expert,
            "error_rate": error_rate,
            "waves": stacked,
            "mask": mask,
            "wave_features": wave_features,
            "language": language,
        }
