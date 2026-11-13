"""Configuration loading. All paths and thresholds live in `configs/*.yaml`."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class Config:
    """Thin wrapper around the parsed YAML with typed accessors for common fields."""

    raw: dict[str, Any]
    path: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def resolve(self, p: str | os.PathLike | None) -> Path | None:
        """Resolve a config path: absolute paths stay, relative ones are taken from the repo root."""
        if p is None:
            return None
        p = Path(p).expanduser()
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def bfmd_root(self) -> Path:
        env = os.environ.get("BFMD_ROOT")
        return self.resolve(env or self.raw["paths"]["bfmd_root"])  # type: ignore[return-value]

    @property
    def ann_root(self) -> Path:
        return self.bfmd_root / "annotations"

    @property
    def cache_dir(self) -> Path:
        return self.resolve(self.raw["paths"]["cache_dir"])  # type: ignore[return-value]

    @property
    def output_dir(self) -> Path:
        return self.resolve(self.raw["paths"]["output_dir"])  # type: ignore[return-value]

    @property
    def video_root(self) -> Path | None:
        return self.resolve(self.raw["paths"].get("video_root"))

    @property
    def width(self) -> int:
        return int(self.raw["video"]["width"])

    @property
    def height(self) -> int:
        return int(self.raw["video"]["height"])

    @property
    def fps(self) -> float:
        return float(self.raw["video"]["fps"])

    def shift(self, source: str) -> int:
        """Frame shift for one annotation source (aligned = raw + shift)."""
        return int(self.raw.get("frame_shift", {}).get(source, 0))


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load a YAML config (default: configs/default.yaml).

    A config may start with `base: other.yaml` (relative to its own folder); it is then a set of
    overrides deep-merged onto that base. configs/colab.yaml uses this to change only the paths.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    base = raw.pop("base", None)
    if base:
        raw = _deep_merge(load_config(path.parent / base).raw, raw)
    return Config(raw=raw, path=path)
