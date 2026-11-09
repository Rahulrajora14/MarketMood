"""Load the YAML run configuration and resolve paths relative to the project root."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def dir(self, name: str) -> Path:
        """Absolute path for one of the entries under `paths:`; created if missing."""
        p = Path(self.raw["paths"][name])
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def tickers(self) -> list[str]:
        return list(self.raw["universe"]["tickers"] or [])


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, path=path)


def save_universe(cfg: Config, tickers: list[str]) -> None:
    """Freeze the ticker universe into the config file (done once)."""
    cfg.raw["universe"]["tickers"] = sorted(tickers)
    text = cfg.path.read_text(encoding="utf-8")
    block = "  tickers: [" + ", ".join(sorted(tickers)) + "]"
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("tickers:"):
            comment = ""
            if "#" in line:
                comment = "  #" + line.split("#", 1)[1]
            lines[i] = block + comment
            break
    cfg.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
