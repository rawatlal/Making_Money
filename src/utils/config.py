import os
from pathlib import Path

import yaml

_config = None

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def load_config(path=None):
    """Load YAML config, merging with defaults."""
    defaults = _load_yaml(DEFAULT_CONFIG_PATH)
    if path and Path(path).exists():
        overrides = _load_yaml(path)
        return _deep_merge(defaults, overrides)
    return defaults


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base, override):
    """Recursively merge override into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_config(path=None):
    """Singleton access to the active config."""
    global _config
    if _config is None or path is not None:
        _config = load_config(path)
    return _config
