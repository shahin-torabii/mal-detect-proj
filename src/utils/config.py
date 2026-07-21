"""Small helper to load the project's YAML config file."""
from pathlib import Path
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load config.yaml (or a custom path) into a plain dict.

    Using a config file instead of hard-coded paths means the same code
    runs on any machine (Windows/Linux/Colab) — you only edit config.yaml.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{path}'. "
            "Copy config.yaml to the project root and edit the paths."
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(preference: str = "auto") -> str:
    """Resolve the requested device, falling back to CPU if CUDA is unavailable."""
    import torch

    if preference == "cpu":
        return "cpu"
    if preference == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU.")
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
