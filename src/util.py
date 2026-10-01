import re


def slugify(text: str) -> str:
    """Turn a circuit Location string into a stable, filesystem-safe folder name."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")
