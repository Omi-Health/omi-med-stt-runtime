from __future__ import annotations

import os


def hf_token() -> str | None:
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    return None
