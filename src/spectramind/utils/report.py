import base64
import os
from typing import Optional

from .paths import ensure_dir


def html_img_base64(img_path: str, alt: str = "", width: Optional[int] = None) -> str:
    """Embed a local image as base64 <img> HTML."""
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    w = f' width="{width}"' if width else ""
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}"{w} />'


def html_section(title: str, content_html: str) -> str:
    return f"<section><h2>{title}</h2>\n{content_html}\n</section>"


def save_html_fragment(html: str, out_path: str) -> str:
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
