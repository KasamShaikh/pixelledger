"""Document preprocessing: PDF→images, deskew, denoise, page range."""

from __future__ import annotations

import io
from typing import Optional

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image


def _pdf_to_png_pages(content: bytes, dpi: int = 200) -> list[bytes]:
    pages: list[bytes] = []
    with fitz.open(stream=content, filetype="pdf") as pdf:
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        for page in pdf:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pages.append(pix.tobytes("png"))
    return pages


def _image_to_png(content: bytes) -> bytes:
    img = Image.open(io.BytesIO(content)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _deskew(png_bytes: bytes) -> bytes:
    img = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return png_bytes
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.5:
        return png_bytes
    (h, w) = img.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    out = io.BytesIO()
    Image.fromarray(rotated).save(out, format="PNG")
    return out.getvalue()


def _denoise(png_bytes: bytes) -> bytes:
    img = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
    den = cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)
    out = io.BytesIO()
    Image.fromarray(den).save(out, format="PNG")
    return out.getvalue()


def _grayscale(png_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(png_bytes)).convert("L")
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def preprocess(
    content: bytes,
    mime_type: str,
    *,
    deskew: bool = False,
    denoise: bool = False,
    grayscale: bool = False,
    page_range: Optional[tuple[int, int]] = None,
    dpi: int = 200,
) -> list[bytes]:
    """Return per-page PNG bytes after applying optional preprocessing."""
    if mime_type == "application/pdf" or content[:4] == b"%PDF":
        pages = _pdf_to_png_pages(content, dpi=dpi)
    else:
        pages = [_image_to_png(content)]

    if page_range is not None:
        start, end = page_range
        start = max(start, 1)
        end = min(end, len(pages))
        pages = pages[start - 1 : end]

    if grayscale:
        pages = [_grayscale(p) for p in pages]
    if denoise:
        pages = [_denoise(p) for p in pages]
    if deskew:
        pages = [_deskew(p) for p in pages]

    return pages
