"""Small evidence-linked OCR pipeline."""

from .pipeline import process_document
from .providers import (
    GLMOCRDirectReader,
    GLMOCRReader,
    LocalReader,
    PaddleOCRVLReader,
    TesseractReader,
)

__all__ = [
    "GLMOCRDirectReader",
    "GLMOCRReader",
    "LocalReader",
    "PaddleOCRVLReader",
    "TesseractReader",
    "process_document",
]
