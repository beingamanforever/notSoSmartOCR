"""Small evidence-linked OCR pipeline."""

from .pipeline import process_document
from .providers import LocalReader, TesseractReader

__all__ = ["LocalReader", "TesseractReader", "process_document"]
