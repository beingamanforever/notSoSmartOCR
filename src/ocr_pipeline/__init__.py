"""Small evidence-linked OCR pipeline."""

from .cross_page_tables import (
    CrossPageTableStage,
    TableContinuationClassifier,
    TableContinuationPrediction,
)
from .evidence_layout import EvidenceLayoutStage
from .falcon_layout import FalconLayoutReader
from .orientation import DocTROrientationDetector, OrientationReader
from .pipeline import falcon_pipeline, process_document
from .preprocessing import RoutedTesseractReader, locate_dark_frame
from .providers import LocalReader, ReaderError, TesseractReader
from .risk import EvidenceRiskStage

__all__ = [
    "CrossPageTableStage",
    "DocTROrientationDetector",
    "EvidenceLayoutStage",
    "EvidenceRiskStage",
    "FalconLayoutReader",
    "LocalReader",
    "OrientationReader",
    "ReaderError",
    "RoutedTesseractReader",
    "TableContinuationClassifier",
    "TableContinuationPrediction",
    "TesseractReader",
    "falcon_pipeline",
    "locate_dark_frame",
    "process_document",
]
