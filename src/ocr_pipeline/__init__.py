"""Small evidence-linked OCR pipeline."""

from .controls import GeometricControlStage, detect_controls
from .handwriting import HandwritingStage
from .orientation import DocTROrientationDetector, OrientationReader
from .pipeline import process_document
from .preprocessing import RoutedTesseractReader, locate_dark_frame
from .providers import (
    GLMOCRDirectReader,
    GLMOCRReader,
    GraniteDoclingReader,
    LocalReader,
    MinistralOCRReader,
    NemotronOCRV2Reader,
    PaddleOCRVLReader,
    Phi4HandwritingReader,
    Phi4HandwritingServiceReader,
    TesseractReader,
)
from .risk import EvidenceRiskStage
from .tables import TableChallenger, TatrTableExtractor, TatrTableStage

__all__ = [
    "GLMOCRDirectReader",
    "GLMOCRReader",
    "DocTROrientationDetector",
    "EvidenceRiskStage",
    "GeometricControlStage",
    "GraniteDoclingReader",
    "HandwritingStage",
    "LocalReader",
    "MinistralOCRReader",
    "NemotronOCRV2Reader",
    "OrientationReader",
    "PaddleOCRVLReader",
    "Phi4HandwritingReader",
    "Phi4HandwritingServiceReader",
    "RoutedTesseractReader",
    "TesseractReader",
    "TableChallenger",
    "TatrTableExtractor",
    "TatrTableStage",
    "detect_controls",
    "locate_dark_frame",
    "process_document",
]
