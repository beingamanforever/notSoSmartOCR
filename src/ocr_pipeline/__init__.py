"""Small evidence-linked OCR pipeline."""

from .controls import GeometricControlStage, detect_controls
from .cross_page_tables import (
    CrossPageTableStage,
    TableContinuationClassifier,
    TableContinuationPrediction,
)
from .evidence_layout import EvidenceLayoutStage
from .falcon import FalconOCRReader, FalconOCRServiceReader
from .falcon_presentation import FalconPresentationReader
from .faint_text import FaintTinyTextStage
from .handwriting import HandwritingStage
from .orientation import DocTROrientationDetector, OrientationReader
from .pipeline import process_document
from .preprocessing import RoutedTesseractReader, locate_dark_frame
from .pilot import PilotOCRReader
from .providers import (
    GLMOCRDirectReader,
    GLMOCRReader,
    GraniteDoclingReader,
    LocalReader,
    MinistralOCRReader,
    MinistralOCRServiceReader,
    MinistralStructuredImageCall,
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
    "CrossPageTableStage",
    "EvidenceRiskStage",
    "EvidenceLayoutStage",
    "FalconOCRReader",
    "FalconOCRServiceReader",
    "FalconPresentationReader",
    "FaintTinyTextStage",
    "GeometricControlStage",
    "GraniteDoclingReader",
    "HandwritingStage",
    "LocalReader",
    "MinistralOCRReader",
    "MinistralOCRServiceReader",
    "MinistralStructuredImageCall",
    "NemotronOCRV2Reader",
    "OrientationReader",
    "PaddleOCRVLReader",
    "Phi4HandwritingReader",
    "Phi4HandwritingServiceReader",
    "PilotOCRReader",
    "RoutedTesseractReader",
    "TesseractReader",
    "TableChallenger",
    "TableContinuationClassifier",
    "TableContinuationPrediction",
    "TatrTableExtractor",
    "TatrTableStage",
    "detect_controls",
    "locate_dark_frame",
    "process_document",
]
