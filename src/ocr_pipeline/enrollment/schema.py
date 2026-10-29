"""What the enrollment agent extracts: document types, the field catalog, LLM reply models.

The catalog is plain data; the Pydantic models below are the only reply contract. The
prompt shows the LLM the models' JSON schema and every reply is validated against them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# id -> (label, what the classifier should look for, whether it carries enrollment data)
DOCUMENT_TYPES: dict[str, tuple[str, str, bool]] = {
    "practitioner_application": (
        "Practitioner enrollment application",
        "Health plan or CAQH-style form about one practitioner: name, NPI, DOB, licenses, "
        "practice locations, group affiliation.",
        True,
    ),
    "group_application": (
        "Group enrollment application",
        "Form about a group or organization: legal/DBA name, TIN, group NPI, locations.",
        True,
    ),
    "w9": ("W-9", "IRS Form W-9, Request for Taxpayer Identification Number.", True),
    "state_license": (
        "State license",
        "State board license or certificate for a practitioner (MD, DO, NP, PA ...).",
        True,
    ),
    "malpractice_certificate": (
        "Malpractice certificate",
        "Certificate of professional liability (malpractice) insurance.",
        True,
    ),
    "dea_certificate": (
        "DEA registration",
        "DEA controlled substance registration certificate or wallet card.",
        True,
    ),
    "board_certification": ("Board certification", "Specialty board certificate.", True),
    "npi_confirmation": (
        "NPI confirmation",
        "NPPES / CMS NPI notification letter or registry printout.",
        True,
    ),
    "roster_or_cv": ("Roster or CV", "Provider roster, CV or work history.", True),
    "submission_confirmation": (
        "Submission confirmation",
        "A portal 'thank you / we received your application' page; no provider data.",
        False,
    ),
    "cover_letter_or_instructions": (
        "Cover letter or instructions",
        "Fax cover, cover letter, FAQ or generic instructions; no provider data.",
        False,
    ),
    "other": ("Other", "Anything else.", True),
}

Record = Literal["group", "practitioner", "coverage"]
Category = Literal["required", "load_if_provided", "defaulted", "supporting"]


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    record: Record
    category: Category
    description: str
    check: str | None = None  # name in validate.CHECKS
    default: str | None = None
    # TODO: replace with the client's enrollment screen labels from the setup guide.
    target: str = ""


def _spec(key, label, category, description, check=None, default=None):
    return FieldSpec(
        key, label, key.split(".")[0], category, description, check, default,
        target=label,
    )


FIELDS: tuple[FieldSpec, ...] = (
    # Group (PVEN): needed before any practitioner of the group can be enrolled.
    _spec("group.legal_name", "Group legal name", "required",
          "Legal business name of the group or organization (W-9 line 1, insured name)."),
    _spec("group.dba_name", "Group DBA name", "load_if_provided",
          "Doing-business-as / trade name, if different from the legal name."),
    _spec("group.tin", "Group TIN", "required",
          "Group taxpayer ID / EIN, 9 digits (W-9 Part I).", "tin"),
    _spec("group.npi", "Group NPI", "required",
          "Organization (Type 2) NPI of the group, 10 digits.", "npi"),
    _spec("group.taxonomy", "Group taxonomy", "load_if_provided",
          "Group taxonomy code, e.g. 261QP2300X.", "taxonomy"),
    _spec("group.specialty", "Group specialty", "required",
          "Group specialty or organization type in words."),
    _spec("group.tax_classification", "Tax classification", "supporting",
          "W-9 federal tax classification that is checked (C corporation, LLC-P, ...)."),
    # Practitioner (PCEN), required.
    _spec("practitioner.first_name", "First name", "required", "Practitioner first name."),
    _spec("practitioner.middle_name", "Middle name", "load_if_provided",
          "Practitioner middle name or initial, only if printed."),
    _spec("practitioner.last_name", "Last name", "required", "Practitioner last name."),
    _spec("practitioner.credential", "Credential", "required",
          "Degree or title after the name: MD, DO, NP, PA ...", "credential"),
    _spec("practitioner.npi", "Practitioner NPI", "required",
          "Individual (Type 1) NPI of the practitioner, 10 digits.", "npi"),
    _spec("practitioner.dob", "Date of birth", "required", "Practitioner date of birth.", "date"),
    _spec("practitioner.gender", "Gender", "required", "Practitioner gender.", "gender"),
    _spec("practitioner.primary_specialty", "Primary specialty", "required",
          "Practitioner primary specialty (e.g. Family Nurse Practitioner, Emergency Medicine)."),
    _spec("practitioner.taxonomy", "Practitioner taxonomy", "load_if_provided",
          "Practitioner taxonomy code, e.g. 363LF0000X.", "taxonomy"),
    # Loaded if provided; credentialing collects the rest without holding enrollment.
    _spec("practitioner.license_number", "License number", "load_if_provided",
          "State professional license number (not DEA, not NPI)."),
    _spec("practitioner.license_state", "License state", "load_if_provided",
          "State that issued the professional license.", "state"),
    _spec("practitioner.license_expiration", "License expiration", "load_if_provided",
          "Expiration date of the state license.", "date"),
    _spec("practitioner.medicare_id", "Medicare ID", "load_if_provided",
          "Medicare PTAN or Medicare provider number."),
    _spec("practitioner.medicaid_id", "Medicaid ID", "load_if_provided",
          "State Medicaid provider ID (e.g. AHCCCS ID in Arizona)."),
    _spec("practitioner.dea_number", "DEA number", "supporting",
          "DEA registration number: 2 letters + 7 digits.", "dea"),
    _spec("practitioner.dea_expiration", "DEA expiration", "supporting",
          "Date the DEA registration expires.", "date"),
    _spec("practitioner.dea_state", "DEA state", "supporting",
          "State of the DEA registered address.", "state"),
    _spec("practitioner.caqh_id", "CAQH ID", "supporting", "CAQH ProView provider ID."),
    _spec("practitioner.board_certification", "Board certification", "supporting",
          "Specialty board certification, if stated."),
    # Defaulted when not provided; real defaults come from the client's setup guide table.
    _spec("practitioner.panel_status", "Panel status", "defaulted",
          "Accepting new patients / panel open or closed.", "yes_no", "Yes"),
    _spec("practitioner.pcp", "PCP indicator", "defaulted",
          "Whether the practitioner is a primary care provider.", "yes_no", "No"),
    _spec("practitioner.directory_indicator", "Directory indicator", "defaulted",
          "Whether the practitioner appears in the provider directory.", "yes_no", "Yes"),
    _spec("practitioner.languages", "Languages", "defaulted",
          "Languages spoken, comma separated.", None, "English"),
    _spec("practitioner.age_restriction", "Age restriction", "defaulted",
          "Patient age range served, e.g. 2 years and older.", None, "None"),
    _spec("practitioner.gender_restriction", "Gender restriction", "defaulted",
          "Patient gender restriction.", None, "None"),
    _spec("practitioner.telehealth", "Telehealth", "defaulted",
          "Whether telehealth visits are offered.", "yes_no", "No"),
    # Malpractice coverage, evidence for credentialing.
    _spec("coverage.carrier", "Malpractice carrier", "supporting", "Insurance company name."),
    _spec("coverage.policy_number", "Policy number", "supporting", "Malpractice policy number."),
    _spec("coverage.effective_date", "Coverage effective", "supporting",
          "Policy effective date.", "date"),
    _spec("coverage.expiration_date", "Coverage expiration", "supporting",
          "Policy expiration date.", "date"),
    _spec("coverage.per_claim_limit", "Per-claim limit", "supporting",
          "Each claim / each occurrence limit.", "money"),
    _spec("coverage.aggregate_limit", "Aggregate limit", "supporting",
          "Annual aggregate limit.", "money"),
    _spec("coverage.policy_form", "Policy form", "supporting",
          "Claims-made or occurrence, as printed."),
    _spec("coverage.additional_insured", "Additional insured", "supporting",
          "Practitioner named as additional insured on the group policy."),
)
FIELDS_BY_KEY = {spec.key: spec for spec in FIELDS}

LOCATION_ROLES = (
    "service_location",  # where patients are seen; required with a phone
    "billing",           # pay-to / remit / W-9 mailing address
    "mailing",
    "insured",           # address of the insured entity on a malpractice certificate
    "registrant",        # DEA registered address
    "license",           # address printed on a license
    "issuer",            # address of a carrier, board, agency or health plan
    "other",
)
ENTITY_TYPES = (
    "person", "organization", "identifier", "date", "amount", "phone", "email", "other",
)

DocumentType = Literal[tuple(DOCUMENT_TYPES)]
FieldKey = Literal[tuple(FIELDS_BY_KEY)]
Status = Literal["present", "absent", "illegible"]


class _Reply(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FieldReading(_Reply):
    key: FieldKey
    status: Status = Field(description="present: printed on the document; illegible: there but unreadable")
    value: str | None = Field(description="Exactly as printed; null unless present")
    region_ids: list[str] = Field(description="Ids of the OCR regions that contain the value")
    note: str | None = Field(default=None, description="Only when something needs a reviewer's eye")


class LocationReading(_Reply):
    role: Literal[LOCATION_ROLES]
    address: str = Field(description="The whole address exactly as printed, lines joined with ', '")
    address_region_ids: list[str]
    phone: str | None = None
    phone_region_ids: list[str] = []
    fax: str | None = None
    fax_region_ids: list[str] = []
    primary: bool | None = Field(default=None, description="True if marked as the primary location")


class EntityReading(_Reply):
    type: Literal[ENTITY_TYPES]
    label: str = Field(description="What it is, e.g. 'DEA registration number', 'Issue date'")
    value: str = Field(description="Exactly as printed")
    region_ids: list[str]


# The extraction is asked for in parts, in parallel calls, each with only its part's schema.
class FieldsReply(_Reply):
    document_type: DocumentType
    document_type_reason: str = Field(description="A few words: what identifies the document")
    fields: list[FieldReading]

    @field_validator("fields", mode="before")
    @classmethod
    def _catalog_keys_only(cls, readings):
        # Now and then a reading is filed under a key outside the catalog ("locations"); it
        # is no catalog value, so it is left out instead of failing the whole reply.
        if not isinstance(readings, list):
            return readings
        return [reading for reading in readings if not isinstance(reading, dict) or reading.get("key") in FIELDS_BY_KEY]


class LocationsReply(_Reply):
    locations: list[LocationReading]


class LocationsEntitiesReply(LocationsReply):
    entities: list[EntityReading] = Field(
        description="Other identifiers, dates, amounts, people and organizations that fit no catalog field"
    )


class ClassificationReply(_Reply):
    document_type: DocumentType
    confidence: Literal["high", "medium", "low"]
    evidence: str = Field(description="A few words: what on the page identifies the type")


# The case contract below is shared by extraction, the case store, the bundle and the UI.

class Box(BaseModel):
    left: int
    top: int
    right: int
    bottom: int


class Evidence(BaseModel):
    document_id: str
    page: int
    region_ids: list[str]
    box: Box  # page pixels of the OCR page image, origin top-left


class Alternative(BaseModel):
    value: str | None
    evidence: list[Evidence] = []


class Components(BaseModel):
    """Why a value got its confidence; shown to the reviewer as-is."""

    ocr: float | None = None  # min calibrated OCR confidence of the value's regions
    grounding: float | None = None  # how much of the value the cited regions' text holds
    validator: Literal["pass", "fail", "none"] = "none"
    verifier: float | None = None  # independent yes/no check of value vs image crop
    llm: float | None = None  # the LLM's own probability for the value it wrote (weakest token)
    repaired: bool = False  # normalized past an OCR misread ("$10.000.000", "26IQP2300X")
    agreement: int = 1  # distinct documents that carry the same value


class Review(BaseModel):
    state: Literal["pending", "approved", "edited", "rejected"] = "pending"
    by: str | None = None
    at: str | None = None
    reason_code: str | None = None
    reason: str | None = None


class Value(BaseModel):
    """One extracted, defaulted or reviewer-entered value: the unit a reviewer works on."""

    id: str  # unique in a case: the catalog key, or "location-N" / "entity-N"
    key: str  # catalog key, "location" or "entity"
    label: str
    record: Literal["group", "practitioner", "coverage", "other"]
    category: Literal["required", "load_if_provided", "defaulted", "supporting", "other"]
    role: str | None = None  # location role or entity type
    raw_value: str | None = None  # exactly as printed
    value: str | None = None  # normalized
    parts: dict[str, str] | None = None  # split name or address (+ phone)
    source: Literal["document", "default", "reviewer"] = "document"
    evidence: list[Evidence] = []
    alternatives: list[Alternative] = []
    confidence: float | None = None
    components: Components = Components()
    problems: list[str] = []
    note: str | None = None
    route: Literal["quick_verify", "priority_review", "exception"] = "priority_review"
    audit_sample: bool = False
    review: Review = Review()


class DocumentExtraction(BaseModel):
    """What extract_document returns for one document."""

    document_type: str
    document_type_reason: str
    values: list[Value]
    llm: dict  # cost, latency_ms, attempts, prompt_version, model
    problems: list[str] = []


class CaseException(BaseModel):
    id: str
    kind: Literal[
        "missing_document", "missing_field", "duplicate", "ocr", "expired",
        "extraction_failed", "classification", "conflict", "invalid", "group_not_enrolled",
    ]
    severity: Literal["error", "warning", "info"]
    message: str
    value_id: str | None = None
    document_id: str | None = None
    state: Literal["open", "resolved"] = "open"


class DocumentRecord(BaseModel):
    document_id: str
    file_name: str
    sha256: str
    bytes: int
    format: Literal["digital_pdf", "scanned_pdf", "image", "fax"]
    intake_at: str
    duplicate_of: str | None = None
    status: Literal[
        "queued", "classifying", "ocr", "extracting", "extracted", "skipped", "duplicate",
        "failed",
    ] = "queued"
    error: str | None = None
    page_count: int | None = None
    document_type: str | None = None
    document_type_label: str | None = None
    classification: dict = {}  # method, confidence, evidence, probability
    ocr: dict = {}  # session_id, backend_version, elapsed_seconds, status, failures
    timings: dict[str, float] = {}  # seconds per stage, plus the OCR service's steps
    llm: dict = {}


class Case(BaseModel):
    schema_version: int = 1
    case_id: str
    ticket_id: str
    enrollment_type: Literal["group", "practitioner"]
    synthetic: bool = False  # only synthetic cases may use the verifier service
    status: Literal["processing", "review", "submitted", "rejected", "failed"] = "processing"
    created_at: str
    created_by: str
    submitted_at: str | None = None
    submitted_by: str | None = None
    rejection: dict | None = None  # reason_code, reason, by, at
    documents: list[DocumentRecord] = []
    values: list[Value] = []
    exceptions: list[CaseException] = []
    versions: dict = {}  # extraction_prompt, schema


class AuditEvent(BaseModel):
    ts: str
    actor: str
    action: str  # intake, classify, ocr, extract, edit, add, change_source, approve, ...
    case_id: str
    document_id: str | None = None
    value_id: str | None = None
    before: object = None
    after: object = None
    source: Evidence | None = None
    confidence: float | None = None
    reason: str | None = None
    versions: dict = {}
