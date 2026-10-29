"""Field-level validators and normalizers for enrollment form data.

Every `check_*` function takes a raw string and returns
`(normalized value or None, problem message or None)`. They never raise on
any string input, including empty, whitespace, or unicode garbage.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Callable

import usaddress

Check = Callable[[str], tuple[str | None, str | None]]


def _luhn_check_digit(payload: str) -> int:
    """Standard Luhn check digit for a numeric string (digit to be appended)."""
    total = 0
    for index, char in enumerate(reversed(payload)):
        digit = int(char)
        if index % 2 == 0:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return (10 - total % 10) % 10


REPAIRED = "read a letter as a digit"
# OCR confuses these letters with digits; only read them as digits inside a digit run.
_LOOKALIKES = str.maketrans({"O": "0", "o": "0", "Q": "0", "D": "0", "I": "1", "l": "1", "|": "1", "Z": "2", "S": "5", "B": "8"})


def _as_digits(raw: str) -> tuple[str, bool]:
    """The digits of an identifier, and whether a lookalike letter was read as a digit."""
    repaired = re.sub(
        r"[0-9OoQDIl|ZSB]+",
        lambda match: match.group().translate(_LOOKALIKES) if re.search(r"\d", match.group()) else match.group(),
        raw,
    )
    return re.sub(r"\D", "", repaired), repaired != raw


def check_npi(raw: str) -> tuple[str | None, str | None]:
    digits, repaired = _as_digits(raw)
    if len(digits) != 10:
        return None, "NPI must be 10 digits"
    if _luhn_check_digit("80840" + digits[:9]) != int(digits[9]):
        return None, "NPI check digit fails"
    return digits, REPAIRED if repaired else None


def check_tin(raw: str) -> tuple[str | None, str | None]:
    digits, repaired = _as_digits(raw)
    if len(digits) != 9:
        return None, "TIN must be 9 digits"
    return f"{digits[:2]}-{digits[2:]}", REPAIRED if repaired else None


def check_dea(raw: str) -> tuple[str | None, str | None]:
    text = re.sub(r"\s+", "", raw).upper()
    # The two leading letters are real letters (the second is often an O); only the
    # seven digits after them can hold a lookalike.
    digits = text[2:].translate(_LOOKALIKES)
    repaired = digits != text[2:]
    text = text[:2] + digits
    if not re.fullmatch(r"[A-Z]{2}\d{7}", text):
        return None, "DEA must be 2 letters and 7 digits"
    digits = [int(char) for char in text[2:]]
    checksum = (digits[0] + digits[2] + digits[4]) + 2 * (digits[1] + digits[3] + digits[5])
    if checksum % 10 != digits[6]:
        return None, "DEA check digit fails"
    return text, REPAIRED if repaired else None


_ISO_DATE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_NUMERIC_DATE = re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4}|\d{2})$")
_MONTH_NAME_DATE = re.compile(r"^([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})$")
_TRAILING_TIME = re.compile(r",?\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:[ap]\.?\s?m\.?)?\s*$", re.I)


def _build_date(year: int, month: int, day: int) -> tuple[str | None, str | None]:
    try:
        return date(year, month, day).isoformat(), None
    except ValueError:
        return None, "impossible date"


def check_date(raw: str) -> tuple[str | None, str | None]:
    exact = _exact_date(raw)
    if exact[0] is not None:
        return exact
    # "7/1/2027 12:01 a.m. unless premium is not paid": the date, flagged for a look.
    for pattern in (_ISO_DATE, _NUMERIC_DATE, _MONTH_NAME_DATE):
        match = re.search(r"(?<!\d)" + pattern.pattern.strip("^$") + r"(?!\d)", raw)
        if match:
            found = _exact_date(match.group())
            if found[0] is not None:
                return found[0], "text around the date ignored"
    return exact


def _exact_date(raw: str) -> tuple[str | None, str | None]:
    text = raw.strip()
    if not text:
        return None, "unrecognized date"
    text = _TRAILING_TIME.sub("", text).strip()

    match = _ISO_DATE.match(text)
    if match:
        year, month, day = (int(group) for group in match.groups())
        return _build_date(year, month, day)

    match = _NUMERIC_DATE.match(text)
    if match:
        month, day, year_text = match.groups()
        year = int(year_text) if len(year_text) == 4 else 2000 + int(year_text)
        return _build_date(year, int(month), int(day))

    match = _MONTH_NAME_DATE.match(text)
    if match:
        month_name, day, year = match.groups()
        for month_format in ("%b", "%B"):
            try:
                month = datetime.strptime(month_name, month_format).month
                return _build_date(int(year), month, int(day))
            except ValueError:
                continue
        return None, "unrecognized date"

    return None, "unrecognized date"


_EXTENSION = re.compile(r"\s*(?:x|ext\.?|extension)\s*(\d+)\s*$", re.I)


def check_phone(raw: str) -> tuple[str | None, str | None]:
    text = raw.strip()
    extension_match = _EXTENSION.search(text)
    extension = ""
    if extension_match:
        extension = extension_match.group(1)
        text = text[: extension_match.start()]
    digits = re.sub(r"\D", "", text)
    if len(digits) == 11 and digits[0] == "1":
        digits = digits[1:]
    if len(digits) != 10:
        return None, "phone must be 10 digits"
    formatted = f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    if extension:
        formatted += f" x{extension}"
    return formatted, None


def check_zip(raw: str) -> tuple[str | None, str | None]:
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 5:
        return digits, None
    if len(digits) == 9:
        return f"{digits[:5]}-{digits[5:]}", None
    return None, "zip must be 5 or 9 digits"


_STATE_NAMES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
    "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC", "puerto rico": "PR", "virgin islands": "VI",
    "guam": "GU", "american samoa": "AS", "northern mariana islands": "MP",
}
_STATE_CODES = set(_STATE_NAMES.values())


def check_state(raw: str) -> tuple[str | None, str | None]:
    text = raw.strip()
    if text.upper() in _STATE_CODES:
        return text.upper(), None
    code = _STATE_NAMES.get(text.lower())
    if code:
        return code, None
    return None, "unrecognized state"


def check_taxonomy(raw: str) -> tuple[str | None, str | None]:
    text = re.sub(r"\s+", "", raw).upper()
    head = text[:3].translate(_LOOKALIKES)  # the first three characters are always digits
    fixed = head + text[3:]
    if re.fullmatch(r"\d{3}[A-Z0-9]{6}X", fixed):
        return fixed, REPAIRED if fixed != text else None
    return None, "invalid taxonomy code"


_CREDENTIALS = (
    "MD", "DO", "NP", "PA", "PA-C", "APRN", "FNP", "FNP-C", "CNM", "CRNA",
    "DPM", "DC", "OD", "DDS", "DMD", "PhD", "PsyD", "LCSW", "LPC", "LMFT",
    "RN", "PT", "OT", "SLP", "AuD", "RD",
)
_CREDENTIAL_LOOKUP = {credential.upper(): credential for credential in _CREDENTIALS}


def check_credential(raw: str) -> tuple[str | None, str | None]:
    cleaned = raw.strip().strip("()").replace(".", "").strip().upper()
    canonical = _CREDENTIAL_LOOKUP.get(cleaned)
    if canonical:
        return canonical, None
    return None, "unknown credential"


def check_gender(raw: str) -> tuple[str | None, str | None]:
    text = raw.strip().lower()
    if text in {"f", "female", "woman"}:
        return "F", None
    if text in {"m", "male", "man"}:
        return "M", None
    if text in {"x", "non-binary"}:
        return "X", None
    return None, "unrecognized gender"


_MONEY_MULTIPLIERS = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}


def check_money(raw: str) -> tuple[str | None, str | None]:
    lowered = raw.strip().lower()
    multiplier = 1
    for word, value in _MONEY_MULTIPLIERS.items():
        if word in lowered:
            multiplier = value
            lowered = lowered.replace(word, "")
            break
    cleaned = re.sub(r"[^0-9.]", "", lowered)
    if not cleaned:
        return None, "invalid amount"

    problem = None
    dot_count = cleaned.count(".")
    if dot_count > 1:
        cleaned = cleaned.replace(".", "")
        problem = "separator misread"
    elif dot_count == 1:
        whole, fraction = cleaned.split(".")
        if len(fraction) == 3:
            cleaned = whole + fraction
            problem = "separator misread"

    try:
        amount = float(cleaned) * multiplier
    except ValueError:
        return None, "invalid amount"

    if amount == int(amount):
        return str(int(amount)), problem
    return f"{amount:.2f}", problem


def check_yes_no(raw: str) -> tuple[str | None, str | None]:
    text = raw.strip().lower()
    if text in {"yes", "y", "true", "checked", "x"}:
        return "Yes", None
    if text in {"no", "n", "false", "unchecked"}:
        return "No", None
    return None, "unrecognized value"


CHECKS: dict[str, Check] = {
    "npi": check_npi,
    "tin": check_tin,
    "dea": check_dea,
    "date": check_date,
    "phone": check_phone,
    "zip": check_zip,
    "state": check_state,
    "taxonomy": check_taxonomy,
    "credential": check_credential,
    "gender": check_gender,
    "money": check_money,
    "yes_no": check_yes_no,
}


_TRAILING_PAREN = re.compile(r"\(([^)]*)\)\s*$")


def split_person_name(raw: str) -> tuple[dict | None, str | None]:
    text = raw.strip()
    if not text:
        return None, "name missing"
    was_upper = text.isupper()

    credential = ""
    paren_match = _TRAILING_PAREN.search(text)
    if paren_match:
        candidate_credential, _ = check_credential(paren_match.group(1))
        if candidate_credential:
            credential = candidate_credential
            text = text[: paren_match.start()].strip()

    if "," in text:
        before, after = text.split(",", 1)
        before, after = before.strip(), after.strip()
        whole_after_credential, _ = check_credential(after) if not credential else (None, None)
        if whole_after_credential:
            # "First [Middle] Last, Credential"
            credential = whole_after_credential
            tokens = before.split()
            first, middle, last = _split_natural_order(tokens)
        else:
            # "Last, First [Middle] [Credential]"
            last = before
            tokens = after.split()
            if tokens and not credential:
                trailing_credential, _ = check_credential(tokens[-1])
                if trailing_credential:
                    credential = trailing_credential
                    tokens = tokens[:-1]
            first = tokens[0] if tokens else ""
            middle = " ".join(tokens[1:]) if len(tokens) > 1 else ""
    else:
        tokens = text.split()
        if tokens and not credential:
            trailing_credential, _ = check_credential(tokens[-1])
            if trailing_credential:
                credential = trailing_credential
                tokens = tokens[:-1]
        first, middle, last = _split_natural_order(tokens)

    if was_upper:
        first, middle, last = first.title(), middle.title(), last.title()
    return {"first": first, "middle": middle, "last": last, "credential": credential}, None


def _split_natural_order(tokens: list[str]) -> tuple[str, str, str]:
    if len(tokens) >= 2:
        return tokens[0], " ".join(tokens[1:-1]), tokens[-1]
    if len(tokens) == 1:
        return tokens[0], "", ""
    return "", "", ""


_ADDRESS_LINE1_PREFIXES = ("AddressNumber", "StreetName", "USPSBox", "LandmarkName")
_ADDRESS_LINE2_PREFIXES = ("Occupancy", "Subaddress")


def split_address(raw: str) -> tuple[dict | None, str | None]:
    text = raw.replace("\n", ", ")
    try:
        tagged, _ = usaddress.tag(text)
    except usaddress.RepeatedLabelError:
        return None, "address ambiguous"
    except Exception:
        return None, "address ambiguous"

    line1_parts: list[str] = []
    line2_parts: list[str] = []
    city = ""
    state = ""
    zip_code = ""
    for label, value in tagged.items():
        if label == "Recipient":
            continue
        if label.startswith(_ADDRESS_LINE1_PREFIXES):
            line1_parts.append(value)
        elif label.startswith(_ADDRESS_LINE2_PREFIXES):
            line2_parts.append(value)
        elif label == "PlaceName":
            city = value
        elif label == "StateName":
            state, _ = check_state(value)
        elif label == "ZipCode":
            zip_code, _ = check_zip(value)

    if not city or not state or not zip_code:
        return None, "address incomplete"

    line1 = " ".join(line1_parts)
    line2 = " ".join(line2_parts).replace("# ", "#")
    return {"line1": line1, "line2": line2, "city": city, "state": state, "zip": zip_code}, None
