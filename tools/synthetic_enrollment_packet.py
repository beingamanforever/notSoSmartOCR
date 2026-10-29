"""Build a fictional provider enrollment packet for demos and extraction tests.

Every person, organisation, number and address here is invented. NPIs, the TIN and
the DEA number are made checksum-valid so validators can be exercised; one handwritten
group NPI is deliberately transposed so the packet carries a real conflict.

Writes six documents in the formats a health plan receives (born-digital PDF, scanned
PDF, phone photo, fax TIFF) plus truth.json with the expected field values:

    python tools/synthetic_enrollment_packet.py OUT_DIR
"""

from __future__ import annotations

import io
import json
import random
import sys
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

PAGE = fitz.paper_rect("letter")
MARGIN = 42
HAND_FONT = "/System/Library/Fonts/Noteworthy.ttc"
INK = (0.08, 0.16, 0.45)

GROUP = {
    "legal_name": "Saguaro Valley Family Medicine PLLC",
    "dba_name": "Saguaro Valley Primary Care",
    "tin": "86-4419027",
    "group_npi": "1093817462",
    "taxonomy": "261QP2300X",
    "specialty": "Primary Care Clinic",
    "billing_address": "PO Box 30417, Phoenix, AZ 85046",
    "locations": [
        {
            "line1": "4420 W Cactus Blossom Dr",
            "line2": "Ste 210",
            "city": "Mesa",
            "state": "AZ",
            "zip": "85201",
            "phone": "(480) 555-0147",
            "fax": "(480) 555-0148",
            "hours": "Mon-Fri 8:00-5:00",
            "primary": True,
        },
        {
            "line1": "1185 N Ocotillo Pkwy",
            "line2": "Building B",
            "city": "Chandler",
            "state": "AZ",
            "zip": "85224",
            "phone": "(480) 555-0192",
            "fax": "",
            "hours": "Tue-Sat 9:00-6:00",
            "primary": False,
        },
    ],
}
PRACTITIONER = {
    "first_name": "Maya",
    "middle_name": "Renee",
    "last_name": "Okafor",
    "credentials": "NP",
    "npi": "1548293764",
    "dob": "1986-04-17",
    "gender": "F",
    "primary_specialty": "Family Nurse Practitioner",
    "taxonomy": "363LF0000X",
    "license_number": "AP-238114",
    "license_state": "AZ",
    "license_expiration": "2027-01-15",
    "medicaid_id": "784512",
    "medicare_id": "",
    "caqh_id": "16452389",
    "dea_number": "MO4826155",
    "dea_expiration": "2028-06-30",
    "languages": ["English", "Spanish"],
    "telehealth": True,
    "accepting_new_patients": True,
    "pcp": True,
    "age_range": "2 years and older",
}
COVERAGE = {
    "carrier": "Copper State Medical Mutual",
    "policy_number": "CSM-2219047",
    "effective": "2026-01-01",
    "expiration": "2027-01-01",
    "each_claim": "$1,000,000",
    "aggregate": "$3,000,000",
}
# The handwritten group form carries the group NPI with two digits swapped.
TRANSPOSED_GROUP_NPI = "1093817426"


class Form:
    """A tiny top-to-bottom form writer on one PyMuPDF page."""

    def __init__(self, doc: fitz.Document, hand: fitz.Font | None = None):
        self.doc, self.hand = doc, hand
        self.page = doc.new_page(width=PAGE.width, height=PAGE.height)
        self.y = MARGIN

    def new_page(self) -> None:
        self.page = self.doc.new_page(width=PAGE.width, height=PAGE.height)
        self.y = MARGIN

    def title(self, text: str, sub: str = "") -> None:
        self.page.insert_text((MARGIN, self.y + 18), text, fontname="hebo", fontsize=16)
        if sub:
            self.page.insert_text((MARGIN, self.y + 34), sub, fontname="helv", fontsize=9)
        self.y += 48

    def section(self, text: str) -> None:
        bar = fitz.Rect(MARGIN, self.y, PAGE.width - MARGIN, self.y + 16)
        self.page.draw_rect(bar, color=None, fill=(0.85, 0.88, 0.93))
        self.page.insert_text((MARGIN + 4, self.y + 12), text, fontname="hebo", fontsize=9.5)
        self.y += 22

    def row(self, fields: list[tuple[str, str, float]], handwritten: bool = False) -> None:
        """fields: (label, value, width fraction); fractions in a row sum to 1."""
        x, width = MARGIN, PAGE.width - 2 * MARGIN
        for label, value, share in fields:
            box = fitz.Rect(x, self.y, x + width * share - 4, self.y + 30)
            self.page.draw_rect(box, color=(0.35, 0.35, 0.35), width=0.6)
            self.page.insert_text((box.x0 + 3, box.y0 + 8), label, fontname="helv", fontsize=6.5)
            if value and handwritten and self.hand:
                self.page.insert_text(
                    (box.x0 + 5, box.y0 + 25), value, fontname="hand", fontfile=HAND_FONT,
                    fontsize=12, color=INK,
                )
            elif value:
                self.page.insert_text((box.x0 + 4, box.y0 + 23), value, fontname="helv", fontsize=10)
            x += width * share
        self.y += 34

    def checks(self, label: str, options: list[str], selected: set[str], handwritten: bool = False) -> None:
        self.page.insert_text((MARGIN, self.y + 10), label, fontname="helv", fontsize=8)
        x = MARGIN + 8 + fitz.get_text_length(label, "helv", 8)
        for option in options:
            box = fitz.Rect(x, self.y + 2, x + 9, self.y + 11)
            self.page.draw_rect(box, color=(0, 0, 0), width=0.7)
            if option in selected:
                mark_color = INK if handwritten else (0, 0, 0)
                self.page.draw_line(box.tl + (1.5, 4.5), box.bl + (3.5, -1.5), color=mark_color, width=1.4)
                self.page.draw_line(box.bl + (3.5, -1.5), box.tr + (1, -1), color=mark_color, width=1.4)
            self.page.insert_text((x + 13, self.y + 10), option, fontname="helv", fontsize=8)
            x += 22 + fitz.get_text_length(option, "helv", 8)
        self.y += 18

    def table(self, headers: list[str], rows: list[list[str]], shares: list[float], handwritten: bool = False) -> None:
        width = PAGE.width - 2 * MARGIN
        for index, cells in enumerate([headers, *rows]):
            x, height = MARGIN, 18 if index == 0 else 24
            for cell, share in zip(cells, shares):
                box = fitz.Rect(x, self.y, x + width * share, self.y + height)
                fill = (0.93, 0.94, 0.96) if index == 0 else None
                self.page.draw_rect(box, color=(0.35, 0.35, 0.35), fill=fill, width=0.6)
                if index == 0:
                    self.page.insert_text((box.x0 + 3, box.y0 + 12), cell, fontname="hebo", fontsize=7)
                elif handwritten and self.hand:
                    self.page.insert_text(
                        (box.x0 + 3, box.y0 + 17), cell, fontname="hand", fontfile=HAND_FONT,
                        fontsize=10, color=INK,
                    )
                else:
                    self.page.insert_text((box.x0 + 3, box.y0 + 16), cell, fontname="helv", fontsize=8.5)
                x += width * share
            self.y += height
        self.y += 10

    def para(self, text: str, size: float = 8) -> None:
        rect = fitz.Rect(MARGIN, self.y, PAGE.width - MARGIN, self.y + 200)
        spare = self.page.insert_textbox(rect, text, fontname="helv", fontsize=size)
        self.y += 200 - spare + 6

    def signature(self, name: str, date: str) -> None:
        self.page.draw_line((MARGIN, self.y + 30), (MARGIN + 220, self.y + 30), width=0.6)
        self.page.insert_text((MARGIN + 6, self.y + 26), name, fontname="hand", fontfile=HAND_FONT, fontsize=16, color=INK)
        self.page.insert_text((MARGIN, self.y + 40), "Signature", fontname="helv", fontsize=7)
        self.page.draw_line((MARGIN + 300, self.y + 30), (MARGIN + 420, self.y + 30), width=0.6)
        self.page.insert_text((MARGIN + 306, self.y + 26), date, fontname="hand", fontfile=HAND_FONT, fontsize=12, color=INK)
        self.page.insert_text((MARGIN + 300, self.y + 40), "Date", fontname="helv", fontsize=7)
        self.y += 52


def us_date(iso: str) -> str:
    year, month, day = iso.split("-")
    return f"{month}/{day}/{year}"


def address(location: dict) -> str:
    return f"{location['line1']}, {location['line2']}, {location['city']}, {location['state']} {location['zip']}"


def practitioner_application() -> fitz.Document:
    p, g = PRACTITIONER, GROUP
    doc = fitz.open()
    form = Form(doc)
    form.title("Practitioner Enrollment Application", "Desert Sky Health Plan - Provider Network Management - fictional sample")
    form.section("Section 1. Practitioner identification")
    form.row([("Last name", p["last_name"], 0.34), ("First name", p["first_name"], 0.33), ("Middle name", p["middle_name"], 0.23), ("Degree", p["credentials"], 0.10)])
    form.row([("Individual NPI", p["npi"], 0.25), ("Date of birth (MM/DD/YYYY)", us_date(p["dob"]), 0.25), ("SSN (last 4)", "XXX-XX-4417", 0.2), ("CAQH ID", p["caqh_id"], 0.3)])
    form.checks("Gender:", ["Female", "Male", "Non-binary", "Prefer not to say"], {"Female"})
    form.row([("Primary specialty", p["primary_specialty"], 0.5), ("Taxonomy code", p["taxonomy"], 0.25), ("Secondary specialty", "", 0.25)])
    form.section("Section 2. Licensure and certifications")
    form.row([("State license number", p["license_number"], 0.3), ("State", p["license_state"], 0.1), ("License expiration", us_date(p["license_expiration"]), 0.25), ("DEA number", p["dea_number"], 0.35)])
    form.row([("Medicaid (AHCCCS) provider ID", p["medicaid_id"], 0.35), ("Medicare PTAN", p["medicare_id"], 0.3), ("Board certification", "AANP Family NP", 0.35)])
    form.section("Section 3. Affiliated group")
    form.row([("Group legal name", g["legal_name"], 0.5), ("Group TIN", g["tin"], 0.22), ("Group NPI", g["group_npi"], 0.28)])
    form.section("Section 4. Service locations")
    form.table(
        ["Address", "City", "State", "ZIP", "Phone", "Primary"],
        [[f"{loc['line1']}, {loc['line2']}", loc["city"], loc["state"], loc["zip"], loc["phone"], "Yes" if loc["primary"] else "No"] for loc in g["locations"]],
        [0.38, 0.14, 0.07, 0.1, 0.19, 0.12],
    )
    form.section("Section 5. Directory information")
    form.checks("Accepting new patients:", ["Yes", "No"], {"Yes"})
    form.checks("Primary care provider (PCP):", ["Yes", "No"], {"Yes"})
    form.checks("Telehealth available:", ["Yes", "No"], {"Yes"})
    form.row([("Languages spoken", ", ".join(p["languages"]), 0.5), ("Patient age range", p["age_range"], 0.3), ("Gender restriction", "None", 0.2)])
    form.new_page()
    form.section("Section 6. Attestation")
    form.para(
        "I attest that the information in this application is true, correct and complete to the best of my "
        "knowledge. I authorize the health plan to verify the information with primary sources and understand "
        "that misstatements may result in denial of participation. I will notify the plan within 30 days of any "
        "change to my licensure, DEA registration, malpractice coverage or practice locations."
    )
    form.signature("Maya R. Okafor", "10/12/2026")
    return doc


def group_enrollment_form() -> fitz.Document:
    g = GROUP
    doc = fitz.open()
    form = Form(doc, hand=True)
    form.title("Group / Organization Enrollment Form", "Desert Sky Health Plan - complete in ink - fictional sample")
    form.section("A. Organization")
    form.row([("Legal business name (as on W-9)", g["legal_name"], 0.6), ("Tax ID (TIN)", g["tin"], 0.4)], handwritten=True)
    form.row([("Doing business as (DBA)", g["dba_name"], 0.6), ("Group NPI (Type 2)", TRANSPOSED_GROUP_NPI, 0.4)], handwritten=True)
    form.row([("Group specialty / taxonomy", f"{g['specialty']} {g['taxonomy']}", 0.6), ("Billing address", g["billing_address"], 0.4)], handwritten=True)
    form.section("B. Practice locations (list all)")
    form.table(
        ["Street address", "City", "St", "ZIP", "Phone", "Hours"],
        [[f"{loc['line1']} {loc['line2']}", loc["city"], loc["state"], loc["zip"], loc["phone"], loc["hours"]] for loc in g["locations"]],
        [0.34, 0.14, 0.06, 0.1, 0.18, 0.18],
        handwritten=True,
    )
    form.section("C. Contact")
    form.row([("Credentialing contact", "Luis Ferreira", 0.4), ("Phone", "(480) 555-0150", 0.3), ("Email", "credentialing@example.org", 0.3)], handwritten=True)
    form.signature("L. Ferreira", "10/12/2026")
    return doc


def w9() -> fitz.Document:
    g = GROUP
    doc = fitz.open()
    form = Form(doc, hand=True)
    form.title("Form W-9  Request for Taxpayer Identification Number and Certification", "(Rev. March 2024) Department of the Treasury - Internal Revenue Service - fictional sample")
    form.row([("1 Name of entity/individual", g["legal_name"], 1.0)])
    form.row([("2 Business name/disregarded entity name, if different from above", g["dba_name"], 1.0)])
    form.checks("3a Federal tax classification:", ["Individual/sole proprietor", "C corporation", "S corporation", "Partnership", "Trust/estate", "LLC"], {"LLC"})
    form.row([("LLC tax classification (C, S, or P)", "P", 0.4), ("4 Exemptions", "", 0.6)])
    form.row([("5 Address (number, street, and apt. or suite no.)", "PO Box 30417", 0.6), ("Requester's name and address (optional)", "Desert Sky Health Plan", 0.4)])
    form.row([("6 City, state, and ZIP code", "Phoenix, AZ 85046", 1.0)])
    form.section("Part I  Taxpayer Identification Number (TIN)")
    form.row([("Social security number", "", 0.4), ("Employer identification number", g["tin"], 0.6)])
    form.section("Part II  Certification")
    form.para("Under penalties of perjury, I certify that the number shown on this form is my correct taxpayer identification number and that I am a U.S. person.")
    form.signature("L. Ferreira", "01/05/2026")
    return doc


def license_certificate() -> fitz.Document:
    p = PRACTITIONER
    doc = fitz.open()
    page = doc.new_page(width=PAGE.height, height=PAGE.width)  # landscape wall certificate
    frame = page.rect + (30, 30, -30, -30)
    page.draw_rect(frame, color=(0.45, 0.32, 0.1), width=4)
    page.draw_rect(frame + (8, 8, -8, -8), color=(0.45, 0.32, 0.1), width=1)
    lines = [
        ("ARIZONA STATE BOARD OF NURSING (fictional sample)", "hebo", 20, 90),
        ("Advanced Practice Registered Nurse License", "tiro", 18, 125),
        ("This certifies that", "tiro", 13, 175),
        (f"{p['first_name'].upper()} {p['middle_name'][0]} {p['last_name'].upper()}, {p['credentials']}", "hebo", 24, 215),
        ("is licensed to practice as a Family Nurse Practitioner in the State of Arizona", "tiro", 13, 255),
        (f"License No. {p['license_number']}          Issued 03/01/2023          Expires {us_date(p['license_expiration'])}", "helv", 13, 315),
        ("Verify at azbn.example.gov", "helv", 10, 345),
    ]
    for text, font, size, y in lines:
        width = fitz.get_text_length(text, font, size)
        page.insert_text(((page.rect.width - width) / 2, y + 40), text, fontname=font, fontsize=size)
    page.draw_circle((page.rect.width - 140, page.rect.height - 120), 42, color=(0.6, 0.45, 0.1), width=2)
    page.insert_text((page.rect.width - 168, page.rect.height - 116), "SEAL", fontname="hebo", fontsize=14, color=(0.6, 0.45, 0.1))
    return doc


def malpractice_certificate() -> fitz.Document:
    c, g, p = COVERAGE, GROUP, PRACTITIONER
    doc = fitz.open()
    form = Form(doc)
    page = form.page
    page.insert_text((MARGIN, 24), "FROM: COPPER STATE MEDICAL MUTUAL  FAX (602) 555-0110   10/09/2026 14:22   P. 1/1", fontname="cour", fontsize=8)
    form.y = 40
    form.title("Certificate of Professional Liability Insurance", f"{c['carrier']} - 100 N Central Ave, Phoenix, AZ 85004 - fictional sample")
    form.row([("Insurance company", c["carrier"], 0.5), ("Policy number", c["policy_number"], 0.25), ("Policy form", "Claims-made", 0.25)])
    form.row([("Named insured", g["legal_name"], 0.5), ("Insured address", "4420 W Cactus Blossom Dr Ste 210, Mesa, AZ 85201", 0.5)])
    form.row([("Effective date", us_date(c["effective"]), 0.25), ("Expiration date", us_date(c["expiration"]), 0.25), ("Each claim", c["each_claim"], 0.25), ("Aggregate", c["aggregate"], 0.25)])
    form.para(
        f"It is agreed that the policy is amended to include {p['first_name']} {p['middle_name'][0]}. {p['last_name']}, {p['credentials']} "
        "as an additional insured, effective on the policy effective date, while acting within the scope of duties "
        "for the named insured. This certificate is issued as a matter of information only and confers no rights "
        "upon the certificate holder.",
        size=9,
    )
    form.signature("A. Whitfield", "10/09/2026")
    return doc


def dea_certificate() -> fitz.Document:
    p, loc = PRACTITIONER, GROUP["locations"][0]
    doc = fitz.open()
    form = Form(doc)
    form.y = 160
    form.title("CONTROLLED SUBSTANCE REGISTRATION CERTIFICATE", "UNITED STATES DEPARTMENT OF JUSTICE - DRUG ENFORCEMENT ADMINISTRATION - fictional sample")
    form.row([("DEA REGISTRATION NUMBER", p["dea_number"], 0.34), ("THIS REGISTRATION EXPIRES", us_date(p["dea_expiration"]).replace("/", "-"), 0.33), ("FEE PAID", "$888", 0.33)])
    form.row([("SCHEDULES", "2,2N,3,3N,4,5", 0.34), ("BUSINESS ACTIVITY", "MLP-NURSE PRACTITIONER", 0.33), ("ISSUE DATE", "07-01-2025", 0.33)])
    form.para(f"{p['last_name'].upper()}, {p['first_name'].upper()} ({p['credentials']})\n{loc['line1'].upper()} {loc['line2'].upper()}\n{loc['city'].upper()}, {loc['state']} {loc['zip']}", size=10)
    form.para("THIS CERTIFICATE IS NOT TRANSFERABLE ON CHANGE OF OWNERSHIP, CONTROL, LOCATION, OR BUSINESS ACTIVITY, AND IT IS NOT VALID AFTER THE EXPIRATION DATE.")
    return doc


def page_images(doc: fitz.Document, dpi: int) -> list[Image.Image]:
    return [
        Image.open(io.BytesIO(page.get_pixmap(dpi=dpi).tobytes("png"))).convert("RGB")
        for page in doc
    ]


def scanned(image: Image.Image, rng: random.Random, angle: float) -> Image.Image:
    gray = ImageOps.grayscale(image).rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)
    pixels = np.asarray(gray, dtype=np.float32)
    pixels += np.random.default_rng(rng.randrange(1 << 30)).normal(0, 9, pixels.shape)
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.5))


def faxed(image: Image.Image) -> Image.Image:
    # Standard-resolution fax: 204 x 98 dpi, bilevel, then stretched back to square pixels.
    width, height = image.size
    low = ImageOps.grayscale(image).resize((width * 204 // 300, height * 98 // 300), Image.BILINEAR)
    return low.point(lambda value: 255 if value > 150 else 0, "1").resize((width * 204 // 300, height * 196 // 300))


def phone_photo(image: Image.Image) -> Image.Image:
    width, height = image.size
    canvas = Image.new("RGB", (int(width * 1.15), int(height * 1.2)), (96, 84, 70))
    tilted = image.rotate(-3.5, resample=Image.BICUBIC, expand=True, fillcolor=(96, 84, 70))
    canvas.paste(tilted, ((canvas.width - tilted.width) // 2, (canvas.height - tilted.height) // 2))
    warm = ImageEnhance.Color(canvas).enhance(0.85)
    shade = np.linspace(1.0, 0.78, warm.width, dtype=np.float32)[None, :, None]
    return Image.fromarray(np.clip(np.asarray(warm, dtype=np.float32) * shade, 0, 255).astype(np.uint8))


def image_pdf(images: list[Image.Image], path: Path, dpi: int) -> None:
    images[0].save(path, "PDF", resolution=dpi, save_all=True, append_images=images[1:])


def truth() -> dict:
    return {
        "note": "Fictional data. group_enrollment_form carries a transposed group NPI on purpose.",
        "group": {key: value for key, value in GROUP.items()},
        "practitioner": PRACTITIONER,
        "coverage": COVERAGE,
        "conflicts": [{"field": "group_npi", "document": "02_group_enrollment_form.pdf", "value": TRANSPOSED_GROUP_NPI}],
        "documents": {
            "01_practitioner_application.pdf": "practitioner_application",
            "02_group_enrollment_form.pdf": "group_application",
            "03_group_w9.pdf": "w9",
            "04_az_np_license.jpg": "state_license",
            "05_malpractice_coi_fax.tif": "malpractice_certificate",
            "06_dea_certificate.pdf": "dea_certificate",
        },
    }


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261029)
    practitioner_application().save(out / "01_practitioner_application.pdf")
    image_pdf([scanned(image, rng, 0.8) for image in page_images(group_enrollment_form(), 200)], out / "02_group_enrollment_form.pdf", 200)
    image_pdf([scanned(image, rng, -0.5) for image in page_images(w9(), 200)], out / "03_group_w9.pdf", 200)
    phone_photo(page_images(license_certificate(), 170)[0]).save(out / "04_az_np_license.jpg", quality=82)
    fax_pages = [faxed(image) for image in page_images(malpractice_certificate(), 300)]
    fax_pages[0].save(out / "05_malpractice_coi_fax.tif", compression="group4", dpi=(204, 196), save_all=True, append_images=fax_pages[1:])
    image_pdf([scanned(image, rng, 0.3) for image in page_images(dea_certificate(), 200)], out / "06_dea_certificate.pdf", 200)
    (out / "truth.json").write_text(json.dumps(truth(), indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]))
