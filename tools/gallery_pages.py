"""Build fictional pages exercising arrangement kinds the synthetic packet does not cover.

Writes one PDF/image per page kind, plus a few degraded variants (rotated, a phone
photo, a bilevel fax), for experiments/arrangement_gallery.py to exercise. Reuses the
page-building style (PyMuPDF `Form` helper, handwritten font, fax degradation) from
tools/synthetic_enrollment_packet.py. All text is fictional.

    python tools/gallery_pages.py OUT_DIR
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import fitz
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_enrollment_packet import HAND_FONT, INK, MARGIN, PAGE, Form, faxed, page_images  # noqa: E402

ABSTRACT = (
    "We survey understory plant diversity across twelve fictional canyon woodland plots "
    "following a decade of mesquite encroachment. Shade-tolerant forbs declined where canopy "
    "cover exceeded 70 percent, while fern cover increased near perennial seeps. These results "
    "are illustrative only and do not describe a real study site."
)
LEFT_COLUMN = (
    "Canyon woodlands in the fictional Verdeluna range form a mosaic of closed-canopy mesquite "
    "stands and open rocky slopes. Prior fictional surveys suggested that understory richness "
    "tracks canopy openness, but no plot-level data had been collected before this study.\n\n"
    "We established twelve 20-by-20 meter plots along three drainages, stratified by canopy "
    "cover class (open, intermediate, closed). Within each plot we recorded every vascular "
    "plant rooted below one meter in height, along with canopy cover estimated from a "
    "convex densiometer at plot center."
)
RIGHT_COLUMN = (
    "Across the twelve plots, mean understory richness fell from 14.2 species in open plots to "
    "6.8 species in closed plots. Fern cover was highest within five meters of a seep regardless "
    "of canopy class, suggesting soil moisture outweighs light availability for this guild.\n\n"
    "These fictional results are consistent with a light-driven decline in forbs offset by a "
    "moisture-driven refuge for ferns, and motivate a larger, multi-year fictional follow-up "
    "across the full canyon network."
)


def two_column_article() -> fitz.Document:
    doc = fitz.open()
    page = doc.new_page(width=PAGE.width, height=PAGE.height)
    title = "Diffusion of Shade-Tolerant Understory Flora Across Fragmented Canyon Woodlands"
    page.insert_textbox(
        fitz.Rect(MARGIN, MARGIN, PAGE.width - MARGIN, MARGIN + 56), title, fontname="hebo", fontsize=15, align=1
    )
    y = MARGIN + 64
    page.insert_text((MARGIN, y), "T. Alvarez, K. Nduka and R. Soltani -- fictional sample", fontname="tiro", fontsize=9)
    y += 20
    page.insert_textbox(
        fitz.Rect(MARGIN, y, PAGE.width - MARGIN, y + 60), f"Abstract. {ABSTRACT}", fontname="tiro", fontsize=8.5
    )
    y += 72
    gutter = 16
    column_width = (PAGE.width - 2 * MARGIN - gutter) / 2
    column_bottom = PAGE.height - 215
    left = fitz.Rect(MARGIN, y, MARGIN + column_width, column_bottom)
    right = fitz.Rect(MARGIN + column_width + gutter, y, PAGE.width - MARGIN, column_bottom)
    page.insert_textbox(left, LEFT_COLUMN, fontname="tiro", fontsize=9)
    page.insert_textbox(right, RIGHT_COLUMN, fontname="tiro", fontsize=9)
    figure = fitz.Rect(MARGIN, PAGE.height - 205, PAGE.width - MARGIN, PAGE.height - 140)
    page.draw_rect(figure, color=(0.4, 0.4, 0.4), width=0.8)
    axis_x, axis_y = figure.x0 + 20, figure.y1 - 10
    page.draw_line((axis_x, figure.y0 + 10), (axis_x, axis_y))
    page.draw_line((axis_x, axis_y), (figure.x1 - 20, axis_y))
    rng = random.Random(7)
    points = sorted((axis_x + i * 18, axis_y - rng.randint(5, 55)) for i in range(12))
    for point_a, point_b in zip(points, points[1:]):
        page.draw_line(point_a, point_b, color=(0.2, 0.3, 0.55))
    page.insert_textbox(
        fitz.Rect(MARGIN, PAGE.height - 135, PAGE.width - MARGIN, PAGE.height - 115),
        "Figure 1. Canopy cover vs. understory species richness across twelve fictional survey plots.",
        fontname="tiro",
        fontsize=7.5,
        align=1,
    )
    page.insert_textbox(fitz.Rect(MARGIN, PAGE.height - 40, PAGE.width - MARGIN, PAGE.height - 20), "1", fontname="helv", fontsize=9, align=1)
    return doc


def _claims_rows(start: int, count: int) -> list[list[str]]:
    rng = random.Random(1000 + start)
    rows = []
    for i in range(start, start + count):
        billed = rng.choice([120, 145, 210, 88, 305, 460])
        allowed = round(billed * rng.choice([0.6, 0.65, 0.72]))
        paid = round(allowed * rng.choice([0.8, 0.9, 1.0]))
        rows.append(
            [
                f"CLM-{2026000 + i}",
                f"Member {chr(65 + i % 26)}. Doe",
                f"10/{(i % 28) + 1:02d}/2026",
                rng.choice(["99213", "99214", "90834", "97110", "71045"]),
                f"${billed:.2f}",
                f"${allowed:.2f}",
                f"${paid:.2f}",
            ]
        )
    return rows


def multi_page_table() -> fitz.Document:
    doc = fitz.open()
    form = Form(doc)
    headers = ["Claim ID", "Member", "DOS", "CPT", "Billed", "Allowed", "Paid"]
    shares = [0.17, 0.22, 0.13, 0.11, 0.13, 0.12, 0.12]
    form.title("Claims Register -- Page 1 of 2", "Desert Sky Health Plan -- fictional sample")
    form.table(headers, _claims_rows(1, 14), shares)
    form.new_page()
    form.title("Claims Register -- Page 2 of 2 (continued)", "Desert Sky Health Plan -- fictional sample")
    form.table(headers, _claims_rows(15, 14), shares)
    return doc


def financial_table_dense() -> fitz.Document:
    doc = fitz.open()
    form = Form(doc)
    form.title(
        "Condensed Interim Financial Statement",
        "Saguaro Valley Holdings LLC -- fictional sample, amounts in USD thousands",
    )
    page = form.page
    width = PAGE.width - 2 * MARGIN
    shares = [0.28, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]
    label_w, q1_w, q2_w = width * shares[0], width * sum(shares[1:4]), width * sum(shares[4:7])
    x0, y, row_h = MARGIN, form.y, 16
    page.draw_rect(fitz.Rect(x0, y, x0 + label_w, y + row_h), color=(0.35, 0.35, 0.35), fill=(0.93, 0.94, 0.96), width=0.6)
    for offset, width_, label in ((label_w, q1_w, "Q1 2026"), (label_w + q1_w, q2_w, "Q2 2026")):
        box = fitz.Rect(x0 + offset, y, x0 + offset + width_, y + row_h)
        page.draw_rect(box, color=(0.35, 0.35, 0.35), fill=(0.85, 0.88, 0.93), width=0.6)
        page.insert_text((box.x0 + 4, box.y0 + 11), label, fontname="hebo", fontsize=8)
    y += row_h
    x = x0
    for month, share in zip(["(in thousands)", "Jan", "Feb", "Mar", "Apr", "May", "Jun"], shares):
        w = width * share
        box = fitz.Rect(x, y, x + w, y + row_h)
        page.draw_rect(box, color=(0.35, 0.35, 0.35), fill=(0.93, 0.94, 0.96), width=0.6)
        page.insert_text((box.x0 + 3, box.y0 + 11), month, fontname="hebo", fontsize=7)
        x += w
    y += row_h
    rows = [
        ("Revenue", ["412", "398", "455", "470", "481", "502"]),
        ("Cost of services", ["(201)", "(196)", "(214)", "(221)", "(228)", "(235)"]),
        ("Gross margin", ["211", "202", "241", "249", "253", "267"]),
        ("Operating expenses", ["(140)", "(138)", "(145)", "(149)", "(151)", "(156)"]),
        ("EBITDA", ["71", "64", "96", "100", "102", "111"]),
        ("Net income", ["52", "47", "71", "74", "76", "83"]),
    ]
    for label, values in rows:
        x = x0
        for cell, share in zip([label, *values], shares):
            w = width * share
            box = fitz.Rect(x, y, x + w, y + row_h)
            page.draw_rect(box, color=(0.35, 0.35, 0.35), width=0.6)
            page.insert_text((box.x0 + 3, box.y0 + 11), cell, fontname="helv", fontsize=7.5)
            x += w
        y += row_h
    page.insert_textbox(
        fitz.Rect(MARGIN, y + 12, PAGE.width - MARGIN, y + 60),
        "Unaudited. Prepared for internal review only; figures are illustrative and do not represent an actual entity.",
        fontname="tiro",
        fontsize=8,
    )
    return doc


RECEIPT_WIDTH = 220
RECEIPT_ITEMS = [
    ("Exam Room Supplies Kit", "1", "18.50"),
    ("Nitrile Gloves (box)", "3", "7.25"),
    ("Alcohol Prep Pads (100ct)", "2", "3.10"),
    ("Thermometer Covers (50ct)", "4", "2.40"),
    ("Specimen Bags", "2", "1.85"),
]


def receipt() -> fitz.Document:
    subtotal = sum(float(quantity) * float(price) for _, quantity, price in RECEIPT_ITEMS)
    tax = subtotal * 0.081
    total = subtotal + tax
    height = 100 + len(RECEIPT_ITEMS) * 22 + 90
    doc = fitz.open()
    page = doc.new_page(width=RECEIPT_WIDTH, height=height)
    y = 14

    def centered(text: str, size: float = 9, font: str = "helv") -> None:
        nonlocal y
        text_width = fitz.get_text_length(text, font, size)
        page.insert_text(((RECEIPT_WIDTH - text_width) / 2, y), text, fontname=font, fontsize=size)
        y += size + 4

    centered("Desert Sky Medical Supply", 10, "hebo")
    centered("4420 W Cactus Blossom Dr, Mesa AZ", 6.5)
    centered("fictional sample receipt", 6)
    y += 6
    page.draw_line((8, y), (RECEIPT_WIDTH - 8, y), width=0.5)
    y += 10
    page.insert_text((8, y), "Order #SK-20261029-0447", fontname="cour", fontsize=7)
    y += 12
    page.insert_text((8, y), "10/29/2026  09:41 AM", fontname="cour", fontsize=7)
    y += 14
    for name, quantity, price in RECEIPT_ITEMS:
        page.insert_text((8, y), name, fontname="helv", fontsize=7)
        y += 10
        line_total = f"{float(quantity) * float(price):.2f}"
        page.insert_text((8, y), f"  {quantity} x {price}", fontname="cour", fontsize=7)
        total_width = fitz.get_text_length(line_total, "cour", 7)
        page.insert_text((RECEIPT_WIDTH - 8 - total_width, y), line_total, fontname="cour", fontsize=7)
        y += 12
    page.draw_line((8, y), (RECEIPT_WIDTH - 8, y), width=0.5)
    y += 10
    for label, value, bold in [("Subtotal", f"{subtotal:.2f}", False), ("Tax (8.1%)", f"{tax:.2f}", False), ("Total", f"{total:.2f}", True)]:
        font, size = ("hebo", 8) if bold else ("cour", 7)
        page.insert_text((8, y), label, fontname=font, fontsize=size)
        value_width = fitz.get_text_length(value, font, size)
        page.insert_text((RECEIPT_WIDTH - 8 - value_width, y), value, fontname=font, fontsize=size)
        y += 13
    y += 6
    centered("Thank you for your order", 7)
    return doc


def letterhead_letter() -> fitz.Document:
    doc = fitz.open()
    page = doc.new_page(width=PAGE.width, height=PAGE.height)
    page.insert_text((MARGIN, 60), "Desert Sky Health Plan", fontname="hebo", fontsize=16)
    page.insert_text(
        (MARGIN, 76),
        "Provider Network Management -- 100 N Central Ave, Phoenix, AZ 85004 -- fictional sample",
        fontname="helv",
        fontsize=8,
    )
    page.draw_line((MARGIN, 86), (PAGE.width - MARGIN, 86), width=0.8)
    y = 120
    page.insert_text((MARGIN, y), "October 29, 2026", fontname="helv", fontsize=10)
    y += 30
    for line in ["Maya R. Okafor, NP", "Saguaro Valley Family Medicine PLLC", "4420 W Cactus Blossom Dr, Ste 210", "Mesa, AZ 85201"]:
        page.insert_text((MARGIN, y), line, fontname="helv", fontsize=10)
        y += 14
    y += 16
    page.insert_text((MARGIN, y), "Dear Ms. Okafor,", fontname="helv", fontsize=10)
    y += 24
    body = (
        "This letter confirms that your credentialing file for Desert Sky Health Plan has been reviewed "
        "and is complete pending verification of your current malpractice coverage. Please submit an "
        "updated certificate of insurance within 30 days of this notice. Once received, your effective "
        "network date will be confirmed in writing.\n\nIf you have questions about this request, contact "
        "the credentialing department at the number below."
    )
    spare = page.insert_textbox(fitz.Rect(MARGIN, y, PAGE.width - MARGIN, y + 160), body, fontname="helv", fontsize=10)
    y += 160 - spare + 20
    page.insert_text((MARGIN, y), "Sincerely,", fontname="helv", fontsize=10)
    y += 36
    page.insert_text((MARGIN, y), "Amara Whitfield", fontname="hand", fontfile=HAND_FONT, fontsize=18, color=INK)
    y += 16
    page.insert_text((MARGIN, y), "Amara Whitfield, Credentialing Manager", fontname="helv", fontsize=9)
    y += 12
    page.insert_text((MARGIN, y), "(602) 555-0133 -- credentialing@example.org", fontname="helv", fontsize=9)
    return doc


def checkbox_grid() -> fitz.Document:
    doc = fitz.open()
    page = doc.new_page(width=PAGE.width, height=PAGE.height)
    page.insert_text((MARGIN, MARGIN + 10), "Practice Capability Checklist", fontname="hebo", fontsize=14)
    page.insert_text((MARGIN, MARGIN + 26), "Desert Sky Health Plan -- fictional sample", fontname="helv", fontsize=8)
    rows = [
        "On-site laboratory draw station",
        "Point-of-care rapid testing",
        "In-house imaging (X-ray)",
        "Electronic health records (EHR)",
        "Patient portal access",
        "After-hours triage line",
        "Wheelchair-accessible entrance",
        "Interpreter services available",
        "Telehealth video visits",
        "On-site pharmacy dispensing",
    ]
    ticked = {0, 1, 3, 4, 6, 8}
    yes_x, no_x = PAGE.width - MARGIN - 70, PAGE.width - MARGIN - 25
    y = MARGIN + 50
    page.insert_text((MARGIN, y), "Capability", fontname="hebo", fontsize=8.5)
    page.insert_text((yes_x, y), "Yes", fontname="hebo", fontsize=8.5)
    page.insert_text((no_x, y), "No", fontname="hebo", fontsize=8.5)
    y += 14
    for index, label in enumerate(rows):
        page.insert_text((MARGIN, y + 8), label, fontname="helv", fontsize=9)
        for x, is_yes in ((yes_x, True), (no_x, False)):
            box = fitz.Rect(x, y, x + 10, y + 10)
            page.draw_rect(box, color=(0, 0, 0), width=0.7)
            if (index in ticked) == is_yes:
                page.draw_line(box.tl + (1.5, 5), box.bl + (3.5, -1.5), color=(0, 0, 0), width=1.3)
                page.draw_line(box.bl + (3.5, -1.5), box.tr + (1, -1), color=(0, 0, 0), width=1.3)
        y += 22
    return doc


def form_label_styles() -> fitz.Document:
    doc = fitz.open()
    form = Form(doc)
    form.title(
        "Patient Intake -- Label Placement Sample",
        "fictional sample -- exercises label-inside, label-above and label-beside layouts",
    )
    form.section("A. Labels beside values (inline)")
    for label, value in [("Name: ", "Jordan T. Vasquez"), ("Date of birth: ", "03/22/1991"), ("Phone: ", "(480) 555-0199")]:
        form.page.insert_text((MARGIN, form.y + 10), f"{label}{value}", fontname="helv", fontsize=10)
        form.y += 18
    form.y += 10
    form.section("B. Labels above values")
    for label, value in [
        ("Reason for visit", "Annual wellness exam"),
        ("Referring provider", "Dr. S. Mehta"),
        ("Pharmacy", "Saguaro Valley Pharmacy, Mesa AZ"),
    ]:
        form.page.insert_text((MARGIN, form.y + 8), label, fontname="helv", fontsize=7.5)
        form.page.insert_text((MARGIN, form.y + 22), value, fontname="helv", fontsize=10)
        form.y += 32
    form.y += 6
    form.section("C. Labels inside boxes")
    form.row([("Insurance carrier", "Copper State Medical Mutual", 0.5), ("Policy number", "CSM-2219047", 0.5)])
    form.row([("Emergency contact", "Elena Vasquez", 0.5), ("Relationship", "Spouse", 0.5)])
    return doc


def certificate() -> fitz.Document:
    doc = fitz.open()
    page = doc.new_page(width=PAGE.height, height=PAGE.width)  # landscape wall certificate
    frame = page.rect + (30, 30, -30, -30)
    page.draw_rect(frame, color=(0.45, 0.32, 0.1), width=4)
    page.draw_rect(frame + (8, 8, -8, -8), color=(0.45, 0.32, 0.1), width=1)
    lines = [
        ("DESERT SKY HEALTH PLAN -- fictional sample", "hebo", 18, 90),
        ("Certificate of Network Participation", "tiro", 20, 130),
        ("This certifies that", "tiro", 12, 175),
        ("SAGUARO VALLEY FAMILY MEDICINE PLLC", "hebo", 22, 212),
        ("is a participating provider in good standing in the Desert Sky network", "tiro", 12, 248),
        ("Effective 01/01/2026          Valid through 12/31/2026", "helv", 12, 300),
    ]
    for text, font, size, y in lines:
        text_width = fitz.get_text_length(text, font, size)
        page.insert_text(((page.rect.width - text_width) / 2, y), text, fontname=font, fontsize=size)
    page.draw_circle((page.rect.width - 140, page.rect.height - 110), 42, color=(0.6, 0.45, 0.1), width=2)
    page.insert_text((page.rect.width - 168, page.rect.height - 106), "SEAL", fontname="hebo", fontsize=14, color=(0.6, 0.45, 0.1))
    return doc


def _perspective_shadow(image: Image.Image) -> Image.Image:
    """A handheld photo: top edge pulled in (camera tilt) plus a soft directional shadow."""
    width, height = image.size
    skew = int(width * 0.05)
    quad = (skew, 0, 0, height, width, height, width - skew, int(height * 0.04))
    warped = image.transform((width, height), Image.QUAD, quad, resample=Image.BICUBIC, fillcolor=(90, 82, 70))
    shade = np.linspace(0.72, 1.0, width, dtype=np.float32)[None, :, None]
    shaded = np.clip(np.asarray(warped, dtype=np.float32) * shade, 0, 255).astype(np.uint8)
    return Image.fromarray(shaded).convert("RGB")


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    documents = {
        "two_column_article": two_column_article(),
        "multi_page_table": multi_page_table(),
        "financial_table_dense": financial_table_dense(),
        "receipt": receipt(),
        "letterhead_letter": letterhead_letter(),
        "checkbox_grid": checkbox_grid(),
        "form_label_styles": form_label_styles(),
        "certificate": certificate(),
    }
    for name, doc in documents.items():
        doc.save(out / f"{name}.pdf")

    page_images(documents["two_column_article"], 200)[0].rotate(90, expand=True).save(out / "two_column_article_rotated90.png")
    page_images(documents["form_label_styles"], 200)[0].rotate(180).save(out / "form_label_styles_rotated180.png")
    _perspective_shadow(page_images(documents["receipt"], 250)[0]).save(out / "receipt_phone_photo.jpg", quality=80)

    fax_pages = [faxed(image) for image in page_images(documents["letterhead_letter"], 300)]
    fax_pages[0].save(
        out / "letterhead_letter_fax.tif", compression="group4", dpi=(204, 196), save_all=True, append_images=fax_pages[1:]
    )
    print(f"Wrote {len(documents) + 4} pages to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]))
