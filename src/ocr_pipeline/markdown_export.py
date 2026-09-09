"""Preserve unnamed trailing columns when exporting model-generated Markdown.

Uses markdown-it-py to distinguish tables from code and ordinary prose.
"""

from markdown_it import MarkdownIt
from markdown_it.rules_block.table import escapedSplit

MARKDOWN = MarkdownIt("commonmark", {"html": True}).enable("table")


def normalize_table_headers(text):
    # GFM truncates cells beyond the header width. Preserve those cells with
    # unnamed headers; never invent labels or change the original saved output.
    lines = text.splitlines(keepends=True)
    for token in MARKDOWN.parse(text):
        if token.type != "table_open" or token.level != 0:
            continue
        start, end = token.map
        widths = []
        for line in lines[start:end]:
            cells = escapedSplit(line.strip())
            if cells and cells[0] == "":
                cells.pop(0)
            if cells and cells[-1] == "":
                cells.pop()
            widths.append(len(cells))
        extra = max(widths) - widths[0]
        if extra:
            for index in (start, start + 1):
                line = lines[index].rstrip("\r\n").rstrip()
                if escapedSplit(line)[-1] != "":
                    line += " |"
                lines[index] = (
                    line + (" |" if index == start else " --- |") * extra + "\n"
                )
    return "".join(lines)
