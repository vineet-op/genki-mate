"""Clean and validate the Lesson 13 extracted Markdown.

This script changes formatting only. It does not correct Japanese, repair OCR,
or invent missing vocabulary.
"""

from html.parser import HTMLParser
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
INPUT_PATH = OUTPUT_DIR / "output.md"
CLEANED_PATH = OUTPUT_DIR / "cleaned_lesson13.md"
REVIEW_PATH = OUTPUT_DIR / "lesson13_review.md"

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
MOJIBAKE_PATTERN = re.compile(r"(?:Ã|Â|ã.|â.)")
INCOMPLETE_ARROW_PATTERN = re.compile(r"(?:→|->)\s*$|^\s*(?:→|->)")

# Lines that are pure noise in retrieved chunks:
# 1. Orphan furigana — entire line is a <sup>…</sup> tag (kana with no kanji context)
FURIGANA_LINE_PATTERN = re.compile(r"^\s*<sup>[^<]*</sup>\s*$")
# 2. GENKI page footers: "第13課 ◀ 27"  or  "26 ▶ 会話・文法編"
PAGE_FOOTER_PATTERN = re.compile(r"^\s*(?:第\d+課\s*[◀▶]\s*\d+|\d+\s*[◀▶]\s*\S+)\s*$")


class TableParser(HTMLParser):
    """Read table rows and cells without changing their text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.cell_colspans: list[list[int]] = []
        self.current_row: list[str] | None = None
        self.current_colspans: list[int] | None = None
        self.current_cell: list[str] | None = None
        self.current_colspan = 1
        self.bold_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.current_row = []
            self.current_colspans = []
        elif tag in {"td", "th"}:
            self.current_cell = []
            self.current_colspan = 1
            for name, value in attrs:
                if name == "colspan" and value and value.isdigit():
                    self.current_colspan = int(value)
        elif tag in {"b", "strong"}:
            self.bold_depth += 1

        if self.current_cell is not None and tag == "br":
            self.current_cell.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.current_cell is not None:
            text = "".join(self.current_cell).strip()
            assert self.current_row is not None
            assert self.current_colspans is not None
            self.current_row.append(text)
            self.current_colspans.append(self.current_colspan)
            self.current_cell = None
        elif tag == "tr" and self.current_row is not None:
            self.rows.append(self.current_row)
            assert self.current_colspans is not None
            self.cell_colspans.append(self.current_colspans)
            self.current_row = None
            self.current_colspans = None
        elif tag in {"b", "strong"}:
            self.bold_depth = max(0, self.bold_depth - 1)

    def handle_data(self, data: str) -> None:
        if self.current_cell is not None:
            if self.bold_depth:
                self.current_cell.append(f"**{data}**")
            else:
                self.current_cell.append(data)


def table_to_markdown(table_html: str, source_line: int, review: list[str]) -> str:
    """Convert one HTML table and record structural uncertainty."""
    parser = TableParser()
    parser.feed(table_html)
    if not parser.rows:
        review.append(f"- Line {source_line}: table has no recoverable rows.")
        return table_html

    column_count = max(sum(spans) for spans in parser.cell_colspans)
    expanded_rows: list[list[str]] = []

    for row_index, (row, spans) in enumerate(zip(parser.rows, parser.cell_colspans), 1):
        expanded: list[str] = []
        for value, colspan in zip(row, spans):
            expanded.append(value)
            expanded.extend([""] * (colspan - 1))

        if len(expanded) < column_count:
            review.append(
                f"- Line {source_line}: table row {row_index} has {len(expanded)} cells; "
                f"padded to {column_count}. Missing cells were preserved as empty."
            )
            expanded.extend([""] * (column_count - len(expanded)))
        elif len(expanded) > column_count:
            review.append(
                f"- Line {source_line}: table row {row_index} has more cells than the first rows; "
                "structure may be corrupted."
            )
        expanded_rows.append(expanded[:column_count])

        if any(not cell.strip() for cell in expanded):
            review.append(
                f"- Line {source_line}: table row {row_index} contains an empty cell; "
                "left unchanged for manual verification."
            )

    def markdown_row(row: list[str]) -> str:
        escaped = [cell.replace("|", r"\|") for cell in row]
        return "| " + " | ".join(escaped) + " |"

    separator = ["---"] * column_count
    result = [markdown_row(expanded_rows[0]), markdown_row(separator)]
    result.extend(markdown_row(row) for row in expanded_rows[1:])
    return "\n".join(result)


def clean_markdown(markdown: str, review: list[str]) -> str:
    """Apply formatting-only cleanup and convert recoverable HTML tables."""
    table_pattern = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)

    def replace_table(match: re.Match[str]) -> str:
        line_number = markdown.count("\n", 0, match.start()) + 1
        return table_to_markdown(match.group(0), line_number, review)

    cleaned = table_pattern.sub(replace_table, markdown)
    cleaned_lines: list[str] = []
    previous_blank = False
    removed_furigana = 0
    removed_footers = 0

    for line in cleaned.splitlines():
        # Skip brand watermark
        if line.strip() == "JapanWithAdi":
            continue
        # Strip email addresses
        line = EMAIL_PATTERN.sub("", line)
        # Fix 1: drop orphan furigana lines (pure <sup>…</sup>, no kanji context)
        if FURIGANA_LINE_PATTERN.match(line):
            removed_furigana += 1
            continue
        # Fix 2: drop GENKI page footers ("第13課 ◀ 27", "26 ▶ 会話・文法編")
        if PAGE_FOOTER_PATTERN.match(line):
            removed_footers += 1
            continue
        # Collapse consecutive blank lines into one
        if not line.strip():
            if previous_blank:
                continue
            previous_blank = True
        else:
            previous_blank = False
        cleaned_lines.append(line.rstrip())

    if removed_furigana:
        review.append(f"- Removed {removed_furigana} orphan furigana line(s) (<sup>…</sup> with no kanji on the same line).")
    if removed_footers:
        review.append(f"- Removed {removed_footers} GENKI page footer line(s) (e.g. 第13課 ◀ 27).")

    return "\n".join(cleaned_lines).strip() + "\n"


def collect_text_flags(markdown: str, review: list[str]) -> None:
    """Report suspicious text without changing it."""
    for line_number, line in enumerate(markdown.splitlines(), 1):
        if MOJIBAKE_PATTERN.search(line):
            review.append(
                f"- Line {line_number}: possible mixed-script or encoding corruption; text was preserved."
            )
        if INCOMPLETE_ARROW_PATTERN.search(line):
            review.append(
                f"- Line {line_number}: possible incomplete conjugation/example around an arrow; preserved."
            )

    review.append(
        "- Furigana and pronunciation lines were not automatically paired or corrected. "
        "Review short Japanese-only lines manually."
    )


def write_review(review: list[str]) -> None:
    """Write the review findings separately from the cleaned textbook text."""
    lines = [
        "# Lesson 13 Manual Review",
        "",
        "This report flags uncertainty found during formatting cleanup.",
        "No Japanese text, OCR spelling, or missing vocabulary was invented or corrected.",
        "",
        "## Findings",
        "",
    ]
    lines.extend(review or ["- No suspicious patterns were detected."])
    REVIEW_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    original = INPUT_PATH.read_text(encoding="utf-8")
    review: list[str] = []
    cleaned = clean_markdown(original, review)
    collect_text_flags(original, review)
    CLEANED_PATH.write_text(cleaned, encoding="utf-8")
    write_review(review)
    print(f"Wrote {CLEANED_PATH}")
    print(f"Wrote {REVIEW_PATH}")


if __name__ == "__main__":
    main()
