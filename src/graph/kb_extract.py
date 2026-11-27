"""Extract scoped passages, retaining table structure and PDF page numbers."""

import re
import unicodedata

from bs4 import BeautifulSoup

from .kb import normalize

TARGET = 1050
MAX_PASSAGE = 1650
IGNORE_HEADING = re.compile(r"^(?:references|sources|on this page|in this section|related (?:pages|topics)|more information|for everyone|clinical trials|contact us)\b", re.I)


def clean(value: str) -> str:
    value = unicodedata.normalize("NFC", value).replace("\u00ad", "")
    return " ".join(value.split())


def _parts(text: str) -> list[str]:
    if len(text) <= MAX_PASSAGE:
        return [text]
    return re.split(r"(?<=[.!?;])\s+(?=[A-ZÀ-Ỹ0-9])", text)


def _passages(blocks):
    section, page, pending = None, None, ""
    for title, number, content, table in blocks:
        if pending and ((title, number) != (section, page) or table or len(pending) + len(content) > TARGET):
            yield {"section": section, "page_start": page, "text": pending}
            pending = ""
        section, page = title, number
        if table:
            yield {"section": section, "page_start": page, "text": content}
            continue
        for part in _parts(content):
            if pending and len(pending) + len(part) > MAX_PASSAGE:
                yield {"section": section, "page_start": page, "text": pending}
                pending = ""
            pending = f"{pending}\n{part}" if pending else part
    if pending:
        yield {"section": section, "page_start": page, "text": pending}


def table_text(rows: list[list[str]], caption: str = "") -> str:
    lines = [" | ".join(clean(cell or "") for cell in row).strip(" |") for row in rows]
    return "\n".join(item for item in [caption, *lines] if item)


def extract_web(raw: bytes) -> list[dict]:
    soup = BeautifulSoup(raw, "html.parser")
    root = soup.select_one("article.dk-content, #content.cdc-page-content, article, main")
    if not root:
        raise ValueError("Article content container not found")
    for tag in root.select("script, style, nav, aside, footer, form, button, figure, .cdc-page-meta, .cdc-page-footer, .dk-related-content"):
        tag.decompose()
    section = ""
    blocks = []
    for tag in root.find_all(["h2", "h3", "h4", "p", "li", "table"]):
        if tag.name != "table" and tag.find_parent("table"):
            continue
        body = clean(tag.get_text(" ", strip=True))
        if tag.name.startswith("h"):
            section = body
            continue
        if IGNORE_HEADING.search(section):
            continue
        if tag.name == "table":
            caption = tag.find("caption")
            rows = [[cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"], recursive=False)]
                    for row in tag.find_all("tr")]
            notes = tag.find_next_sibling(["p", "div"])
            footnote = clean(notes.get_text(" ", strip=True)) if notes else ""
            content = table_text(rows, clean(caption.get_text(" ", strip=True)) if caption else "")
            if footnote.startswith(("*", "†", "‡", "Note:")) and len(footnote) < 650:
                content += "\n" + footnote
            if len(content) >= 40:
                blocks.append((section, None, content, True))
        elif len(body) >= 22:
            if tag.name == "p" and tag.find_parent("li"):
                continue
            if tag.name == "li" and (tag.find(["li", "p"]) or clean(tag.get_text(" ", strip=True)) == " ".join(
                clean(link.get_text(" ", strip=True)) for link in tag.find_all("a")
            )):
                continue
            if not re.match(r"(?:content source|last reviewed|this content is provided|the niddk would like to thank|view or print all)", body, re.I):
                blocks.append((section, None, body, False))
    return list(_passages(blocks))


def extract_pdf(path, segments: list[dict]) -> list[dict]:
    import pymupdf

    result = []
    with pymupdf.open(path) as document:
        for segment in segments:
            if segment["last"] > len(document):
                raise ValueError("Manifest page range exceeds the PDF length")
            for number in range(segment["first"], segment["last"] + 1):
                page = document[number - 1]
                section = segment["section"]
                # Retain detected table cells in reading order and exclude their
                # underlying text blocks to avoid indexing a flattened copy.
                tables = page.find_tables().tables if number in segment.get("tables", []) else []
                items = []
                for item in page.get_text("blocks", sort=True):
                    x0, y0, x1, y1, raw, _, kind = item
                    body = clean(raw)
                    if kind or len(body) < 20:
                        continue
                    box = pymupdf.Rect(x0, y0, x1, y1)
                    if any((box & pymupdf.Rect(table.bbox)).get_area() > box.get_area() * 0.5 for table in tables):
                        continue
                    items.append((y0, section, number, body, False))
                for table in tables:
                    body = table_text(table.extract())
                    if len(body) >= 40:
                        items.append((table.bbox[1], section, number, body, True))
                for chunk in _passages(item[1:] for item in sorted(items)):
                    chunk["audience"] = segment.get("audience")
                    chunk["purpose"] = segment.get("purpose")
                    result.append(chunk)
    return result


def distinct(passages: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for passage in passages:
        key = normalize(passage["text"])
        if len(key) >= 35 and key not in seen:
            result.append(passage)
            seen.add(key)
    return result
