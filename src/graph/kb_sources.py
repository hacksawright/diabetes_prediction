"""Configured knowledge sources and local raw-file cache."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import sleep
from urllib.parse import urlsplit

import requests

from ..config import ROOT

KB_DATA = ROOT / "data" / "kb"
MANIFEST = KB_DATA / "sources.json"
CACHE = ROOT / "artifacts" / "kb_sources"
USER_AGENT = "Diabet-PTIT-KB/1.0 (educational project; attributed text extraction)"


def configured_sources() -> list[dict]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def cache_path(source: dict) -> Path:
    if not re.fullmatch(r"[a-z0-9-]+", source["id"]):
        raise ValueError("Invalid source ID")
    return CACHE / (source["id"] + (".pdf" if source["format"] == "pdf" else ".html"))


def _fetch_group(sources: list[dict], refresh: bool) -> list[dict]:
    results = []
    with requests.Session() as session:
        session.headers["User-Agent"] = USER_AGENT
        for source in sources:
            path = cache_path(source)
            if path.is_file() and not refresh:
                results.append({"id": source["id"], "status": "cached"})
                continue
            temporary = path.with_suffix(".partial")
            try:
                with session.get(source["url"], stream=True, timeout=(12, 35)) as response:
                    response.raise_for_status()
                    content = bytearray()
                    for block in response.iter_content(64 * 1024):
                        content.extend(block)
                        if len(content) > 20_000_000:
                            raise ValueError("Source exceeds 20 MB")
                if (source["format"] == "pdf") != content.startswith(b"%PDF-"):
                    raise ValueError("Response format does not match the manifest")
                if not content:
                    raise ValueError("Empty source response")
                temporary.write_bytes(content)
                temporary.replace(path)
                results.append({"id": source["id"], "status": "downloaded", "bytes": len(content)})
            except (requests.RequestException, ValueError, OSError) as exc:
                results.append({"id": source["id"], "status": "error", "error": str(exc)})
            finally:
                temporary.unlink(missing_ok=True)
            sleep(0.5)
    return results


def fetch_sources(sources: list[dict], *, refresh: bool = False) -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    hosts = {}
    for source in sources:
        hosts.setdefault(urlsplit(source["url"]).hostname, []).append(source)
    with ThreadPoolExecutor(max_workers=4) as executor:
        return [
            item for result in executor.map(lambda group: _fetch_group(group, refresh), hosts.values())
            for item in result
        ]
