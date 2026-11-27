"""Fetch and ingest the curated Diabet corpus without recreating the database."""

import argparse
import json
import re
from datetime import datetime, timezone
from hashlib import sha256

from neo4j import GraphDatabase

from ..config import neo4j_config
from .kb import normalize
from .kb_extract import distinct, extract_pdf, extract_web
from .kb_sources import CACHE, KB_DATA, cache_path, configured_sources, fetch_sources

PROJECT_REFERENCE = KB_DATA / "brfss_reference.md"
PROJECT_URL = "/kb/brfss"
PROJECT_SOURCE = {
    "id": "diabet-brfss", "publisher": "Diabet PTIT", "title": "Mô hình BRFSS trong Diabet",
    "url": PROJECT_URL, "format": "project", "language": "vi",
}


def _extract(source: dict) -> list[dict]:
    if source["format"] == "project":
        groups = re.split(r"^## ", PROJECT_REFERENCE.read_text(encoding="utf-8"), flags=re.M)[1:]
        passages = [{"section": group.split("\n", 1)[0], "text": group.split("\n", 1)[1].strip(),
                     "page_start": None} for group in groups]
    else:
        path = cache_path(source)
        if source["format"] == "pdf":
            if source.get("reviewed_notes"):
                notes = json.loads((KB_DATA / source["reviewed_notes"]).read_text(encoding="utf-8"))
                if sha256(path.read_bytes()).hexdigest() != notes["pdf_sha256"]:
                    raise ValueError("PDF changed since the page notes were reviewed")
                passages = notes["passages"]
            else:
                passages = extract_pdf(path, source["segments"])
        else:
            passages = extract_web(path.read_bytes())
    rows = []
    for position, passage in enumerate(distinct(passages)):
        content = passage["text"]
        audience = passage.get("audience") or source.get("audience")
        purpose = passage.get("purpose") or source.get("purpose")
        fingerprint = sha256(json.dumps([content, passage.get("section"), passage["page_start"], audience, purpose],
                                        ensure_ascii=False).encode()).hexdigest()
        rows.append({
            **passage, "audience": audience, "purpose": purpose, "position": position,
            "id": f"{source['url']}#{position}", "fingerprint": fingerprint, "normalized": normalize(content),
        })
    if not rows:
        raise ValueError("No usable passages extracted")
    return rows


def _upsert(transaction, source: dict, rows: list[dict]):
    # Failed fetches or parsing never reach this transaction.
    transaction.run("""
        MERGE (a:Article {url: $source.url})
        SET a.source_id = $source.id, a.title = $source.title, a.publisher = $source.publisher,
            a.format = $source.format, a.language = $source.language,
            a.extraction = $source.extraction,
            a.published_at = $source.published_at, a.fetched_at = $source.fetched_at
        WITH a
        UNWIND $rows AS row
        MERGE (c:Chunk {id: row.id})
        MERGE (a)-[:HAS_CHUNK]->(c)
        WITH c, row, coalesce(c.fingerprint, '') <> row.fingerprint AS changed
        SET c.text = row.text, c.normalized = row.normalized, c.position = row.position,
            c.section = row.section, c.page_start = row.page_start, c.audience = row.audience, c.purpose = row.purpose,
            c.fingerprint = row.fingerprint
        FOREACH (_ IN CASE WHEN changed THEN [1] ELSE [] END |
            SET c.rag_enabled = false, c.rag_embedding = null, c.rag_search_text = null
        )
    """, source=source, rows=rows).consume()
    transaction.run("""
        MATCH (:Article {url: $url})-[:HAS_CHUNK]->(c:Chunk)
        WHERE c.fingerprint IS NOT NULL AND c.position >= $active_count
        DETACH DELETE c
    """, url=source["url"], active_count=len(rows)).consume()


def ingest_cached(sources: list[dict], *, dry_run: bool = False) -> list[dict]:
    extracted, results = [], []
    for source in [*sources, PROJECT_SOURCE]:
        try:
            rows = _extract(source)
        except (ValueError, OSError) as exc:
            results.append({"id": source["id"], "status": "error", "error": str(exc)})
            continue
        path = PROJECT_REFERENCE if source["format"] == "project" else cache_path(source)
        metadata = {
            **source, "published_at": source.get("published_at"),
            "extraction": "reviewed_scan_notes" if source.get("reviewed_notes") else "parsed_text",
            "fetched_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        }
        extracted.append((metadata, rows))
        results.append({
            "id": source["id"], "status": "preview" if dry_run else "ok",
            "type": source["format"], "url": source["url"], "chunks": len(rows),
            "pages": sorted({row["page_start"] for row in rows if row["page_start"] is not None}),
        })
    if extracted and not dry_run:
        cfg = neo4j_config()
        with GraphDatabase.driver(cfg["uri"], auth=cfg["auth"]) as driver:
            with driver.session(database=cfg["database"]) as session:
                session.run("CREATE CONSTRAINT rag_article_url IF NOT EXISTS FOR (a:Article) REQUIRE a.url IS UNIQUE").consume()
                session.run("CREATE CONSTRAINT rag_chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE").consume()
                for metadata, rows in extracted:
                    session.execute_write(_upsert, metadata, rows)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Refresh raw files instead of reusing cache")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="Process only files already in the cache")
    mode.add_argument("--fetch-only", action="store_true", help="Download source files without touching Neo4j")
    parser.add_argument("--dry-run", action="store_true", help="Preview passages without writing Neo4j")
    args = parser.parse_args()
    sources = configured_sources()
    fetched = [] if args.offline else fetch_sources(sources, refresh=args.refresh)
    ingested = [] if args.fetch_only else ingest_cached(sources, dry_run=args.dry_run)
    report = {"fetch": fetched, "ingest": ingested}
    if not args.dry_run:
        CACHE.mkdir(parents=True, exist_ok=True)
        (CACHE / "last_ingest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if any(item["status"] == "error" for item in [*fetched, *ingested]):
        raise SystemExit(1)
