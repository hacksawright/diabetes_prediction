"""Build the retrieval indexes from articles already stored in Neo4j."""

import argparse
import json
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests
from neo4j import GraphDatabase

from ..config import neo4j_config
from .kb import normalize
from .rag_models import EMBEDDING_DIM, EMBEDDING_VERSION, MODEL_DIR, MODEL_FILES, available, embed
from .rag_topics import TOPICS, audiences, match_topics


def download_models():
    for name, assets in MODEL_FILES.items():
        directory = MODEL_DIR / name
        directory.mkdir(parents=True, exist_ok=True)
        for filename, url in assets.items():
            path = directory / filename
            if path.is_file():
                continue
            with NamedTemporaryFile(dir=directory, delete=False) as tmp:
                temporary = Path(tmp.name)
            try:
                with requests.get(url, stream=True, timeout=(15, 90)) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as output:
                        for block in response.iter_content(chunk_size=1024 * 1024):
                            output.write(block)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)


def _metadata(rows):
    current_source = None
    prediabetes_section = False
    for index, row in enumerate(rows):
        original = row["text"]
        value = normalize(original)
        source = row["url"].rstrip("/").split("/")[-1]
        if row["url"] != current_source:
            current_source = row["url"]
            prediabetes_section = False
        if source == "xet-nghiem-chan-doan-dai-thao-duong-vi" and re.match(r"chan doan tien dai thao duong\b", value):
            prediabetes_section = True
        boilerplate = bool(re.search(
            r"(^bai viet duoc (?:viet|tham van)|dat lich kham|hotline|dang trien khai goi|cung cap goi|day thi som)",
            value,
        ))
        # The general type-2 overview has abbreviated diagnostic bullets with
        # inconsistent boundary notation. Use the dedicated test articles for numbers.
        abbreviated_cutoff = source == "tieu-duong-tuyp-2-3042" and bool(
            re.match(r"(?:duong huyet (?:bat ki|luc doi|2 gio)|hba1c)\b", value) and re.search(r"\d", value)
        )
        passage = original
        if index and re.match(r"(?:neu ket qua a1c|muc chi so duong huyet)\b", value):
            previous = rows[index - 1]
            if previous["url"] == row["url"] and previous["position"] + 1 == row["position"]:
                passage = previous["text"] + "\n" + original
        section = row.get("section") or ""
        retrieval_text = f"{section}\n{passage}".strip()
        topics = match_topics(retrieval_text)
        if (prediabetes_section or source.startswith("dau-hieu-va-tieu-chuan-chan-doan-tien-tieu-duong")) and "prediabetes" not in topics:
            topics.append("prediabetes")
        aliases = " ".join(TOPICS[key][0] for key in topics)
        declared_audience = row.get("audience")
        if isinstance(declared_audience, list):
            scopes = declared_audience
        elif declared_audience:
            scopes = [declared_audience]
        else:
            scopes = audiences(section or passage)
        yield {
            "id": row["id"], "enabled": row.get("format") != "project" and not (boilerplate or abbreviated_cutoff),
            "passage": passage, "search_text": normalize(f"{retrieval_text} {row.get('title') or ''} {aliases}"),
            "topics": topics, "audiences": scopes,
            "embedding_text": retrieval_text,
        }


def build_indexes() -> dict:
    if not available("e5") or not available("reranker"):
        raise RuntimeError("Thiếu tệp model RAG. Chạy lại với --download-models.")
    cfg = neo4j_config()
    with GraphDatabase.driver(cfg["uri"], auth=cfg["auth"]) as driver:
        with driver.session(database=cfg["database"]) as session:
            rows = session.run("""
                MATCH (a:Article)-[:HAS_CHUNK]->(c:Chunk)
                RETURN a.url AS url, c.id AS id, c.position AS position, c.text AS text,
                       a.title AS title, a.format AS format, c.section AS section, c.audience AS audience
                ORDER BY url, position
            """).data()
            if not rows:
                raise RuntimeError("Neo4j chưa có đoạn KB để lập chỉ mục.")
            chunks = list(_metadata(rows))
            searchable = [chunk for chunk in chunks if chunk["enabled"]]
            vectors = iter(embed([chunk["embedding_text"] for chunk in searchable]))
            for chunk in chunks:
                chunk["embedding"] = next(vectors) if chunk["enabled"] else None
                del chunk["embedding_text"]
            session.run("CREATE CONSTRAINT rag_chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE").consume()
            session.run("CREATE CONSTRAINT concept_key IF NOT EXISTS FOR (c:Concept) REQUIRE c.key IS UNIQUE").consume()
            session.run("""
                UNWIND $chunks AS row
                MATCH (c:Chunk {id: row.id})
                SET c.rag_search_text = CASE WHEN row.enabled THEN row.search_text ELSE null END,
                    c.rag_passage = row.passage, c.rag_topics = row.topics,
                    c.rag_audiences = row.audiences, c.rag_enabled = row.enabled,
                    c.rag_embedding = row.embedding,
                    c.rag_embedding_version = CASE WHEN row.enabled THEN $version ELSE null END
                WITH c, row
                OPTIONAL MATCH (c)-[old:RAG_MENTIONS]->(:Concept)
                DELETE old
                WITH DISTINCT c, row
                WHERE row.enabled
                UNWIND row.topics AS key
                MERGE (concept:Concept {key: key})
                ON CREATE SET concept.name = $names[key], concept.kind = 'Topic'
                MERGE (c)-[:RAG_MENTIONS]->(concept)
            """, chunks=chunks, version=EMBEDDING_VERSION, names={key: entry[0] for key, entry in TOPICS.items()}).consume()
            session.run("""
                CREATE FULLTEXT INDEX rag_chunks_text IF NOT EXISTS
                FOR (c:Chunk) ON EACH [c.rag_search_text]
            """).consume()
            session.run(f"""
                CREATE VECTOR INDEX rag_chunks_vector IF NOT EXISTS
                FOR (c:Chunk) ON (c.rag_embedding)
                OPTIONS {{indexConfig: {{`vector.dimensions`: {EMBEDDING_DIM},
                                      `vector.similarity_function`: 'cosine'}}}}
            """).consume()
            for name in ("rag_chunks_text", "rag_chunks_vector"):
                session.run("CALL db.awaitIndex($name, 60)", name=name).consume()
    return {"articles": len({row["url"] for row in rows}), "chunks": len(chunks), "indexed": len(searchable)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-models", action="store_true")
    args = parser.parse_args()
    if args.download_models:
        download_models()
    print(json.dumps(build_indexes()))
