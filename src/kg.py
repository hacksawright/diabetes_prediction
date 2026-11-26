"""Build a small, source-linked medical knowledge graph from the Vinmec KB."""
import json
import re

from neo4j import GraphDatabase

from .config import neo4j_config
from .kb import normalize

# Each pattern must appear in the evidence paragraph. Relations are limited to
# broad educational statements; they are not diagnostic rules or thresholds.
RULES = [
    ("HAS_SYMPTOM", "Symptom", "Tiểu nhiều", r"tieu nhieu|di tieu (thuong xuyen|nhieu)"),
    ("HAS_SYMPTOM", "Symptom", "Khát nước", r"khat nuoc|uong nhieu"),
    ("HAS_SYMPTOM", "Symptom", "Sụt cân", r"sut can|giam can"),
    ("DIAGNOSED_WITH", "Test", "HbA1c", r"hba1c|hba1c|hemoglobin a1c"),
    ("DIAGNOSED_WITH", "Test", "Glucose máu lúc đói", r"glucose (huyet tuong |mau )?luc doi|duong (huyet|mau) luc doi"),
    ("DIAGNOSED_WITH", "Test", "Nghiệm pháp dung nạp glucose", r"nghiem phap (dung nap|tang duong)|uong 75g glucose"),
]


def build_kg() -> dict:
    cfg = neo4j_config()
    created = 0
    with GraphDatabase.driver(cfg["uri"], auth=cfg["auth"]) as driver:
        with driver.session(database=cfg["database"]) as session:
            session.run("CREATE CONSTRAINT concept_key IF NOT EXISTS FOR (c:Concept) REQUIRE c.key IS UNIQUE").consume()
            rows = session.run("""MATCH (a:Article)-[:HAS_CHUNK]->(c:Chunk)
                RETURN a.url AS url, a.title AS title, c.id AS chunk_id, c.text AS text""").data()
            # Rebuild evidence edges so refreshing articles cannot leave stale claims.
            session.run("MATCH ()-[r:HAS_KNOWLEDGE]->() DELETE r").consume()
            session.run("MERGE (d:Concept {key:'diabetes'}) SET d.name='Đái tháo đường', d.kind='Disease'").consume()
            for row in rows:
                text = normalize(row["text"])
                for relation, kind, name, pattern in RULES:
                    if not re.search(pattern, text):
                        continue
                    # Symptom claims are linked only to the type 2 overview;
                    # tests are linked to articles explicitly about diagnosis.
                    title = normalize(row["title"])
                    if relation == "HAS_SYMPTOM" and "tieu duong tuyp 2" not in title:
                        continue
                    if relation == "DIAGNOSED_WITH" and "chan doan" not in title:
                        continue
                    key = normalize(name).replace(" ", "_")
                    session.run("""MATCH (d:Concept {key:'diabetes'}), (ch:Chunk {id:$chunk_id})
                        MERGE (c:Concept {key:$key}) SET c.name=$name, c.kind=$kind
                        MERGE (ch)-[:MENTIONS]->(c)
                        MERGE (d)-[r:HAS_KNOWLEDGE {evidence_chunk_id:$chunk_id, predicate:$predicate}]->(c)
                        SET r.source_url=$url, r.evidence=$evidence""",
                        chunk_id=row["chunk_id"], key=key, name=name, kind=kind,
                        predicate=relation, url=row["url"], evidence=row["text"][:500]).consume()
                    created += 1
    return {"articles_scanned": len({r["url"] for r in rows}), "evidence_relations": created,
            "note": "HAS_KNOWLEDGE.predicate phân biệt HAS_SYMPTOM và DIAGNOSED_WITH."}


if __name__ == "__main__":
    print(json.dumps(build_kg(), ensure_ascii=True, indent=2))
