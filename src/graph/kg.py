"""Apply the maintained Cypher file to the configured Neo4j database."""

import json

from neo4j import GraphDatabase

from ..config import ROOT, neo4j_config

CYPHER_FILE = ROOT / "database" / "neo4j" / "knowledge_graph.cypher"


def build_kg() -> dict:
    raw = CYPHER_FILE.read_text(encoding="utf-8")
    statements = "\n".join(line for line in raw.splitlines() if not line.lstrip().startswith("//"))
    cfg = neo4j_config()
    with GraphDatabase.driver(cfg["uri"], auth=cfg["auth"]) as driver:
        with driver.session(database=cfg["database"]) as session:
            for statement in statements.split(";"):
                if statement.strip():
                    session.run(statement).consume()
            result = session.run("""
                MATCH (:Concept)-[r:HAS_KNOWLEDGE]->(:Concept)
                WHERE r.curated_by = 'diabet-kb'
                RETURN count(r) AS evidence_relations, collect(DISTINCT r.predicate) AS predicates,
                       collect(DISTINCT r.page) AS pages
            """).single()
            return result.data()


if __name__ == "__main__":
    print(json.dumps(build_kg(), ensure_ascii=True, indent=2))
