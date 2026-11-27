// A source-linked graph view in Neo4j Browser.
MATCH path = (d:Concept)-[r:HAS_KNOWLEDGE]->(c:Concept)
WHERE r.curated_by = 'diabet-kb'
RETURN path LIMIT 40;

// Verify the section, original source and PDF page for every relationship.
MATCH (d:Concept)-[r:HAS_KNOWLEDGE]->(c:Concept)
MATCH (a:Article)-[:HAS_CHUNK]->(chunk:Chunk {id: r.evidence_chunk_id})
WHERE r.curated_by = 'diabet-kb'
RETURN d.name AS topic, r.predicate AS relation, c.name AS concept,
       a.title AS source, r.source_url AS url, r.page AS page, chunk.section AS section
ORDER BY topic, relation, page;
