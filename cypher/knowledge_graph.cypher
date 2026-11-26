// Run after: python -m diabetes_agents.ingest
// Select each statement in Neo4j Browser and click Run, or use cypher-shell -f.
// The source Article and Chunk nodes must already exist in database `neo4j`.

CREATE CONSTRAINT concept_key IF NOT EXISTS
FOR (c:Concept) REQUIRE c.key IS UNIQUE;

MERGE (d:Concept {key: 'diabetes'})
SET d.name = 'Đái tháo đường', d.kind = 'Disease';

// Rebuild evidence relationships from the current KB, avoiding stale claims.
MATCH ()-[r:HAS_KNOWLEDGE]->()
DELETE r;

UNWIND [
  {predicate: 'HAS_SYMPTOM', kind: 'Symptom', key: 'tieu_nhieu', name: 'Tiểu nhiều', pattern: 'tieu nhieu|di tieu (thuong xuyen|nhieu)', scope: 'type2'},
  {predicate: 'HAS_SYMPTOM', kind: 'Symptom', key: 'khat_nuoc', name: 'Khát nước', pattern: 'khat nuoc|uong nhieu', scope: 'type2'},
  {predicate: 'HAS_SYMPTOM', kind: 'Symptom', key: 'sut_can', name: 'Sụt cân', pattern: 'sut can|giam can', scope: 'type2'},
  {predicate: 'DIAGNOSED_WITH', kind: 'Test', key: 'hba1c', name: 'HbA1c', pattern: 'hba1c|hemoglobin a1c', scope: 'diagnosis'},
  {predicate: 'DIAGNOSED_WITH', kind: 'Test', key: 'glucose_mau_luc_doi', name: 'Glucose máu lúc đói', pattern: 'glucose (huyet tuong |mau )?luc doi|duong (huyet|mau) luc doi', scope: 'diagnosis'},
  {predicate: 'DIAGNOSED_WITH', kind: 'Test', key: 'nghiem_phap_dung_nap_glucose', name: 'Nghiệm pháp dung nạp glucose', pattern: 'nghiem phap (dung nap|tang duong)|uong 75g glucose', scope: 'diagnosis'}
] AS rule
MATCH (a:Article)-[:HAS_CHUNK]->(ch:Chunk)
WHERE ch.normalized =~ ('.*(' + rule.pattern + ').*')
  AND ((rule.scope = 'type2' AND a.url CONTAINS '/benh/tieu-duong-tuyp-2-')
    OR (rule.scope = 'diagnosis' AND (a.url CONTAINS 'chan-doan' OR a.url CONTAINS '/benh/tieu-duong-tuyp-2-')))
MATCH (d:Concept {key: 'diabetes'})
MERGE (concept:Concept {key: rule.key})
SET concept.name = rule.name, concept.kind = rule.kind
MERGE (ch)-[:MENTIONS]->(concept)
MERGE (d)-[r:HAS_KNOWLEDGE {evidence_chunk_id: ch.id, predicate: rule.predicate}]->(concept)
SET r.source_url = a.url, r.evidence = substring(ch.text, 0, 500);

// Inspect the structured knowledge graph (Graph view in Neo4j Browser).
MATCH (d:Concept {key: 'diabetes'})-[r:HAS_KNOWLEDGE]->(concept:Concept)
RETURN d, r, concept
LIMIT 30;

// Inspect evidence as a table.
MATCH (d:Concept {key: 'diabetes'})-[r:HAS_KNOWLEDGE]->(concept:Concept)
RETURN d.name AS disease, r.predicate AS relationship, concept.name AS concept,
       r.evidence AS evidence, r.source_url AS source_url
ORDER BY relationship, concept
LIMIT 50;
