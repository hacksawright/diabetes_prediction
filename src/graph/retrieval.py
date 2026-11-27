"""Hybrid Neo4j retrieval, reranking, and one in-KB corrective attempt."""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass

from neo4j import GraphDatabase

from ..config import neo4j_config
from ..agents.retrieval_policy import ACTIONS, RetrievalRLAgent, prefer
from . import rag_models
from .kb import normalize
from .rag_topics import DISEASES, TESTS, TOPICS, audiences, is_comparison, match_topics, query_topics

logger = logging.getLogger(__name__)

CANDIDATES = 20
RERANK_LIMIT = 16
RRF_K = 60
SPECIFIC_TOPICS = TESTS | {"hypoglycemia", "dka", "monitor", "postpartum", "kidney", "foot", "eye"}
THRESHOLD_QUERY = (
    r"nguong|tieu chuan|(?:bao nhieu|chi so nao).*?(?:chan doan|la cao|la thap|binh thuong|tien tieu duong)"
    r"|(?:chi so|muc|gia tri|duong huyet|hba1c).{0,70}bao nhieu"
)
THRESHOLD_EVIDENCE = r"\d\s*(?:%|phan tram|mg\s*/\s*dl|mmol)"
STOPWORDS = set("diabetes dai thao duong tieu la gi co khong nao bao nhieu nen nhu the can cho hoi toi em khi thi va voi giup ve cua dang bi".split())
RETURN_PASSAGE = """
    MATCH (a:Article)-[:HAS_CHUNK]->(node)
    RETURN node.id AS id, a.url AS url, a.title AS title,
           coalesce(node.rag_passage, node.text) AS excerpt,
           coalesce(node.rag_topics, []) AS topics,
           coalesce(node.rag_audiences, []) AS audiences,
           properties(node).section AS section,
           coalesce(properties(node).page_start, properties(node).page) AS page,
           properties(node).purpose AS purpose, properties(a).source_id AS source_id,
           properties(a).publisher AS publisher
"""


@dataclass
class RetrievalResult:
    sources: list[dict]
    missing: list[str]
    attempts: int
    strategy: str
    actions: tuple[str, ...] = ()
    policy: str = "fixed"
    replay: dict | None = None
    scoped: bool = False


def _rrf(rankings):
    scores = defaultdict(float)
    rows = {}
    for candidates, weight in rankings:
        visited = set()
        for position, candidate in enumerate(candidates, 1):
            key = candidate["id"]
            if key in visited:
                continue
            visited.add(key)
            scores[key] += weight / (RRF_K + position)
            rows[key] = candidate
    return [
        {**rows[key], "score": round(scores[key], 6)}
        for key in sorted(rows, key=lambda key: (-scores[key], key))
    ]


def _lexical(question: str, topics: list[str], corrective: bool) -> str:
    terms = list(dict.fromkeys(re.findall(r"[a-z0-9]+", normalize(question))))
    terms = [term for term in terms if term not in STOPWORDS and len(term) > 1][:22]
    clauses = terms.copy()
    if corrective:
        clauses.extend('"' + normalize(TOPICS[key][0]) + '"' for key in topics)
    return " OR ".join(clauses)


def _eligible(row, topics, scope, compare, numerical, by_ministry=False):
    row_topics = set(row["topics"])
    row_scope = set(row["audiences"]) - {"general"}
    if by_ministry and row.get("publisher") != "Bộ Y tế Việt Nam":
        return False
    body = row["excerpt"]
    if len(body) < 120 and re.match(r"^\d+(?:\.\d+)*\.?\s", body) and not re.search(THRESHOLD_EVIDENCE, normalize(body)):
        return False
    if scope and (row_scope or not compare) and not scope.intersection(row_scope):
        return False
    if not scope and row_scope and "general" not in row["audiences"]:
        return False

    topics = set(topics)
    specific = topics.intersection(SPECIFIC_TOPICS) or topics - DISEASES - {"diagnosis"}
    if "postpartum" in topics and not compare and "postpartum" not in row_topics:
        return False
    if specific and not specific.intersection(row_topics):
        return False
    if "prediabetes" in topics and not compare and "prediabetes" not in row_topics:
        return False
    if not specific and topics and not row_topics.intersection(topics):
        return False
    if numerical:
        if row.get("purpose") == "management":
            return False
        if not re.search(THRESHOLD_EVIDENCE, normalize(row["excerpt"])):
            return False
        if not compare and "prediabetes" not in topics and "prediabetes" in row_topics:
            return False
    return True


def _covered(rows, topics, scope, compare):
    present_topics = {topic for row in rows for topic in row["topics"]}
    if compare:
        present_topics -= TESTS
        present_topics.update(topic for row in rows for topic in _primary_tests(row))
    present_scope = {audience for row in rows for audience in row["audiences"]}
    requested = set(topics).intersection(SPECIFIC_TOPICS)
    if compare:
        requested |= set(topics).intersection(DISEASES)
    missing = [TOPICS[key][0] for key in topics if key in requested and key not in present_topics]
    missing.extend({"pregnancy": "thai kỳ", "pediatric": "trẻ em"}[item] for item in sorted(scope - present_scope))
    return missing


def _primary_tests(row) -> set[str]:
    return TESTS.intersection(match_topics(row["excerpt"][:110]))


def _select(candidates, topics, scope, compare, limit):
    selected = []
    uncovered = set(topics).intersection(TESTS | DISEASES)
    unseen_audiences = scope.copy()
    seen_passages = set()
    # Cover explicit subquestions before filling remaining slots by rank.
    for required_only in (True, False):
        for row in candidates:
            body = normalize(row["excerpt"])
            if body in seen_passages:
                continue
            dedicated = _primary_tests(row)
            new_topics = uncovered.intersection((set(row["topics"]) - TESTS) | dedicated)
            new_audiences = unseen_audiences.intersection(row["audiences"])
            if required_only and (not compare or not (new_topics or new_audiences)):
                continue
            selected.append(row)
            seen_passages.add(body)
            uncovered.difference_update(new_topics)
            unseen_audiences.difference_update(new_audiences)
            if len(selected) >= limit:
                return selected
    return selected


def _attempt(query_rows, rerank, indexes, question, vector, topics, scope, compare, numerical, limit, action):
    by_ministry = bool(re.search(r"(?:theo (?:huong dan |quyet dinh )?bo y te|\b(?:1470|5481)\b)", normalize(question)))
    lexical = _lexical(question, topics, action == "aliases")
    fulltext = query_rows(("fulltext", lexical), """
        CALL db.index.fulltext.queryNodes('rag_chunks_text', $terms, {limit: $limit})
        YIELD node, score
        WITH node, score WHERE node.rag_enabled
    """ + RETURN_PASSAGE + " ORDER BY score DESC, id",
        terms=lexical, limit=CANDIDATES,
    ) if lexical else []
    dense = query_rows(("vector",), """
        CALL db.index.vector.queryNodes('rag_chunks_vector', $limit, $embedding)
        YIELD node, score
        WITH node, score WHERE node.rag_embedding_version = $version
    """ + RETURN_PASSAGE + " ORDER BY score DESC, id",
        limit=CANDIDATES, embedding=vector, version=rag_models.EMBEDDING_VERSION,
    ) if vector is not None and "rag_chunks_vector" in indexes else []
    graph = query_rows(("graph", action == "graph"), """
        UNWIND $topics AS topic
        CALL (topic) {
            MATCH (:Concept {key: topic})<-[:RAG_MENTIONS]-(node:Chunk)
            WHERE node.rag_enabled
            RETURN node ORDER BY
                CASE WHEN any(a IN $scope WHERE a IN node.rag_audiences) THEN 0 ELSE 1 END,
                size(node.rag_topics), node.id LIMIT $per_concept
        }
        WITH node, count(*) AS matches
        ORDER BY matches DESC, node.id LIMIT $limit
    """ + RETURN_PASSAGE,
        topics=topics, scope=list(scope), per_concept=18 if action == "graph" else 8,
        limit=36 if action == "graph" else CANDIDATES,
    ) if topics else []
    candidates = [
        row for row in _rrf(((fulltext, 1.0), (dense, 1.0), (graph, 1.4 if action == "graph" else 0.45)))
        if _eligible(row, topics, scope, compare, numerical, by_ministry)
    ]
    candidates = _select(candidates, topics, scope, compare, RERANK_LIMIT)
    strategy = "hybrid_graph" if vector is not None else "fulltext_graph"
    if candidates and rag_models.available("reranker"):
        semantic_scores = rerank(question, [
            f"{row['section']}\n{row['excerpt']}" if row.get("section") else row["excerpt"]
            for row in candidates
        ])
        candidates = sorted(zip(semantic_scores, candidates), key=lambda item: -item[0])
        candidates = [
            {**row, "rerank_score": round(score, 5)} for score, row in candidates
        ]
        tests = [key for key in topics if key in TESTS]
        if compare and len(tests) > 1:
            best_for_each_test = []
            detail = "mức và đơn vị dùng khi chẩn đoán" if numerical else "ý nghĩa và cách thực hiện xét nghiệm"
            for key in tests:
                group = [row for row in candidates if key in _primary_tests(row)]
                if group:
                    scores = rerank(f"{TOPICS[key][0]}: {detail}", [row["excerpt"] for row in group])
                    best_for_each_test.append(group[max(range(len(group)), key=scores.__getitem__)])
            candidates = best_for_each_test + candidates
        strategy += "_rerank"
    return _select(candidates, topics, scope, compare, limit), strategy


def search(question: str, limit: int = 5, *, policy: str = "fixed", collect_replay: bool = False) -> RetrievalResult:
    from .kb import retrieve as retrieve_tfidf

    topics = query_topics(question)
    normalized = normalize(question)
    explicit_scope = re.sub(r"khong (?:mang thai|co thai)", "", normalized)
    scope = set(audiences(explicit_scope))
    compare = is_comparison(question)
    numerical = bool(re.search(THRESHOLD_QUERY, normalized))
    domain = topics or scope or re.search(r"tieu duong|dai thao duong|duong huyet|glucose|\bdiabetes\b|\bblood sugar\b", normalized)
    if not domain:
        return RetrievalResult([], [], 0, "outside_kb")

    cfg = neo4j_config()
    with GraphDatabase.driver(cfg["uri"], auth=cfg["auth"]) as driver:
        with driver.session(database=cfg["database"]) as session:
            indexes = {
                row["name"] for row in session.run("""
                    SHOW INDEXES YIELD name, state WHERE state = 'ONLINE' AND name STARTS WITH 'rag_chunks_'
                    RETURN name
                """)
            }
            if "rag_chunks_text" not in indexes:
                rows = [
                    {**row, "topics": match_topics(row["excerpt"]), "audiences": audiences(row["excerpt"])}
                    for row in retrieve_tfidf(question, limit=CANDIDATES)
                ]
                selected = [
                    row for row in rows if _eligible(row, topics, scope, compare, numerical)
                ][:limit]
                return RetrievalResult(selected, _covered(selected, topics, scope, compare), 1, "tfidf_fallback")

            vector = None
            if "rag_chunks_vector" in indexes and rag_models.available("e5"):
                vector = rag_models.embed([question], query=True)[0]
            query_cache = {}
            score_cache = {}

            def query_rows(key, cypher, **parameters):
                if key not in query_cache:
                    query_cache[key] = session.run(cypher, **parameters).data()
                return query_cache[key]

            def rerank(query, passages):
                scores = score_cache.setdefault(query, {})
                unseen = list(dict.fromkeys(passage for passage in passages if passage not in scores))
                if unseen:
                    scores.update(zip(unseen, rag_models.rerank(query, unseen)))
                return [scores[passage] for passage in passages]

            def run_action(action: str) -> RetrievalResult:
                selected, strategy = _attempt(
                    query_rows, rerank, indexes, question, vector, topics, scope, compare, numerical, limit, action
                )
                for row in selected:
                    if row["page"] is not None:
                        row["url"] = row["url"].split("#", 1)[0] + f"#page={row['page']}"
                return RetrievalResult(selected, _covered(selected, topics, scope, compare), 1, strategy, scoped=bool(scope))

            current = run_action("initial")
            if collect_replay:
                current.replay = {name: run_action(name) for name in ACTIONS[:2]} if topics else {}
                return current
            actions = []
            tried = 0
            selected_policy = "fixed"
            for _ in ACTIONS:
                if not topics or (current.sources and not current.missing):
                    break
                if policy == "rl":
                    action, selected_policy = RetrievalRLAgent().choose(current, bool(scope), tried)
                else:
                    action = "stop" if tried else "aliases"
                actions.append(action)
                if action == "stop":
                    break
                tried |= 1 << ACTIONS.index(action)
                current = prefer(current, run_action(action))
            current.attempts = 1 + tried.bit_count()
            current.actions = tuple(actions)
            current.policy = selected_policy
            logger.debug("retrieval policy=%s actions=%s attempts=%s missing_count=%s",
                         current.policy, current.actions, current.attempts, len(current.missing))
            return current


def retrieve(question: str, limit: int = 4) -> list[dict]:
    return search(question, limit).sources
