"""Legacy TF-IDF baseline and shared text normalization."""
import unicodedata

from neo4j import GraphDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..config import neo4j_config

def normalize(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", value.lower()) if unicodedata.category(c) != "Mn").replace("đ", "d")


def retrieve(query: str, limit: int = 4) -> list[dict]:
    cfg = neo4j_config()
    with GraphDatabase.driver(cfg["uri"], auth=cfg["auth"]) as driver:
        with driver.session(database=cfg["database"]) as session:
            rows = session.run("""MATCH (a:Article)-[:HAS_CHUNK]->(c:Chunk)
                RETURN c.id AS id, a.url AS url, a.title AS title, c.text AS text, c.normalized AS normalized""").data()
    return rank_tfidf(query, rows, limit)


def rank_tfidf(query: str, rows: list[dict], limit: int = 4) -> list[dict]:
    """Rank KB chunks by character n-gram TF-IDF cosine similarity."""
    if not rows or not query.strip():
        return []
    corpus = [normalize(row["title"] + " " + row["normalized"]) for row in rows]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    matrix = vectorizer.fit_transform(corpus)
    scores = cosine_similarity(vectorizer.transform([normalize(query)]), matrix).ravel()
    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
    return [{"id": rows[i].get("id"), "url": rows[i]["url"], "title": rows[i]["title"],
             "excerpt": rows[i]["text"][:500], "score": round(float(score), 4)}
            for i, score in ranked[:limit] if score > 0]
