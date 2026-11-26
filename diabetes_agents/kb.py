"""Vinmec article ingestion and cited retrieval in Neo4j."""
import re
import unicodedata
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup
from neo4j import GraphDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import neo4j_config

SOURCES = [
    "https://www.vinmec.com/vie/benh/tieu-duong-tuyp-2-3042",
    "https://www.vinmec.com/vie/bai-viet/xet-nghiem-chan-doan-dai-thao-duong-vi",
    "https://www.vinmec.com/vie/bai-viet/dau-hieu-va-tieu-chuan-chan-doan-tien-tieu-duong-vi",
    "https://www.vinmec.com/vie/bai-viet/cac-xet-nghiem-can-thuc-hien-trong-chan-doan-dai-thao-duong-vi",
]


def normalize(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", value.lower()) if unicodedata.category(c) != "Mn").replace("đ", "d")


def extract_article(html: str) -> tuple[str, list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "form"]):
        tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else "Vinmec"
    root = soup.find("article") or soup.find("main") or soup.body or soup
    paragraphs = []
    seen = set()
    for element in root.find_all(["p", "h2", "h3", "li"]):
        value = re.sub(r"\s+", " ", element.get_text(" ", strip=True))
        if 50 <= len(value) <= 1200 and value not in seen:
            seen.add(value)
            paragraphs.append(value)
    return title, paragraphs


def ingest() -> list[dict]:
    cfg = neo4j_config()
    results = []
    with GraphDatabase.driver(cfg["uri"], auth=cfg["auth"]) as driver:
        driver.verify_connectivity()
        for url in SOURCES:
            try:
                response = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0 (educational diabetes KB; source attribution)"})
                response.raise_for_status()
                title, paragraphs = extract_article(response.text)
                if not paragraphs:
                    raise ValueError("Không tìm thấy đoạn nội dung")
                now = datetime.now(timezone.utc).isoformat()
                with driver.session(database=cfg["database"]) as session:
                    session.run("MERGE (a:Article {url:$url}) SET a.title=$title, a.fetched_at=$now", url=url, title=title, now=now).consume()
                    # Replace old chunks only after a successful fetch and parse.
                    session.run("MATCH (a:Article {url:$url})-[:HAS_CHUNK]->(c:Chunk) DETACH DELETE c", url=url).consume()
                    for index, paragraph in enumerate(paragraphs[:80]):
                        session.run("""MATCH (a:Article {url:$url})
                            CREATE (a)-[:HAS_CHUNK]->(c:Chunk {id:$id, text:$text, normalized:$normalized, position:$position})""",
                            url=url, id=f"{url}#{index}", text=paragraph, normalized=normalize(paragraph), position=index).consume()
                results.append({"url": url, "title": title, "chunks": min(len(paragraphs), 80), "status": "ok"})
            except (requests.RequestException, ValueError) as exc:
                results.append({"url": url, "status": "error", "error": str(exc)})
    return results


def retrieve(query: str, limit: int = 4) -> list[dict]:
    cfg = neo4j_config()
    with GraphDatabase.driver(cfg["uri"], auth=cfg["auth"]) as driver:
        with driver.session(database=cfg["database"]) as session:
            rows = session.run("""MATCH (a:Article)-[:HAS_CHUNK]->(c:Chunk)
                RETURN a.url AS url, a.title AS title, c.text AS text, c.normalized AS normalized""").data()
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
    return [{"url": rows[i]["url"], "title": rows[i]["title"],
             "excerpt": rows[i]["text"][:500], "score": round(float(score), 4)}
            for i, score in ranked[:limit] if score > 0]
