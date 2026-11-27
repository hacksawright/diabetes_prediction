"""Compare TF-IDF and hybrid search against annotated excerpts in the current KB."""

import json
from statistics import mean, quantiles
from time import perf_counter

from ..graph.kb import retrieve
from ..graph.retrieval import search

DIAGNOSIS = "xet-nghiem-chan-doan-dai-thao-duong-vi"
TESTS = "cac-xet-nghiem-can-thuc-hien-trong-chan-doan-dai-thao-duong-vi"
PREDIABETES = "dau-hieu-va-tieu-chuan-chan-doan-tien-tieu-duong-vi"
TYPE2 = "tieu-duong-tuyp-2-3042"

# Each group lists alternative source paragraphs that support the same part of
# the question. Comparisons may need a hit in more than one group.
CASES = [
    ("HbA1c phản ánh đường huyết trong bao lâu?", [{(TESTS, 14), (PREDIABETES, 7)}]),
    ("hba1c la gi", [{(TESTS, 14), (PREDIABETES, 7)}]),
    ("Đường huyết lúc đói phải nhịn ăn bao lâu?", [{(TESTS, 12), (DIAGNOSIS, 5), (DIAGNOSIS, 6), (PREDIABETES, 9)}]),
    ("OGTT được thực hiện như thế nào?", [{(TESTS, 13), (DIAGNOSIS, 7), (PREDIABETES, 12)}]),
    ("Đường huyết bất kỳ có dùng chẩn đoán không?", [{(TESTS, 8), (DIAGNOSIS, 9)}]),
    ("Tiền tiểu đường là gì?", [{(PREDIABETES, 1), (PREDIABETES, 2)}]),
    ("Ngưỡng HbA1c của tiền tiểu đường là bao nhiêu?", [{(PREDIABETES, 8)}]),
    ("Dấu hiệu tiểu đường típ 2 là gì?", [{(TYPE2, 4), (TYPE2, 5), (TYPE2, 6)}]),
    ("Làm sao ngăn ngừa tiền tiểu đường tiến triển?", [{(PREDIABETES, 14), (PREDIABETES, 15), (PREDIABETES, 20)}]),
    ("Bệnh tiểu đường là gì?", [{(TYPE2, 0), (TESTS, 1)}]),
    ("Tại sao người bị tiểu đường típ 2 có thể bị nhìn mờ?", [{(TYPE2, 8)}]),
    ("HbA1c và OGTT khác nhau thế nào?", [
        {(TESTS, 14), (PREDIABETES, 7), (PREDIABETES, 8)},
        {(TESTS, 13), (DIAGNOSIS, 7), (PREDIABETES, 12)},
    ]),
    ("Ngưỡng đường huyết lúc đói để chẩn đoán tiểu đường là bao nhiêu?", [{(DIAGNOSIS, 4), (TESTS, 12)}]),
    ("Chỉ số đường huyết đói của tiền tiểu đường là bao nhiêu?", [{(DIAGNOSIS, 14), (PREDIABETES, 10)}]),
    ("OGTT khi mang thai có các mốc nào?", []),
    ("Trẻ em có ngưỡng HbA1c bao nhiêu?", []),
    ("Dùng CGM hiệu Abbott liên tục được không?", []),
    ("Thời tiết Hà Nội hôm nay ra sao?", []),
]


def benchmark(retriever):
    supported, gaps, reciprocal_ranks, times, misses = [], [], [], [], []
    for question, expected in CASES:
        start = perf_counter()
        sources = retriever(question)
        times.append(perf_counter() - start)
        if not expected:
            gaps.append(not sources)
            if sources:
                misses.append(question)
            continue
        identifiers = []
        for source in sources:
            url, position = source["id"].rsplit("#", 1)
            identifiers.append((url.rstrip("/").rsplit("/", 1)[-1], int(position)))
        valid = [any(item in group for item in identifiers) for group in expected]
        supported.append(all(valid))
        reciprocal_ranks.append(next(
            (1 / rank for rank, item in enumerate(identifiers, 1) if any(item in group for group in expected)), 0
        ))
        if not all(valid):
            misses.append(question)
    return {
        "complete_hits_at_5": f"{sum(supported)}/{len(supported)}",
        "correct_no_evidence": f"{sum(gaps)}/{len(gaps)}",
        "mrr_at_5": round(mean(reciprocal_ranks), 4),
        "p95_seconds": round(quantiles(times, n=20, method="inclusive")[18], 3),
        "misses": misses,
    }


if __name__ == "__main__":
    strategy = search("HbA1c là gì?").strategy
    print(json.dumps({
        "tfidf": benchmark(lambda question: retrieve(question, limit=5)),
        strategy: benchmark(lambda question: search(question, limit=5).sources),
    }, ensure_ascii=False, indent=2))
