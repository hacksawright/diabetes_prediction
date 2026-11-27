"""Check source coverage of the expanded KB without calling a generator."""

import json
from time import perf_counter

from ..graph.retrieval import search

# Each case specifies the acceptable source IDs. A page expectation requires a
# page-specific citation rather than simply seeing the PDF in the results.
CASES = [
    ("HbA1c cho biết điều gì?", {"niddk-tests", "moh-5481"}, None),
    ("Ngưỡng đường huyết lúc đói để chẩn đoán tiểu đường ở người trưởng thành là bao nhiêu?",
     {"moh-5481"}, 11),
    ("Ngưỡng OGTT 75 g trong thai kỳ theo Bộ Y tế là bao nhiêu?", {"moh-1470"}, 15),
    ("Tầm soát đái tháo đường thai kỳ ở tuần nào?", {"niddk-gestational-tests", "moh-1470"}, None),
    ("Từng bị đái tháo đường thai kỳ thì sau sinh khi nào xét nghiệm lại?", {"moh-1470"}, 30),
    ("Tầm soát đái tháo đường típ 1 cho trẻ em thực hiện thế nào?", {"cdc-type1-screening"}, None),
    ("Máy đo đường huyết liên tục CGM hoạt động như thế nào?", {"cdc-cgm"}, None),
    ("Tụt đường huyết có triệu chứng gì?", {"niddk-hypo"}, None),
    ("Nhiễm toan ceton DKA có dấu hiệu gì?", {"cdc-dka"}, None),
    ("Người tiểu đường nên phân chia khẩu phần bữa ăn thế nào?", {"cdc-meal", "niddk-living"}, None),
    ("Tiểu đường liên quan đến loét bàn chân ra sao?", {"niddk-foot"}, None),
    ("Biến chứng võng mạc tiểu đường ảnh hưởng thị lực thế nào?", {"niddk-eye"}, None),
    ("Bệnh thận do đái tháo đường được theo dõi bằng xét nghiệm gì?", {"niddk-kidney", "moh-5481"}, None),
    ("Nhiệt độ Hà Nội hôm nay bao nhiêu?", set(), None),
]


def evaluate():
    results = []
    for question, expected_ids, page in CASES:
        started = perf_counter()
        result = search(question)
        sources = [{"source_id": row.get("source_id"), "page": row.get("page"),
                    "section": row.get("section"), "url": row["url"]} for row in result.sources]
        matched = any(
            item["source_id"] in expected_ids and (page is None or item["page"] == page)
            for item in sources
        ) if expected_ids else not sources
        results.append({
            "question": question, "pass": matched, "seconds": round(perf_counter() - started, 3),
            "missing": result.missing, "sources": sources,
        })
    return {"passed": sum(item["pass"] for item in results), "cases": len(results), "details": results}


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=True, indent=2))
