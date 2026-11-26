import json
from .kb import ingest

if __name__ == "__main__":
    results = ingest()
    print(json.dumps(results, ensure_ascii=True, indent=2))
    if not any(item["status"] == "ok" for item in results):
        raise SystemExit("Không nạp được bài Vinmec nào. Kiểm tra mạng và URL nguồn.")
