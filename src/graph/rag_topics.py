"""Vietnamese terminology and audience metadata for Diabet retrieval."""

import re

from .kb import normalize

# Each entry provides the graph key, a label for traces, and accepted spellings.
TOPICS = {
    "hba1c": ("HbA1c", r"hba\s?1c|\ba1c\b|hemoglobin glycosyl|duong huyet trung binh"),
    "glucose_mau_luc_doi": ("Đường huyết lúc đói", r"luc doi|duong huyet doi|glucose doi|\bfpg\b|\bifg\b|nhin an"),
    "nghiem_phap_dung_nap_glucose": ("Nghiệm pháp dung nạp glucose", r"\bogtt\b|dung nap glucose|uong (?:nuoc )?duong|uong 75\s?g|uong glucose|\bigt\b"),
    "random_glucose": ("Đường huyết bất kỳ", r"bat ky|bat ki|glucose ngau nhien|duong huyet ngau nhien"),
    "prediabetes": ("Tiền đái tháo đường", r"tien (?:tieu duong|dai thao duong|dtd)|prediabetes"),
    "type1": ("Đái tháo đường típ 1", r"\b(?:tip|tuyp|type|loai)\s*1\b"),
    "type2": ("Đái tháo đường típ 2", r"\b(?:tip|tuyp|type|loai)\s*2\b"),
    "symptoms": ("Triệu chứng", r"trieu chung|dau hieu|tieu nhieu|khat nuoc|uong nhieu|sut can|gay nhieu"),
    "prevention": ("Phòng ngừa", r"phong ngua|phong benh|ngan ngua|giam nguy co|yeu to nguy co"),
    "lifestyle": ("Ăn uống và vận động", r"che do an|dinh duong|tap the duc|van dong|an com|thuc don|loi song|giam can|the thao"),
    "complications": ("Biến chứng", r"bien chung|vong mac|nhin mo|thi luc|benh mat|anh huong mat|dau chan|than kinh|tim mach|\bthan\b"),
    "diagnosis": ("Chẩn đoán và xét nghiệm", r"chan doan|xet nghiem|sang loc|tam soat"),
    "insulin": ("Insulin và thuốc", r"insulin|metformin|thuoc ha duong|thuoc dieu tri"),
    "hypoglycemia": ("Hạ đường huyết", r"ha duong huyet|tut duong|hypoglyc"),
    "dka": ("Nhiễm toan ceton", r"nhiem toan ceton|ketoacidosis|\bdka\b"),
    "monitor": ("Theo dõi đường huyết", r"do (?:duong huyet|glucose) lien tuc|\bcgm\b|may do duong|tu theo doi duong huyet|do duong huyet tai nha"),
    "postpartum": ("Theo dõi sau sinh", r"sau sinh|postpartum|after delivery|after (?:you )?giv(?:e|ing) birth|after (?:i|you) have (?:my|your) baby"),
    "kidney": ("Biến chứng thận", r"suy than|benh than|bien chung than|albumin nieu|\begfr\b|kidney disease|kidney damage|nephropathy"),
    "foot": ("Biến chứng bàn chân", r"ban chan|loet chan|foot (?:ulcers?|problems?|care)"),
    "eye": ("Biến chứng mắt", r"vong mac|nhin mo|thi luc|benh mat|anh huong mat|retinopathy|eye disease"),
}
TESTS = frozenset({"hba1c", "glucose_mau_luc_doi", "nghiem_phap_dung_nap_glucose", "random_glucose"})
DISEASES = frozenset({"type1", "type2", "prediabetes"})
POPULATIONS = {
    "pregnancy": r"thai ky|mang thai|thai phu|bau bi|\bba bau\b|dang bau|sau sinh|pregnan|gestational|postpartum",
    "pediatric": r"tre em|tre nho|tre so sinh|thieu nien|\bbe \d+ tuoi\b|children|childhood|youth|pediatric|teens",
}

EXTRA_TERMS = {
    "hba1c": r"glycated hemoglobin",
    "glucose_mau_luc_doi": r"fasting (?:plasma |blood )?glucose|fasting blood sugar",
    "nghiem_phap_dung_nap_glucose": r"oral glucose tolerance|glucose challenge|nghiem phap 75",
    "random_glucose": r"random (?:plasma |blood )?glucose",
    "prediabetes": r"pre[- ]diabetes",
    "symptoms": r"symptoms?|frequent urination|excessive thirst",
    "prevention": r"prevent(?:ion|ing)?|risk factors?",
    "lifestyle": r"meal planning|healthy eating|carb(?:ohydrate)?s|physical activity|exercise|khau phan|an uong",
    "complications": r"complications?|retinopathy|neuropathy|kidney disease|foot (?:ulcers?|problems?)|stroke|heart disease|ban chan|loet chan|dot quy|suy than|benh than",
    "diagnosis": r"diagnos|screening|blood test|tam soat",
    "insulin": r"medicines?|medications?|metformin|sulfonylurea",
    "hypoglycemia": r"low (?:blood sugar|blood glucose)|ha glucose",
    "dka": r"ketoacidosis|nhiem toan keton",
    "monitor": r"continuous glucose monitoring|glucose monitors?|fingerstick|cam bien duong huyet",
}


def match_topics(value: str) -> list[str]:
    value = normalize(value)
    return [key for key, (_, pattern) in TOPICS.items()
            if re.search(pattern, value) or (key in EXTRA_TERMS and re.search(EXTRA_TERMS[key], value))]


def query_topics(value: str) -> list[str]:
    matched = match_topics(value)
    # When a specific topic is known, generic "xét nghiệm" queries should not
    # independently qualify a chunk that is about a different test.
    if TESTS.intersection(matched) or "monitor" in matched:
        matched = [key for key in matched if key != "diagnosis"]
    return matched


def audiences(value: str) -> list[str]:
    value = normalize(value)
    value = re.sub(r"\b(?:khong (?:mang thai|co thai)|not pregnant|non[- ]pregnant)\b", "", value)
    return [key for key, pattern in POPULATIONS.items() if re.search(pattern, value)]


def is_comparison(value: str) -> bool:
    return bool(re.search(r"\b(?:khac|giong|phan biet|so sanh|so voi)\b", normalize(value)))


def is_follow_up(question: str) -> bool:
    return bool(re.search(r"^(?:con\s+(?:no|cai do|chi so nay|viec do)|(?:the|vay)\s+(?:no|cai do|chi so nay|neu|thi)|(?:no|cai do|chi so nay)\b)", normalize(question).strip()))


def is_model_question(question: str) -> bool:
    return bool(re.search(
        r"\b(?:brfss|mo hinh|diem tham khao|diem du doan|diem nay|ket qua (?:du doan|nay|vua tinh|cua toi|(?:cua )?(?:web|ung dung))|web (?:bao|cho|hien))\b",
        normalize(question),
    ))
