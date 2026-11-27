"""Train offline using replayed Neo4j retrieval and question-level source labels."""

import json
from random import Random

from ..agents.retrieval_policy import ACTIONS, POLICY_PATH, allowed, prefer, state_key, state_of
from ..agents.rl import QLearningAgent
from ..config import ROOT
from .evaluate_kb_coverage import CASES
from ..graph.retrieval import search

TRAIN = CASES[::2] + [
    ("Bộ Y tế quy định ngưỡng CGM ở thai phụ như thế nào?", set(), None),
    ("Ngưỡng HbA1c chẩn đoán típ 1 ở trẻ sơ sinh là bao nhiêu?", set(), None),
    ("Mức CGM bao nhiêu chẩn đoán đái tháo đường thai kỳ?", set(), None),
    ("Cách xét nghiệm OGTT khi đang mang thai ở tuần 24?", {"moh-1470", "niddk-gestational-tests"}, None),
]
TEST = CASES[1::2] + [
    ("Theo QĐ 1470, ngưỡng HbA1c để chẩn đoán đái tháo đường ở trẻ em là bao nhiêu?", set(), None),
    ("Theo Bộ Y tế ngưỡng OGTT thai kỳ lúc đói là bao nhiêu?", {"moh-1470"}, 15),
    ("Theo Bộ Y tế, ngưỡng máy CGM ở trẻ em là gì?", set(), None),
]


def collect(cases):
    return [(question, expected, page, search(question, collect_replay=True))
            for question, expected, page in cases]


def quality(result, expected, page):
    if not expected:
        return not result.sources
    return not result.missing and any(
        row.get("source_id") in expected and (page is None or row.get("page") == page)
        for row in result.sources
    )


def episode(case, select, agent=None):
    question, expected, page, initial = case
    current, tried = initial, 0
    total, history = 0.0, []
    while initial.replay and (not current.sources or current.missing) and tried != 3:
        state = state_of(current, initial.scoped, tried)
        action = select(state)
        history.append(ACTIONS[action])
        if action == 2:
            next_result = current
            terminal = True
        else:
            next_result = prefer(current, initial.replay[ACTIONS[action]])
            tried |= 1 << action
            terminal = tried == 3 or bool(next_result.sources and not next_result.missing)
        correct = quality(next_result, expected, page)
        reward = (-0.06 if action == 0 else -0.10 if action == 1 else 0.0)
        if terminal:
            reward += 1.0 if correct else -1.0
        if agent is not None:
            agent.learn(state, action, reward, state_of(next_result, initial.scoped, tried),
                        terminal=terminal, next_actions=allowed(tried))
        current = next_result
        total += reward
        if terminal:
            break
    return {"question": question, "correct": quality(current, expected, page), "actions": history,
            "retry_count": tried.bit_count(), "reward": round(total, 4)}


def train():
    train_cases = collect(TRAIN)
    test_cases = collect(TEST)
    rng = Random(42)
    agent = QLearningAgent([(features, tried) for features in range(8) for tried in range(4)], 3, 0.10, 0.90, 42)
    visited = set()

    def explore(state):
        visited.add(state)
        options = allowed(state[1])
        return rng.choice(options) if rng.random() < 0.25 else max(options, key=agent.q[state].__getitem__)

    for _ in range(5000):
        episode(rng.choice(train_cases), explore, agent)

    def fixed(state):
        return 2 if state[1] else 0

    def learned(state):
        return max(allowed(state[1]), key=agent.q[state].__getitem__) if state in visited else fixed(state)

    comparisons = {}
    for name, choose in (("fixed", fixed), ("q_learning", learned)):
        records = [episode(case, choose) for case in test_cases]
        comparisons[name] = {
            "correct": sum(row["correct"] for row in records),
            "retries": sum(row["retry_count"] for row in records), "cases": records,
        }
    artifact = {"actions": list(ACTIONS), "values": {state_key(state): agent.q[state] for state in sorted(visited)}}
    POLICY_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    report = {
        "train_size": len(TRAIN), "test_size": len(TEST), "visited_states": len(visited), "comparisons": comparisons,
    }
    path = ROOT / "artifacts" / "rag_rl_evaluation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    train()
