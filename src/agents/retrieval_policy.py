"""Frozen Q-learning policy for RAG corrections, with a fixed-rule fallback."""

import json
from functools import lru_cache

from ..config import ROOT

ACTIONS = ("aliases", "graph", "stop")
POLICY_PATH = ROOT / "models" / "rag_policy.json"


def state_of(result, scoped: bool, tried: int) -> tuple[int, int]:
    return (int(bool(result.sources)) + 2 * int(bool(result.missing)) + 4 * int(scoped), tried)


def state_key(state):
    return f"{state[0]}:{state[1]}"


def allowed(tried: int) -> list[int]:
    return [index for index in range(3) if index == 2 or not tried & (1 << index)]


def prefer(previous, candidate):
    # Keep partial evidence if the next search loses a supported subquestion.
    if (bool(candidate.sources), -len(candidate.missing)) >= (bool(previous.sources), -len(previous.missing)):
        return candidate
    return previous


@lru_cache(maxsize=1)
def _load():
    try:
        artifact = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        return artifact["values"] if artifact["actions"] == list(ACTIONS) else {}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


class RetrievalRLAgent:
    def choose(self, result, scoped: bool, tried: int) -> tuple[str, str]:
        values = _load().get(state_key(state_of(result, scoped, tried)))
        if isinstance(values, list) and len(values) == len(ACTIONS):
            index = max(allowed(tried), key=values.__getitem__)
            return ACTIONS[index], "q_learning"
        return ("stop" if tried else "aliases"), "fixed_fallback"
