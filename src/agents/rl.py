"""Tabular Q-learning for offline retrieval experiments."""

from random import Random

State = tuple[int, int]


class QLearningAgent:
    def __init__(self, states: list[State], action_count: int, alpha: float, gamma: float, seed: int):
        self.q = {state: [0.0] * action_count for state in states}
        self.alpha = alpha
        self.gamma = gamma
        self.rng = Random(seed)

    def choose(self, state: State, epsilon: float = 0.0) -> int:
        values = self.q[state]
        if epsilon > 0 and self.rng.random() < epsilon:
            return self.rng.randrange(len(values))
        return max(range(len(values)), key=values.__getitem__)

    def learn(self, state: State, action: int, reward: float, next_state: State, *, terminal: bool, next_actions=None):
        future = 0.0
        if not terminal:
            actions = range(len(self.q[next_state])) if next_actions is None else next_actions
            future = self.gamma * max(self.q[next_state][index] for index in actions)
        previous = self.q[state][action]
        self.q[state][action] = previous + self.alpha * (reward + future - previous)
