"""A bounded, typed request-understanding step for the orchestrator."""

from __future__ import annotations

from .base import DecisionProvider, DecisionRequest, DecisionResponse


ROUTING_QUESTIONS = {
    "difficulty": {
        "type": "score",
        "instructions": "How hard is `request` for a language model?",
        "criteria": [
            "trivial: a lookup or one-liner",
            "easy: short answer, no reasoning",
            "moderate: several steps",
            "hard: long multi-step reasoning or specialist knowledge",
        ],
    },
    "domain": {
        "type": "choice",
        "instructions": "What domain does `request` belong to?",
        "criteria": {
            "code": "software engineering, programming, refactoring, architecture, debugging",
            "math_or_logic": "mathematics, logic puzzles, proofs, complex calculation",
            "writing": "creative writing, essays, emails, blog posts, copywriting",
            "factual_lookup": "facts, definitions, trivia, history",
            "data_analysis": "statistics, SQL, data manipulation, metrics",
            "chitchat": "casual conversation, greetings, small talk",
        },
    },
    "needs_tools": {
        "type": "noul",
        "instructions": "Does answering `request` require external tools, search or private data?",
    },
    "is_sensitive": {
        "type": "noul",
        "instructions": "Does `request` involve money, legal, medical or safety consequences?",
    },
}


class LayaTaskUnderstanding:
    """Use a decision provider for advisory request classification."""

    def __init__(self, provider: DecisionProvider) -> None:
        self.provider = provider

    def analyze(self, user_request: str) -> DecisionResponse:
        if not user_request.strip():
            raise ValueError("user_request must not be empty")
        return self.provider.predict(
            DecisionRequest(
                state={"request": user_request},
                questions=ROUTING_QUESTIONS,
                model="typed-decisions",
            )
        )
