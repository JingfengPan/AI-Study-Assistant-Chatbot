"""Supplemental, validated LLM judging for evaluation outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass

from config import Settings
from llm import _call_text


JUDGE_PROMPT_VERSION = "rag-judge-v1"


@dataclass(frozen=True)
class JudgeResult:
    correctness: float
    groundedness: float
    relevance: float
    reason: str
    model: str
    prompt_version: str = JUDGE_PROMPT_VERSION


class JudgeError(RuntimeError):
    """Raised when judge output remains invalid after one repair attempt."""


def _parse_judge_output(text: str, model: str) -> JudgeResult:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JudgeError("Judge output was not valid JSON.") from exc
    required = {"correctness", "groundedness", "relevance", "reason"}
    if set(payload) != required:
        raise JudgeError("Judge output has unexpected fields.")
    scores = {}
    for field in ("correctness", "groundedness", "relevance"):
        try:
            score = float(payload[field])
        except (TypeError, ValueError) as exc:
            raise JudgeError(f"Judge field '{field}' is not numeric.") from exc
        if not 0.0 <= score <= 1.0:
            raise JudgeError(f"Judge field '{field}' is outside [0, 1].")
        scores[field] = score
    reason = str(payload["reason"]).strip()
    if not reason:
        raise JudgeError("Judge reason is empty.")
    return JudgeResult(reason=reason, model=model, **scores)


def judge_answer(
    *,
    question: str,
    reference_answer: str | None,
    evidence: list[str],
    generated_answer: str,
    answerable: bool,
    client,
    settings: Settings,
) -> JudgeResult:
    """Judge one answer from only the reference and supplied evidence."""

    instructions = (
        "You are a strict evaluation judge. Score the generated answer only against "
        "the question, reference answer, and evidence excerpts. Do not use outside "
        "knowledge. For an unanswerable item, a correct refusal earns full scores. "
        "Return only a JSON object with exactly these fields: correctness, "
        "groundedness, relevance, reason. Each score must be a number from 0 to 1; "
        "reason must be one brief string."
    )
    prompt = (
        f"QUESTION\n{question}\n\n"
        f"ANSWERABLE\n{answerable}\n\n"
        f"REFERENCE ANSWER\n{reference_answer or '(none; refusal expected)'}\n\n"
        f"EVIDENCE EXCERPTS\n"
        + "\n\n".join(evidence or ["(none)"])
        + f"\n\nGENERATED ANSWER\n{generated_answer}"
    )
    first = _call_text(
        client=client,
        model=settings.chat_model,
        instructions=instructions,
        input_text=prompt,
        temperature=0.0,
        max_output_tokens=300,
    )
    try:
        return _parse_judge_output(first.text, settings.chat_model)
    except JudgeError:
        repair = _call_text(
            client=client,
            model=settings.chat_model,
            instructions=(
                instructions
                + " The prior output was invalid. Repair it without changing the "
                "evaluation judgment."
            ),
            input_text=f"INVALID OUTPUT\n{first.text}\n\nORIGINAL INPUT\n{prompt}",
            temperature=0.0,
            max_output_tokens=300,
        )
        return _parse_judge_output(repair.text, settings.chat_model)

