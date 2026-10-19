"""Verifiers for single-turn tasks in the real-model runner.

A verifier receives the task record and the decoded response and returns a
``Verdict``. ``valid=False`` means the grader could not evaluate the output
(infrastructure/evaluator failure); the whole group is then quarantined and
retried, never trained as a zero reward. ``hack=True`` marks a confirmed
exploit (e.g. the response tampered with the grader).

Custom verifiers are ``module:function`` callables with the signature
``fn(task: dict, response: str) -> Verdict | float | dict``.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import re
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class Verdict:
    score: float
    valid: bool = True
    hack: bool = False
    detail: str = ""


_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_NUMBER = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?")


def _normalize_number(text: str) -> float | None:
    text = text.strip().replace(",", "").rstrip(".")
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            return float(numerator) / float(denominator)
        return float(text)
    except (ValueError, ZeroDivisionError):
        return None


def extract_final_answer(response: str) -> str | None:
    """The last ``\\boxed{...}``, else the last number in the response."""
    boxed = _BOXED.findall(response)
    if boxed:
        return boxed[-1].strip()
    numbers = _NUMBER.findall(response)
    return numbers[-1] if numbers else None


def math_answer(task: Mapping[str, Any], response: str) -> Verdict:
    """Numeric equality (relative tolerance 1e-6), else normalized string equality."""
    expected = str(task["answer"]).strip()
    found = extract_final_answer(response)
    if found is None:
        return Verdict(0.0, detail="no final answer")
    a, b = _normalize_number(found), _normalize_number(expected)
    if a is not None and b is not None:
        return Verdict(float(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)))
    return Verdict(float(re.sub(r"\s+", "", found) == re.sub(r"\s+", "", expected)))


def exact_match(task: Mapping[str, Any], response: str) -> Verdict:
    return Verdict(float(response.strip() == str(task["answer"]).strip()))


BUILTIN = {"math_answer": math_answer, "exact_match": exact_match}


def load_verifier(spec: str) -> Callable[[Mapping[str, Any], str], Verdict]:
    if spec in BUILTIN:
        function = BUILTIN[spec]
    elif ":" in spec:
        module, _, name = spec.partition(":")
        function = getattr(importlib.import_module(module), name)
    else:
        raise ValueError(f"unknown verifier {spec!r}; use {sorted(BUILTIN)} or module:function")

    def verify(task: Mapping[str, Any], response: str) -> Verdict:
        try:
            result = function(task, response)
        except Exception as error:  # noqa: BLE001 - an evaluator crash is not a policy failure.
            return Verdict(0.0, valid=False, detail=f"verifier error: {error}"[:300])
        if isinstance(result, Verdict):
            verdict = result
        elif isinstance(result, Mapping):
            verdict = Verdict(float(result.get("score", 0.0)), bool(result.get("valid", True)),
                              bool(result.get("hack", False)), str(result.get("detail", "")))
        else:
            verdict = Verdict(float(result))
        if verdict.valid and (not math.isfinite(verdict.score) or not 0.0 <= verdict.score <= 1.0):
            return Verdict(0.0, valid=False, detail=f"score {verdict.score!r} outside [0, 1]")
        return verdict

    return verify
