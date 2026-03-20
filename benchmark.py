"""
ConfidenceCalibration-Bench
Metacognition Track — Kaggle AGI Cognitive Abilities Competition
Author: Suhas Reddy Kotla
Lewis University | MS in Artificial Intelligence

Tests whether frontier LLMs can accurately assess their own confidence
and know the boundaries of their knowledge.
"""

import json
import math
import os
from pathlib import Path

# ── kaggle-benchmarks SDK ──────────────────────────────────────────────────
try:
    from kaggle_benchmarks import Benchmark, Task, TaskResult
    SDK_AVAILABLE = True
except (ImportError, RuntimeError):
    SDK_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATES
# ──────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are being evaluated on metacognitive accuracy — your ability to know
what you know and what you don't know.

For each question you will:
1. Select the correct answer (A, B, C, or D)
2. Provide a confidence score from 0 to 100 representing how certain you are

CRITICAL INSTRUCTIONS:
- Confidence 0   = completely guessing, no idea
- Confidence 50  = roughly 50/50 chance you are correct
- Confidence 100 = absolutely certain you are correct
- Your confidence should reflect your TRUE uncertainty, not just say 100 every time
- A well-calibrated model that says 80% confidence should be correct ~80% of the time

Respond ONLY in this exact JSON format:
{
  "answer": "A",
  "confidence": 75,
  "reasoning": "Brief explanation of why you chose this answer"
}"""


def build_question_prompt(item: dict) -> str:
    choices_text = "\n".join(item["choices"])
    return f"""Question (Difficulty: {item['tier'].upper()} | Domain: {item['domain']}):

{item['question']}

{choices_text}

Respond in JSON format with your answer, confidence (0-100), and brief reasoning."""


# ──────────────────────────────────────────────────────────────────────────
# SCORING FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────

def parse_model_response(response_text: str) -> dict:
    """Parse model JSON response, with fallback handling."""
    try:
        # Try direct JSON parse
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        # Try extracting JSON block from text
        import re
        match = re.search(r'\{[^{}]+\}', response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"answer": "X", "confidence": 50, "reasoning": "parse_error"}


def brier_score(confidence: float, is_correct: bool) -> float:
    """
    Brier Score for a single prediction.
    Lower is better. Range: 0 (perfect) to 1 (worst).
    BS = (confidence/100 - outcome)^2
    """
    prob = confidence / 100.0
    outcome = 1.0 if is_correct else 0.0
    return (prob - outcome) ** 2


def expected_calibration_error(results: list, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).
    Measures how well confidence aligns with actual accuracy.
    Lower ECE = better calibrated.
    """
    bins = [[] for _ in range(n_bins)]
    for r in results:
        bin_idx = min(int(r["confidence"] / 10), n_bins - 1)
        bins[bin_idx].append(r)

    ece = 0.0
    total = len(results)
    for b in bins:
        if not b:
            continue
        avg_conf = sum(x["confidence"] for x in b) / len(b) / 100.0
        avg_acc = sum(1 for x in b if x["is_correct"]) / len(b)
        ece += (len(b) / total) * abs(avg_conf - avg_acc)
    return ece


def overconfidence_rate(results: list, threshold: int = 80) -> float:
    """
    Fraction of wrong answers where model was highly confident (>= threshold).
    Reveals dangerous overconfidence.
    """
    wrong = [r for r in results if not r["is_correct"]]
    if not wrong:
        return 0.0
    overconfident_wrong = [r for r in wrong if r["confidence"] >= threshold]
    return len(overconfident_wrong) / len(wrong)


def underconfidence_rate(results: list, threshold: int = 40) -> float:
    """
    Fraction of correct answers where model was very uncertain (<= threshold).
    Reveals unnecessary uncertainty.
    """
    correct = [r for r in results if r["is_correct"]]
    if not correct:
        return 0.0
    underconfident_correct = [r for r in correct if r["confidence"] <= threshold]
    return len(underconfident_correct) / len(correct)


def tier_calibration_breakdown(results: list) -> dict:
    """Calibration metrics broken down by difficulty tier."""
    tiers = {"easy": [], "medium": [], "hard": []}
    for r in results:
        tiers[r["tier"]].append(r)

    breakdown = {}
    for tier, items in tiers.items():
        if not items:
            continue
        accuracy = sum(1 for x in items if x["is_correct"]) / len(items)
        avg_confidence = sum(x["confidence"] for x in items) / len(items)
        avg_brier = sum(x["brier"] for x in items) / len(items)
        breakdown[tier] = {
            "accuracy": round(accuracy, 3),
            "avg_confidence": round(avg_confidence, 1),
            "confidence_accuracy_gap": round(avg_confidence / 100 - accuracy, 3),
            "avg_brier_score": round(avg_brier, 4),
            "n": len(items)
        }
    return breakdown


def compute_metacognition_score(results: list) -> float:
    """
    Composite Metacognition Score (0-100, higher is better).
    Combines:
    - Accuracy (40%)
    - Calibration quality via ECE (30%)
    - Overconfidence penalty (20%)
    - Underconfidence penalty (10%)
    """
    accuracy = sum(1 for r in results if r["is_correct"]) / len(results)
    ece = expected_calibration_error(results)
    overconf = overconfidence_rate(results)
    underconf = underconfidence_rate(results)

    score = (
        0.40 * accuracy +
        0.30 * (1 - ece) +
        0.20 * (1 - overconf) +
        0.10 * (1 - underconf)
    ) * 100

    return round(score, 2)


# ──────────────────────────────────────────────────────────────────────────
# TASK RUNNER
# ──────────────────────────────────────────────────────────────────────────

def load_all_questions() -> list:
    """Load all 60 questions from the three difficulty JSON files."""
    base = Path(__file__).parent
    files = ["dataset.json", "dataset_medium.json", "dataset_hard.json"]
    questions = []
    for f in files:
        with open(base / f) as fp:
            questions.extend(json.load(fp))
    return questions


def run_benchmark_on_model(model_fn, questions: list = None) -> dict:
    """
    Run the full benchmark on a model.

    Args:
        model_fn: callable(system_prompt, user_prompt) -> str
        questions: list of question dicts (defaults to all 60)

    Returns:
        Full results dict with scores and per-question breakdown
    """
    if questions is None:
        questions = load_all_questions()

    results = []
    for item in questions:
        user_prompt = build_question_prompt(item)
        raw_response = model_fn(SYSTEM_PROMPT, user_prompt)
        parsed = parse_model_response(raw_response)

        answer = parsed.get("answer", "X").strip().upper()
        confidence = max(0, min(100, int(parsed.get("confidence", 50))))
        is_correct = answer == item["answer"]
        bs = brier_score(confidence, is_correct)

        results.append({
            "id": item["id"],
            "tier": item["tier"],
            "domain": item["domain"],
            "correct_answer": item["answer"],
            "model_answer": answer,
            "confidence": confidence,
            "is_correct": is_correct,
            "brier": bs,
            "reasoning": parsed.get("reasoning", "")
        })

    # Aggregate metrics
    accuracy = sum(1 for r in results if r["is_correct"]) / len(results)
    avg_brier = sum(r["brier"] for r in results) / len(results)
    ece = expected_calibration_error(results)
    overconf = overconfidence_rate(results)
    underconf = underconfidence_rate(results)
    meta_score = compute_metacognition_score(results)
    tier_breakdown = tier_calibration_breakdown(results)

    return {
        "metacognition_score": meta_score,
        "accuracy": round(accuracy, 4),
        "avg_brier_score": round(avg_brier, 4),
        "expected_calibration_error": round(ece, 4),
        "overconfidence_rate": round(overconf, 4),
        "underconfidence_rate": round(underconf, 4),
        "tier_breakdown": tier_breakdown,
        "per_question": results,
        "total_questions": len(results)
    }


# ──────────────────────────────────────────────────────────────────────────
# KAGGLE BENCHMARKS SDK INTEGRATION
# ──────────────────────────────────────────────────────────────────────────

def create_single_task(item: dict):
    """Create a single Kaggle Benchmark Task for one question."""
    if not SDK_AVAILABLE:
        raise ImportError("Install kaggle-benchmarks: pip install kaggle-benchmarks")

    def task_fn(model):
        response = model.generate(
            system=SYSTEM_PROMPT,
            user=build_question_prompt(item)
        )
        parsed = parse_model_response(response)
        answer = parsed.get("answer", "X").strip().upper()
        confidence = max(0, min(100, int(parsed.get("confidence", 50))))
        is_correct = answer == item["answer"]
        bs = brier_score(confidence, is_correct)

        return TaskResult(
            score=1.0 - bs,  # Higher is better (1 = perfect)
            metadata={
                "correct": is_correct,
                "model_answer": answer,
                "correct_answer": item["answer"],
                "confidence": confidence,
                "brier_score": bs,
                "tier": item["tier"],
                "domain": item["domain"]
            }
        )

    return Task(
        name=f"q{item['id']}_{item['tier']}_{item['domain']}",
        description=f"[{item['tier'].upper()}] {item['question'][:80]}...",
        fn=task_fn
    )


def build_kaggle_benchmark():
    """Build and register the full Kaggle Benchmark."""
    if not SDK_AVAILABLE:
        raise ImportError("Install kaggle-benchmarks: pip install kaggle-benchmarks")

    questions = load_all_questions()
    tasks = [create_single_task(q) for q in questions]

    benchmark = Benchmark(
        name="ConfidenceCalibration-Bench",
        description=(
            "Tests metacognitive calibration: do frontier LLMs know what they know? "
            "Models answer 60 questions across easy/medium/hard tiers and rate their "
            "own confidence. Scored on Brier Score, ECE, and overconfidence rate."
        ),
        tasks=tasks,
        tags=["metacognition", "calibration", "AGI", "cognitive-abilities"]
    )
    return benchmark


# ──────────────────────────────────────────────────────────────────────────
# DEMO / LOCAL TEST
# ──────────────────────────────────────────────────────────────────────────

def demo_random_baseline():
    """Simulate a random-answer model to verify scoring pipeline."""
    import random

    def random_model(system, user):
        answer = random.choice(["A", "B", "C", "D"])
        confidence = random.randint(20, 95)
        return json.dumps({"answer": answer, "confidence": confidence, "reasoning": "random"})

    questions = load_all_questions()
    results = run_benchmark_on_model(random_model, questions)

    print("=" * 60)
    print("ConfidenceCalibration-Bench — Random Baseline Demo")
    print("=" * 60)
    print(f"Metacognition Score : {results['metacognition_score']}/100")
    print(f"Accuracy            : {results['accuracy']*100:.1f}%")
    print(f"Avg Brier Score     : {results['avg_brier_score']:.4f} (lower=better)")
    print(f"ECE                 : {results['expected_calibration_error']:.4f} (lower=better)")
    print(f"Overconfidence Rate : {results['overconfidence_rate']*100:.1f}%")
    print(f"Underconfidence Rate: {results['underconfidence_rate']*100:.1f}%")
    print("\nTier Breakdown:")
    for tier, stats in results["tier_breakdown"].items():
        print(f"  {tier.upper():8s} | Acc: {stats['accuracy']*100:.0f}% | "
              f"Avg Conf: {stats['avg_confidence']:.0f}% | "
              f"Gap: {stats['confidence_accuracy_gap']*100:+.1f}%")
    print("=" * 60)
    return results


if __name__ == "__main__":
    demo_random_baseline()
