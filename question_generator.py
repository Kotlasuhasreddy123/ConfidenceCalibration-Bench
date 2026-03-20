"""
Question Generator using Ollama
Generates metacognition benchmark questions using local LLMs.
Requires: pip install ollama requests
Requires: Ollama installed with llama3 or mistral model
"""

import json
import re
import time
import random
from pathlib import Path
from typing import Optional

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Install ollama: pip install ollama")


# ── CONFIG ─────────────────────────────────────────────────────────────────

GENERATOR_MODEL = "llama3"      # Model used to generate questions
VERIFIER_MODEL  = "mistral"     # Different model used to verify answers

DOMAINS = [
    "computer science", "mathematics", "physics", "chemistry",
    "biology", "history", "economics", "logic", "statistics",
    "machine learning", "algorithms", "neuroscience"
]

TIERS = {
    "easy":   "undergraduate introductory level, well-known facts",
    "medium": "undergraduate advanced or graduate introductory level",
    "hard":   "graduate level or research-level, requires deep expertise"
}

OUTPUT_DIR = Path(__file__).parent.parent / "generated"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── GENERATION PROMPT ──────────────────────────────────────────────────────

def build_generation_prompt(domain: str, tier: str, existing_questions: list) -> str:
    existing_sample = [q["question"] for q in existing_questions[-5:]] if existing_questions else []
    avoid_text = "\n".join(f"- {q}" for q in existing_sample) if existing_sample else "None yet."

    return f"""You are an expert question writer creating benchmark questions to test AI metacognition.

Generate ONE high-quality multiple choice question with these requirements:
- Domain: {domain}
- Difficulty: {tier} ({TIERS[tier]})
- Must have exactly ONE unambiguously correct answer
- Wrong answers (distractors) must be plausible but clearly incorrect to an expert
- Question must test genuine understanding, NOT just memorization of a phrase
- Avoid questions that can be answered by pattern matching alone

AVOID generating questions similar to these (already in dataset):
{avoid_text}

Respond ONLY in this exact JSON format, nothing else:
{{
  "question": "The full question text here?",
  "choices": ["A) First option", "B) Second option", "C) Third option", "D) Fourth option"],
  "answer": "A",
  "explanation": "Clear explanation of why this answer is correct and others are wrong.",
  "domain": "{domain}",
  "tier": "{tier}"
}}"""


def build_verification_prompt(item: dict) -> str:
    choices_text = "\n".join(item["choices"])
    return f"""You are an expert fact-checker. Verify whether the following question has a correct answer key.

Question: {item["question"]}

{choices_text}

Stated correct answer: {item["answer"]}
Stated explanation: {item["explanation"]}

Analyze carefully:
1. Is the stated answer actually correct?
2. Are the other options clearly wrong?
3. Is the question unambiguous?

Respond ONLY in this JSON format:
{{
  "verified": true,
  "confidence": 95,
  "correct_answer": "A",
  "issue": "none"
}}

If there is an issue, set verified to false and describe the issue.
confidence should be 0-100 representing how certain you are the answer key is correct."""


# ── OLLAMA CALLER ──────────────────────────────────────────────────────────

def call_ollama(model: str, prompt: str, temperature: float = 0.7) -> Optional[str]:
    """Call Ollama model and return response text."""
    if not OLLAMA_AVAILABLE:
        raise ImportError("Install ollama: pip install ollama")
    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={"temperature": temperature, "num_predict": 512}
        )
        return response["response"].strip()
    except Exception as e:
        print(f"  Ollama error ({model}): {e}")
        return None


def parse_json_response(text: str) -> Optional[dict]:
    """Extract and parse JSON from model response."""
    if not text:
        return None
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


# ── QUALITY FILTERS ────────────────────────────────────────────────────────

def is_valid_question(item: dict) -> tuple[bool, str]:
    """Check if a generated question meets quality standards."""
    required = ["question", "choices", "answer", "explanation", "domain", "tier"]
    for field in required:
        if field not in item:
            return False, f"missing field: {field}"

    if len(item["choices"]) != 4:
        return False, "must have exactly 4 choices"

    if item["answer"] not in ["A", "B", "C", "D"]:
        return False, f"invalid answer key: {item['answer']}"

    if len(item["question"]) < 20:
        return False, "question too short"

    if len(item["explanation"]) < 30:
        return False, "explanation too short"

    # Check choices start with A) B) C) D)
    for i, letter in enumerate(["A", "B", "C", "D"]):
        if not item["choices"][i].startswith(f"{letter})"):
            return False, f"choice {i} doesn't start with {letter})"

    return True, "ok"


def is_duplicate(new_q: str, existing: list, threshold: int = 15) -> bool:
    """Simple duplicate check based on word overlap."""
    new_words = set(new_q.lower().split())
    for existing_item in existing:
        existing_words = set(existing_item["question"].lower().split())
        overlap = len(new_words & existing_words)
        if overlap > threshold:
            return True
    return False


# ── CORE PIPELINE ──────────────────────────────────────────────────────────

def generate_one_question(domain: str, tier: str, existing: list) -> Optional[dict]:
    """Generate a single question using the generator model."""
    prompt = build_generation_prompt(domain, tier, existing)
    raw = call_ollama(GENERATOR_MODEL, prompt, temperature=0.8)
    if not raw:
        return None
    item = parse_json_response(raw)
    if not item:
        return None
    # Normalize
    item["domain"] = domain
    item["tier"] = tier
    return item


def verify_one_question(item: dict) -> tuple[bool, int, str]:
    """
    Verify a question using the verifier model.
    Returns (is_verified, confidence, issue)
    """
    prompt = build_verification_prompt(item)
    raw = call_ollama(VERIFIER_MODEL, prompt, temperature=0.1)
    if not raw:
        return False, 0, "verifier_failed"

    result = parse_json_response(raw)
    if not result:
        return False, 0, "parse_failed"

    verified = result.get("verified", False)
    confidence = int(result.get("confidence", 0))
    issue = result.get("issue", "unknown")

    # Cross-check: verifier's answer must match generator's answer
    verifier_answer = result.get("correct_answer", "").strip().upper()
    if verifier_answer and verifier_answer != item["answer"]:
        return False, confidence, f"answer_mismatch: generator={item['answer']} verifier={verifier_answer}"

    return verified, confidence, issue


def generate_dataset(
    target_count: int = 500,
    min_verification_confidence: int = 85,
    save_every: int = 50,
    tier_distribution: dict = None
) -> list:
    """
    Main pipeline: generate and verify questions until target_count is reached.

    Args:
        target_count: How many verified questions to collect
        min_verification_confidence: Minimum verifier confidence to accept (0-100)
        save_every: Save checkpoint every N accepted questions
        tier_distribution: e.g. {"easy": 0.25, "medium": 0.40, "hard": 0.35}
    """
    if tier_distribution is None:
        tier_distribution = {"easy": 0.25, "medium": 0.40, "hard": 0.35}

    accepted = []
    rejected = []
    attempts = 0
    start_time = time.time()

    # Load existing questions to avoid duplicates
    existing_file = OUTPUT_DIR / "verified_questions.json"
    if existing_file.exists():
        with open(existing_file) as f:
            accepted = json.load(f)
        print(f"Loaded {len(accepted)} existing verified questions.")

    print(f"\n{'='*60}")
    print(f"Starting generation pipeline")
    print(f"Target: {target_count} verified questions")
    print(f"Generator: {GENERATOR_MODEL} | Verifier: {VERIFIER_MODEL}")
    print(f"Min verification confidence: {min_verification_confidence}%")
    print(f"{'='*60}\n")

    while len(accepted) < target_count:
        attempts += 1

        # Pick domain and tier based on distribution
        tier = random.choices(
            list(tier_distribution.keys()),
            weights=list(tier_distribution.values())
        )[0]
        domain = random.choice(DOMAINS)

        print(f"[{len(accepted)}/{target_count}] Attempt {attempts} | {tier} | {domain}")

        # Step 1: Generate
        item = generate_one_question(domain, tier, accepted)
        if not item:
            print(f"  ✗ Generation failed")
            rejected.append({"reason": "generation_failed", "domain": domain, "tier": tier})
            continue

        # Step 2: Structural validation
        valid, reason = is_valid_question(item)
        if not valid:
            print(f"  ✗ Invalid structure: {reason}")
            rejected.append({"reason": reason, "question": item.get("question", "")[:50]})
            continue

        # Step 3: Duplicate check
        if is_duplicate(item["question"], accepted):
            print(f"  ✗ Duplicate detected")
            rejected.append({"reason": "duplicate", "question": item["question"][:50]})
            continue

        # Step 4: Verify with second model
        verified, confidence, issue = verify_one_question(item)
        if not verified or confidence < min_verification_confidence:
            print(f"  ✗ Verification failed: {issue} (confidence: {confidence}%)")
            rejected.append({"reason": f"verification_failed: {issue}", "confidence": confidence})
            continue

        # Step 5: Accept
        item["id"] = len(accepted) + 1
        item["verification_confidence"] = confidence
        accepted.append(item)
        print(f"  ✓ Accepted! (verification confidence: {confidence}%) | Total: {len(accepted)}")

        # Save checkpoint
        if len(accepted) % save_every == 0:
            _save_checkpoint(accepted, rejected, attempts, start_time)

    # Final save
    _save_checkpoint(accepted, rejected, attempts, start_time)
    _print_summary(accepted, rejected, attempts, start_time)
    return accepted


def _save_checkpoint(accepted: list, rejected: list, attempts: int, start_time: float):
    """Save current progress to disk."""
    with open(OUTPUT_DIR / "verified_questions.json", "w") as f:
        json.dump(accepted, f, indent=2)

    stats = {
        "total_accepted": len(accepted),
        "total_rejected": len(rejected),
        "total_attempts": attempts,
        "acceptance_rate": f"{len(accepted)/max(attempts,1)*100:.1f}%",
        "elapsed_seconds": round(time.time() - start_time, 1)
    }
    with open(OUTPUT_DIR / "generation_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  💾 Checkpoint saved: {len(accepted)} questions | "
          f"Acceptance rate: {stats['acceptance_rate']}\n")


def _print_summary(accepted: list, rejected: list, attempts: int, start_time: float):
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Accepted  : {len(accepted)}")
    print(f"Rejected  : {len(rejected)}")
    print(f"Attempts  : {attempts}")
    print(f"Accept %  : {len(accepted)/max(attempts,1)*100:.1f}%")
    print(f"Time      : {elapsed/60:.1f} minutes")
    print(f"Output    : {OUTPUT_DIR / 'verified_questions.json'}")

    # Tier breakdown
    from collections import Counter
    tier_counts = Counter(q["tier"] for q in accepted)
    domain_counts = Counter(q["domain"] for q in accepted)
    print(f"\nTier breakdown: {dict(tier_counts)}")
    print(f"Top domains: {domain_counts.most_common(5)}")
    print(f"{'='*60}\n")


# ── ENTRY POINT ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate metacognition benchmark questions")
    parser.add_argument("--count",      type=int, default=500,  help="Target number of verified questions")
    parser.add_argument("--confidence", type=int, default=85,   help="Min verifier confidence (0-100)")
    parser.add_argument("--save-every", type=int, default=50,   help="Save checkpoint every N questions")
    parser.add_argument("--generator",  type=str, default="llama3",  help="Ollama model for generation")
    parser.add_argument("--verifier",   type=str, default="mistral", help="Ollama model for verification")
    args = parser.parse_args()

    GENERATOR_MODEL = args.generator
    VERIFIER_MODEL  = args.verifier

    generate_dataset(
        target_count=args.count,
        min_verification_confidence=args.confidence,
        save_every=args.save_every
    )
