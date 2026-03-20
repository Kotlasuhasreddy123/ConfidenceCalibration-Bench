"""
Merge generated questions with hand-crafted ones,
deduplicate, and split into train/test files for the benchmark.
"""

import json
import random
from pathlib import Path
from collections import Counter

BASE = Path(__file__).parent.parent
GENERATED_FILE = BASE / "generated" / "verified_questions.json"
HANDCRAFTED_FILES = [
    BASE / "dataset.json",
    BASE / "dataset_medium.json",
    BASE / "dataset_hard.json"
]
OUTPUT_DIR = BASE / "final_dataset"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_all() -> list:
    questions = []

    # Load hand-crafted
    for f in HANDCRAFTED_FILES:
        if f.exists():
            with open(f) as fp:
                questions.extend(json.load(fp))
    print(f"Loaded {len(questions)} hand-crafted questions")

    # Load generated
    if GENERATED_FILE.exists():
        with open(GENERATED_FILE) as fp:
            generated = json.load(fp)
        questions.extend(generated)
        print(f"Loaded {len(generated)} generated questions")
    else:
        print("No generated questions found. Run question_generator.py first.")

    return questions


def deduplicate(questions: list, threshold: int = 12) -> list:
    """Remove near-duplicate questions based on word overlap."""
    unique = []
    for q in questions:
        words = set(q["question"].lower().split())
        is_dup = False
        for u in unique:
            u_words = set(u["question"].lower().split())
            if len(words & u_words) > threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(q)
    print(f"After deduplication: {len(unique)} questions (removed {len(questions)-len(unique)})")
    return unique


def balance_tiers(questions: list, per_tier: int = None) -> list:
    """Balance questions across tiers."""
    tiers = {"easy": [], "medium": [], "hard": []}
    for q in questions:
        tier = q.get("tier", "medium")
        if tier in tiers:
            tiers[tier].append(q)

    if per_tier is None:
        per_tier = min(len(v) for v in tiers.values())

    balanced = []
    for tier, items in tiers.items():
        random.shuffle(items)
        selected = items[:per_tier]
        balanced.extend(selected)
        print(f"  {tier}: {len(selected)} questions (available: {len(items)})")

    return balanced


def assign_ids(questions: list) -> list:
    for i, q in enumerate(questions, 1):
        q["id"] = i
    return questions


def split_and_save(questions: list, test_ratio: float = 0.2):
    """Split into benchmark (test) and reserve sets."""
    random.shuffle(questions)
    split = int(len(questions) * (1 - test_ratio))
    benchmark_set = questions[:split]
    reserve_set   = questions[split:]

    # Save by tier for benchmark
    for tier in ["easy", "medium", "hard"]:
        tier_qs = [q for q in benchmark_set if q.get("tier") == tier]
        out_file = OUTPUT_DIR / f"benchmark_{tier}.json"
        with open(out_file, "w") as f:
            json.dump(tier_qs, f, indent=2)
        print(f"Saved {len(tier_qs)} {tier} questions → {out_file.name}")

    # Save full benchmark set
    with open(OUTPUT_DIR / "benchmark_all.json", "w") as f:
        json.dump(benchmark_set, f, indent=2)

    # Save reserve (for future expansion)
    with open(OUTPUT_DIR / "reserve.json", "w") as f:
        json.dump(reserve_set, f, indent=2)

    print(f"\nBenchmark set : {len(benchmark_set)} questions")
    print(f"Reserve set   : {len(reserve_set)} questions")

    # Stats
    tier_counts  = Counter(q["tier"] for q in benchmark_set)
    domain_counts = Counter(q["domain"] for q in benchmark_set)
    print(f"Tier distribution : {dict(tier_counts)}")
    print(f"Top domains       : {domain_counts.most_common(6)}")


def run():
    print("=" * 60)
    print("Merging and splitting dataset")
    print("=" * 60)

    questions = load_all()
    questions = deduplicate(questions)
    questions = balance_tiers(questions)
    questions = assign_ids(questions)
    split_and_save(questions)

    print("\nDone. Final dataset saved to kaggle_metacognition/final_dataset/")


if __name__ == "__main__":
    run()
