# ConfidenceCalibration-Bench

**Metacognitive Calibration Benchmark for Frontier LLMs**
Kaggle — Measuring Progress Toward AGI | Metacognition Track

---

## What This Benchmark Tests

Do frontier LLMs know what they know?

This benchmark measures **confidence calibration** — whether a model's stated certainty matches its actual accuracy. A model claiming 90% confidence should be correct ~90% of the time. When it isn't, that's a measurable metacognitive failure.

## Dataset

- 560 total questions (60 hand-authored + 500 LLM-generated & verified)
- 3 difficulty tiers: Easy / Medium / Hard
- 12 domains: CS, Math, Physics, Biology, Neuroscience, Logic, History, Economics...
- Generated via Llama 3 + verified by Mistral (61% acceptance rate, 85% confidence threshold)

## Scoring Metrics

| Metric | Description |
|--------|-------------|
| Brier Score | `(confidence/100 - outcome)²` per question |
| ECE | Expected Calibration Error across 10 confidence bins |
| Overconfidence Rate | % of wrong answers with ≥80% confidence |
| Metacognition Score | Composite: 40% accuracy + 30% calibration + 20% overconfidence + 10% underconfidence |

## Quick Start

```bash
pip install kaggle-benchmarks
python benchmark.py
