# Setup & Run Guide

## Step 1: Install Ollama

Download from https://ollama.com and install it.
Then pull the two models:

```bash
ollama pull llama3
ollama pull mistral
```

## Step 2: Install Python dependencies

```bash
pip install ollama
```

## Step 3: Run the generator

```bash
# Generate 500 verified questions (takes ~30-60 min)
python question_generator.py --count 500 --confidence 85

# For more questions (takes longer):
python question_generator.py --count 2000 --confidence 80

# Custom models:
python question_generator.py --count 500 --generator llama3 --verifier mistral
```

Progress is saved every 50 questions. If interrupted, re-running resumes from checkpoint.

## Step 4: Merge with hand-crafted questions

```bash
python merge_and_split.py
```

This creates `final_dataset/` with:
- `benchmark_easy.json`
- `benchmark_medium.json`
- `benchmark_hard.json`
- `benchmark_all.json`
- `reserve.json`

## Step 5: Run the benchmark

```bash
cd ..
python benchmark.py
```

## Expected acceptance rate

~40-60% of generated questions pass verification.
To get 500 clean questions, expect ~900-1200 generation attempts.
At ~10 seconds per attempt, that's roughly 2-3 hours.

## Tips for faster generation

- Use `--confidence 75` for faster generation (less strict)
- Run multiple terminals with different domains
- Use GPU-accelerated Ollama for 3-5x speedup
