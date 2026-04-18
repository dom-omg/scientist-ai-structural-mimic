"""
Training pipeline: synthetic corpus generation + WorldModel training.

Corpus: 10K synthetic (action, context, harm_label) triples.
Label noise: 5% on safe, 5% on harmful — simulates real-world annotation noise.
Distribution: 40% harmful / 60% safe.
"""
import sys
import time
import random
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scientist_ai.world_model import WorldModel

SEED = 42
N_TRAIN = 10_000
N_THEORIES = 50
ALPHA = 0.01
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"

HARM_ACTIONS = [
    "hack into", "steal", "destroy", "attack", "poison",
    "inject malware into", "bypass security of", "forge documents for",
    "exploit vulnerability in", "manipulate", "deceive", "threaten",
    "corrupt data in", "exfiltrate data from", "kill the process of",
]
SAFE_ACTIONS = [
    "summarize", "explain", "translate", "help with", "analyze",
    "write a report about", "calculate", "search for", "recommend",
    "schedule a meeting about", "review", "read and organize",
    "create documentation for", "learn about", "describe",
]
HARM_CONTEXTS = [
    "with malicious intent", "criminal organization context", "unauthorized access attempt",
    "fraud scheme targeting", "phishing attack on", "ransomware deployment against",
    "data theft operation", "terrorist cell planning", "illegal surveillance of",
]
SAFE_CONTEXTS = [
    "educational purpose", "authorized penetration test", "personal project",
    "enterprise business context", "academic research", "home automation",
    "professional development", "customer service workflow", "creative writing",
    "technical documentation task",
]
HARM_OBJECTS = [
    "the banking system", "user personal data", "financial records",
    "critical infrastructure", "election servers", "medical records",
    "cloud storage", "authentication tokens", "private keys",
]
SAFE_OBJECTS = [
    "this document", "the dataset", "meeting notes",
    "the codebase", "the schedule", "customer feedback report",
    "the quarterly report", "the API documentation", "log files",
]


def generate_corpus(n: int, seed: int = SEED):
    rng = random.Random(seed)
    np.random.seed(seed)
    actions, contexts, labels = [], [], []

    for _ in range(n):
        if rng.random() < 0.40:
            a = rng.choice(HARM_ACTIONS) + " " + rng.choice(HARM_OBJECTS)
            c = rng.choice(HARM_CONTEXTS)
            label = 0 if rng.random() < 0.05 else 1
        else:
            a = rng.choice(SAFE_ACTIONS) + " " + rng.choice(SAFE_OBJECTS)
            c = rng.choice(SAFE_CONTEXTS)
            label = 1 if rng.random() < 0.05 else 0

        actions.append(a)
        contexts.append(c)
        labels.append(label)

    return actions, contexts, np.array(labels)


def main():
    t_start = time.time()
    print("=" * 60)
    print("Scientist AI — WorldModel Training Pipeline v0.1")
    print(f"N={N_TRAIN} | Theories={N_THEORIES} | alpha={ALPHA} | seed={SEED}")
    print("=" * 60)

    print(f"\n[1/4] Generating synthetic corpus ({N_TRAIN} examples)...")
    actions, contexts, labels = generate_corpus(N_TRAIN)
    n_harm = int(labels.sum())
    print(f"      Harmful: {n_harm} ({n_harm/N_TRAIN:.1%}) | Safe: {N_TRAIN - n_harm} ({(N_TRAIN-n_harm)/N_TRAIN:.1%})")

    print("\n[2/4] Encoding features (keyword + statistical features)...")
    model = WorldModel(n_theories=N_THEORIES, alpha=ALPHA)
    X = np.array([model.encode_input(a, c) for a, c in zip(actions, contexts)])
    print(f"      Feature matrix: {X.shape} | dtype: {X.dtype}")
    print(f"      Feature range: [{X.min():.3f}, {X.max():.3f}]")

    print(f"\n[3/4] Training ensemble of {N_THEORIES} theories (bootstrap + MLP)...")
    t_fit = time.time()
    model.fit(X, labels)
    t_fit_elapsed = time.time() - t_fit

    scores = np.array(model.train_loss_history)
    print(f"      Elapsed: {t_fit_elapsed:.1f}s")
    print(f"      Validation accuracy — mean: {scores.mean():.4f} "
          f"| min: {scores.min():.4f} | max: {scores.max():.4f}")
    print(f"      Theories with val_acc > 0.80: {(scores > 0.80).sum()}/{N_THEORIES}")

    print("\n[4/4] Saving checkpoint...")
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    save_path = CHECKPOINT_DIR / "world_model.pkl"
    model.save(str(save_path))

    total = time.time() - t_start
    print(f"\n✓ WorldModel saved → {save_path}")
    print(f"  Wall-clock: {total:.1f}s")
    print(f"\nNext: python benchmark.py")


if __name__ == "__main__":
    main()
