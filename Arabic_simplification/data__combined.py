"""
prepare_data.py
---------------
Builds the training dataset by combining two sources:

  Source A — Legal simplification pairs (original.xlsx + simplified.xlsx)
             2,000 true Arabic simplification pairs (Jordanian legal text)
             Output is ~136% of input length → model learns to expand & clarify

  Source B — Filtered summarization pairs (summarizdataset.xlsx)
             Keep only pairs where output >= 60% of input length
             Removes heavy-compression examples that teach wrong behaviour
             ~2,307 pairs survive the filter

Combined: ~4,300 pairs, all pointing in the correct simplification direction.

Output files written to:  data/train.csv  (80%)
                          data/val.csv    (10%)
                          data/test.csv   (10%)

Each file has two columns:  text     (complex Arabic input)
                            summary  (simplified Arabic output)
"""

import os
import random

# ── Must import pandas BEFORE torch to avoid Windows hang ──────────────────
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Config ─────────────────────────────────────────────────────────────────
LEGAL_ORIGINAL_PATH   = "data/raw/original.xlsx"
LEGAL_SIMPLIFIED_PATH = "data/raw/simplified.xlsx"
SUMMARIZ_PATH         = "data/raw/summarizdataset.xlsx"

OUTPUT_DIR            = "data"
RATIO_FILTER          = 0.60   # keep summariz pairs where output >= 60% of input
TRAIN_RATIO           = 0.80
VAL_RATIO             = 0.10
TEST_RATIO            = 0.10
RANDOM_SEED           = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(RANDOM_SEED)


# ═══════════════════════════════════════════════════════════════════════════
# SOURCE A — Legal simplification pairs
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Loading Source A — Legal simplification pairs")
print("=" * 60)

orig = pd.read_excel(LEGAL_ORIGINAL_PATH)
simp = pd.read_excel(LEGAL_SIMPLIFIED_PATH)

# Merge on the paired id so every row is guaranteed to have both sides
legal_df = pd.merge(
    orig[["id", "نص_القانون"]],
    simp[["paired_original_id", "النص_المبسط"]],
    left_on="id",
    right_on="paired_original_id",
    how="inner"
)

# Rename to unified column names used across the whole pipeline
legal_df = legal_df[["نص_القانون", "النص_المبسط"]].rename(columns={
    "نص_القانون":  "text",
    "النص_المبسط": "summary"
})

# Basic cleaning
legal_df["text"]    = legal_df["text"].str.strip()
legal_df["summary"] = legal_df["summary"].str.strip()
legal_df = legal_df.dropna(subset=["text", "summary"])
legal_df = legal_df[legal_df["text"].str.len() > 0]
legal_df = legal_df[legal_df["summary"].str.len() > 0]

# Length stats
legal_df["_ratio"] = legal_df["summary"].str.len() / legal_df["text"].str.len()
print(f"Pairs loaded:              {len(legal_df)}")
print(f"Avg input length (chars):  {legal_df['text'].str.len().mean():.0f}")
print(f"Avg output length (chars): {legal_df['summary'].str.len().mean():.0f}")
print(f"Avg output/input ratio:    {legal_df['_ratio'].mean():.1%}")
print(f"Pairs where output > input: {(legal_df['_ratio'] > 1.0).sum()} "
      f"({(legal_df['_ratio'] > 1.0).mean():.1%})")

legal_df = legal_df.drop(columns=["_ratio"])
print(f"\nSource A ready: {len(legal_df)} pairs ✓")


# ═══════════════════════════════════════════════════════════════════════════
# SOURCE B — Filtered summarization pairs
# ═══════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Loading Source B — Summarization dataset (filtered)")
print("=" * 60)

summ_raw = pd.read_excel(SUMMARIZ_PATH)

# Row 0 is a duplicate header — drop it
summ_raw = summ_raw.iloc[1:].reset_index(drop=True)
summ_raw.columns = ["text", "type", "summary"]

summ_raw["text"]    = summ_raw["text"].str.strip()
summ_raw["summary"] = summ_raw["summary"].str.strip()
summ_raw = summ_raw.dropna(subset=["text", "summary"])
summ_raw = summ_raw[summ_raw["text"].str.len() > 0]
summ_raw = summ_raw[summ_raw["summary"].str.len() > 0]

print(f"Total pairs before filter: {len(summ_raw)}")

# Ratio filter — only keep pairs where output is at least 60% of input
# This removes the heavy-compression examples that teach the wrong task
summ_raw["_ratio"] = summ_raw["summary"].str.len() / summ_raw["text"].str.len()

before = len(summ_raw)
summ_df = summ_raw[summ_raw["_ratio"] >= RATIO_FILTER].copy()
after  = len(summ_df)

print(f"Pairs removed by filter:   {before - after} "
      f"({(before - after) / before:.1%})")
print(f"Pairs surviving filter:    {after} "
      f"({after / before:.1%})")
print(f"Avg output/input ratio:    {summ_df['_ratio'].mean():.1%}")

summ_df = summ_df[["text", "summary"]].drop_duplicates()
print(f"\nSource B ready: {len(summ_df)} pairs ✓")


# ═══════════════════════════════════════════════════════════════════════════
# COMBINE
# ═══════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Combining sources")
print("=" * 60)

# Tag source so we can inspect distribution later if needed
legal_df["source"] = "legal"
summ_df["source"]  = "summariz"

combined = pd.concat([legal_df, summ_df], ignore_index=True)

# Shuffle so legal and summariz pairs are mixed across all splits
combined = combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# Drop exact duplicates across sources (unlikely but safe)
before_dedup = len(combined)
combined = combined.drop_duplicates(subset=["text", "summary"])
print(f"Duplicates removed: {before_dedup - len(combined)}")
print(f"Total combined pairs: {len(combined)}")
print(f"  Legal pairs:    {(combined['source'] == 'legal').sum()}")
print(f"  Summariz pairs: {(combined['source'] == 'summariz').sum()}")

# Final ratio stats on combined set
combined["_ratio"] = combined["summary"].str.len() / combined["text"].str.len()
print(f"\nCombined ratio distribution:")
print(f"  < 60% (should be 0):   {(combined['_ratio'] < 0.60).sum()}")
print(f"  60-100%:               {((combined['_ratio'] >= 0.60) & (combined['_ratio'] <= 1.0)).sum()}")
print(f"  > 100% (simplification): {(combined['_ratio'] > 1.0).sum()}")
print(f"  Overall avg ratio:     {combined['_ratio'].mean():.1%}")
combined = combined.drop(columns=["_ratio"])


# ═══════════════════════════════════════════════════════════════════════════
# SPLIT — train / val / test
# ═══════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Splitting into train / val / test")
print("=" * 60)

# First split off train, then split remainder into val and test equally
train_df, temp_df = train_test_split(
    combined,
    test_size=(VAL_RATIO + TEST_RATIO),
    random_state=RANDOM_SEED
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,           # half of the remainder = equal val and test
    random_state=RANDOM_SEED
)

print(f"Train: {len(train_df)} pairs  ({len(train_df)/len(combined):.0%})")
print(f"Val:   {len(val_df)} pairs  ({len(val_df)/len(combined):.0%})")
print(f"Test:  {len(test_df)} pairs  ({len(test_df)/len(combined):.0%})")


# ═══════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════
# Keep source column in a separate audit file — don't pass it to the trainer
train_df[["text", "summary"]].to_csv(f"{OUTPUT_DIR}/train.csv", index=False, encoding="utf-8-sig")
val_df[["text", "summary"]].to_csv(f"{OUTPUT_DIR}/val.csv",   index=False, encoding="utf-8-sig")
test_df[["text", "summary"]].to_csv(f"{OUTPUT_DIR}/test.csv",  index=False, encoding="utf-8-sig")

# Audit file — useful for debugging, not used in training
combined.to_csv(f"{OUTPUT_DIR}/combined_audit.csv", index=False, encoding="utf-8-sig")

print()
print("=" * 60)
print("Saved files:")
print(f"  data/train.csv          ({len(train_df)} rows)")
print(f"  data/val.csv            ({len(val_df)} rows)")
print(f"  data/test.csv           ({len(test_df)} rows)")
print(f"  data/combined_audit.csv ({len(combined)} rows, includes source column)")
print("=" * 60)
print("\nDone. Run train.py next.")