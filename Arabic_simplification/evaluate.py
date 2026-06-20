"""
evaluate.py
-----------
Evaluates the Arabic simplification model using ROUGE scores.
Uses character-level tokenization to handle Arabic text correctly.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"]   = "0"

import re
import pandas as pd
print("📂 Loading test set...")
test_df = pd.read_csv("data/test.csv")
print(f"   Test samples: {len(test_df)}")

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── Load model ─────────────────────────────────────────────────────────────
MODEL_PATH = "model/arabic-simplifier"
print("\n⚙️  Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
device    = "cuda" if torch.cuda.is_available() else "cpu"
model     = model.to(device)
print(f"✅ Model loaded on {device.upper()}")

WHITESPACE_HANDLER = lambda k: " ".join(k.strip().split())

# ── Bad words ──────────────────────────────────────────────────────────────
BAD_WORD_PHRASES = [
    "المعنى المبسط", "الإجراء المطلوب", "النتيجة القانونية",
    "تطبق النتيجة", "تطبق العقوبة", "المادة:",
    "المادة ", "المتهم ", "الظنين ",
    "إذا لم يتم الالتزام", "إذا لم يتم", "إذا لم",
    "الالتزام", "يتم الالتزام",
]

def build_bad_words_ids(phrases, tokenizer):
    bad_words_ids = []
    for phrase in phrases:
        ids = tokenizer.encode(phrase, add_special_tokens=False)
        if ids:
            bad_words_ids.append(ids)
    return bad_words_ids

BAD_WORDS_IDS = build_bad_words_ids(BAD_WORD_PHRASES, tokenizer)

GARBAGE_PARTICLES = {
    'و','من','في','على','أن','وقد','وأكد','أعلنت','ومجلس','وشرط',
    'الحكومة','المذكورة','لاحقا','بحل','عنهم','تطبق','القانونية',
    'الرسمي','حكومي','رسمي','رسميا','رسمية','الرسمية','خلسة',
    'قانونيا','هنا','بسنة','السنة','عمن','رأس','اكسب','طول',
    'اشهى','عربى','ريش','يعرفه','انتمائه','دوريات','مسؤول',
    'والمسؤول','والوزارات','رأيه',
}

LABEL_STARTERS = {
    'المادة','الإجراء','النتيجة','ملاحظة','تنبيه','المتهم','الظنين','البند',
}

# ── Post-processing ────────────────────────────────────────────────────────
def fix_leading_waw(text):
    text = re.sub(r'^و\s+', '', text).strip()
    text = re.sub(r'^و،\s*', '', text).strip()
    return text

def remove_boilerplate_sentences(text):
    sentences = re.split(r'[.،؟!]\s*', text)
    clean = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        words = s.split()
        if not words:
            continue
        if words[0] in LABEL_STARTERS:
            continue
        if re.match(r'^المادة\s*[\d٠-٩]*\s*:', s):
            continue
        clean.append(s)
    result = ". ".join(clean).strip()
    return result + "." if result and not result.endswith((".", "،", "؟", "!")) else result

def fix_double_lam(text):
    text = re.sub(r'\sل\s+ل', ' ل', text)
    text = re.sub(r'\sل\s+(?=[\u0600-\u06FF])', ' ل', text)
    text = re.sub(r'ل(ال\w+)', r'ل\1', text)
    return text.strip()

def fix_attached_lam(text):
    return re.sub(r'لا([يتأن]\w+)', r'لا \1', text)

def remove_garbage_suffix(text):
    sentences = re.split(r'(?<=[.،؟!])\s*', text)
    clean = []
    for sentence in sentences:
        words = sentence.strip().split()
        if not words:
            continue
        if re.match(r'^:.*', sentence.strip()):
            break
        if len(words) <= 2 and any(w in GARBAGE_PARTICLES for w in words):
            break
        garbage_count = sum(1 for w in words if w in GARBAGE_PARTICLES)
        ratio = garbage_count / len(words)
        if ratio > 0.4:
            break
        if len(words) <= 4 and ratio > 0.5:
            break
        clean.append(sentence.strip())
    result = " ".join(clean).strip()
    return result + "." if result and not result.endswith((".", "،", "؟", "!")) else result

def remove_garbage_lines(text):
    lines = text.split(".")
    clean_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.search(r'[a-zA-Z]', line):
            continue
        if line.startswith(':'):
            continue
        if re.search(r'[0-9\+\=\@\#\*]', line):
            continue
        if re.match(r'^\d+[-\.\)]\s*', line):
            continue
        if len(line) < 5:
            continue
        clean_lines.append(line)
    result = ". ".join(clean_lines).strip()
    return result + "." if result and not result.endswith(".") else result

def remove_duplicate_sentences(text):
    sentences = re.split(r'[.،؟!]\s*', text)
    seen  = set()
    clean = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        words_new = set(sentence.split())
        if not words_new:
            continue
        is_duplicate = any(
            len(words_new & set(s.split())) / len(words_new) > 0.5
            for s in seen
        )
        if not is_duplicate:
            clean.append(sentence)
            seen.add(sentence)
    result = ". ".join(clean).strip()
    return result + "." if result and not result.endswith(".") else result

def cap_output_length(result, input_text, max_ratio=2.5):
    input_words  = len(input_text.split())
    output_words = result.split()
    max_words    = int(input_words * max_ratio)
    if len(output_words) <= max_words:
        return result
    truncated   = " ".join(output_words[:max_words])
    last_period = max(
        truncated.rfind("."), truncated.rfind("،"),
        truncated.rfind("؟"), truncated.rfind("!")
    )
    if last_period > 0:
        return truncated[:last_period + 1]
    return truncated + "."

def is_garbage_output(result, input_text):
    if not result or len(result.strip()) < 5:
        return True
    words = result.split()
    if len(words) < 3:
        return True
    arabic_words = [w for w in words if re.search(r'[\u0600-\u06FF]', w)]
    if not arabic_words:
        return True
    garbage_count = sum(1 for w in words if w in GARBAGE_PARTICLES)
    if garbage_count / len(words) > 0.6:
        return True
    return False

def postprocess(result, input_text):
    result = result.replace("<extra_id_0>", "").strip()
    result = fix_leading_waw(result)
    result = remove_boilerplate_sentences(result)
    result = fix_double_lam(result)
    result = fix_attached_lam(result)
    result = remove_garbage_suffix(result)
    result = remove_garbage_lines(result)
    result = remove_duplicate_sentences(result)
    result = cap_output_length(result, input_text)
    if is_garbage_output(result, input_text):
        return ""
    return result

def simplify(text):
    cleaned  = WHITESPACE_HANDLER(text)
    prompted = "تبسيط: " + cleaned
    inputs   = tokenizer(
        prompted, return_tensors="pt",
        max_length=512, truncation=True
    ).to(device)
    input_token_len = inputs["input_ids"].shape[1]
    min_len = max(30, int(input_token_len * 0.5))
    max_len = int(input_token_len * 2.5)
    output  = model.generate(
        **inputs,
        min_length=min_len, max_new_tokens=max_len,
        num_beams=4, length_penalty=1.5,
        repetition_penalty=2.5, no_repeat_ngram_size=3,
        early_stopping=True, do_sample=False,
        bad_words_ids=BAD_WORDS_IDS,
    )
    raw    = tokenizer.decode(output[0], skip_special_tokens=True)
    result = postprocess(raw, text)
    return result

# ── Arabic-aware ROUGE implementation ─────────────────────────────────────
def tokenize_arabic(text):
    """Split Arabic text into word tokens, handling Arabic punctuation."""
    text = re.sub(r'[،؟!.,:;()\[\]{}"\']', ' ', text)
    tokens = [t for t in text.split() if t.strip()]
    return tokens

def get_ngrams(tokens, n):
    """Get all n-grams from a token list."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def rouge_n(reference, hypothesis, n):
    """Compute ROUGE-N F1 score."""
    ref_tokens  = tokenize_arabic(reference)
    hyp_tokens  = tokenize_arabic(hypothesis)
    ref_ngrams  = get_ngrams(ref_tokens, n)
    hyp_ngrams  = get_ngrams(hyp_tokens, n)
    if not ref_ngrams or not hyp_ngrams:
        return 0.0
    ref_count = {}
    for ng in ref_ngrams:
        ref_count[ng] = ref_count.get(ng, 0) + 1
    hyp_count = {}
    for ng in hyp_ngrams:
        hyp_count[ng] = hyp_count.get(ng, 0) + 1
    overlap = sum(min(hyp_count.get(ng, 0), ref_count.get(ng, 0)) for ng in hyp_count)
    precision = overlap / len(hyp_ngrams) if hyp_ngrams else 0.0
    recall    = overlap / len(ref_ngrams)  if ref_ngrams  else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def lcs_length(x, y):
    """Compute longest common subsequence length."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    # Space-efficient LCS
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(curr[j-1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]

def rouge_l(reference, hypothesis):
    """Compute ROUGE-L F1 score."""
    ref_tokens = tokenize_arabic(reference)
    hyp_tokens = tokenize_arabic(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs   = lcs_length(ref_tokens, hyp_tokens)
    prec  = lcs / len(hyp_tokens) if hyp_tokens else 0.0
    rec   = lcs / len(ref_tokens)  if ref_tokens  else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

# ── Run evaluation ─────────────────────────────────────────────────────────
print(f"\n🔍 Evaluating on {len(test_df)} test samples...")
print("   This may take a while.\n")

results = []
r1_scores, r2_scores, rL_scores = [], [], []
fallback_count = 0

for i, row in test_df.iterrows():
    reference  = WHITESPACE_HANDLER(str(row["summary"]))
    hypothesis = simplify(str(row["text"]))

    if not hypothesis:
        fallback_count += 1
        hypothesis = ""

    r1 = rouge_n(reference, hypothesis, 1)
    r2 = rouge_n(reference, hypothesis, 2)
    rL = rouge_l(reference, hypothesis)

    r1_scores.append(r1)
    r2_scores.append(r2)
    rL_scores.append(rL)

    results.append({
        "index":      i,
        "input":      str(row["text"])[:100],
        "reference":  reference[:100],
        "hypothesis": hypothesis[:100],
        "rouge1":     round(r1, 4),
        "rouge2":     round(r2, 4),
        "rougeL":     round(rL, 4),
        "fallback":   hypothesis == ""
    })

    if len(results) % 50 == 0:
        print(f"   Progress: {len(results)}/{len(test_df)} done...")

# ── Print results ──────────────────────────────────────────────────────────
avg_r1 = sum(r1_scores) / len(r1_scores)
avg_r2 = sum(r2_scores) / len(r2_scores)
avg_rL = sum(rL_scores) / len(rL_scores)

print(f"\n{'='*50}")
print(f"📊 EVALUATION RESULTS")
print(f"{'='*50}")
print(f"   Test samples:     {len(test_df)}")
print(f"   Fallback outputs: {fallback_count} ({fallback_count/len(test_df)*100:.1f}%)")
print(f"")
print(f"   ROUGE-1:  {avg_r1:.4f}  ({avg_r1*100:.2f}%)")
print(f"   ROUGE-2:  {avg_r2:.4f}  ({avg_r2*100:.2f}%)")
print(f"   ROUGE-L:  {avg_rL:.4f}  ({avg_rL*100:.2f}%)")
print(f"{'='*50}")

# ── Save ───────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv("evaluation_results.csv", index=False, encoding="utf-8-sig")

summary = f"""ARABIC SIMPLIFICATION MODEL — EVALUATION SUMMARY
==================================================
Model:          AraBART fine-tuned
Test samples:   {len(test_df)}
Fallback count: {fallback_count} ({fallback_count/len(test_df)*100:.1f}%)

ROUGE SCORES:
  ROUGE-1:  {avg_r1:.4f}  ({avg_r1*100:.2f}%)
  ROUGE-2:  {avg_r2:.4f}  ({avg_r2*100:.2f}%)
  ROUGE-L:  {avg_rL:.4f}  ({avg_rL*100:.2f}%)
==================================================
"""

with open("evaluation_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print(f"\n💾 Saved: evaluation_results.csv  |  evaluation_summary.txt")
print(f"✅ Done.")
input("\nPress Enter to exit...")