import os
import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── 1. Load model ──────────────────────────────────────────────────────────
MODEL_PATH = "model/arabic-simplifier"

print("⚙️  Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = model.to(device)
print(f"✅ Model loaded on {device.upper()}!\n")

WHITESPACE_HANDLER = lambda k: " ".join(k.strip().split())


# ── 2. Bad words ───────────────────────────────────────────────────────────
# Physically blocked during generation — model cannot generate these at all
BAD_WORD_PHRASES = [
    "المعنى المبسط",
    "الإجراء المطلوب",
    "النتيجة القانونية",
    "تطبق النتيجة",
    "تطبق العقوبة",
    "المادة:",
    "المادة ",        # blocks "المادة يمكن..." pattern
    "المتهم ",        # blocks "المتهم كل من..."
    "الظنين ",        # blocks "الظنين لا يسمح..."
    "إذا لم يتم الالتزام",  # common boilerplate closing
    "إذا لم يتم",           # catches fragmented version too
    "الالتزام",              # catches trailing الالتزام fragments
    "إذا لم",               # catches any إذا لم variant
    "يتم الالتزام",          # catches remaining fragment
]

def build_bad_words_ids(phrases, tokenizer):
    bad_words_ids = []
    for phrase in phrases:
        ids = tokenizer.encode(phrase, add_special_tokens=False)
        if ids:
            bad_words_ids.append(ids)
    return bad_words_ids

BAD_WORDS_IDS = build_bad_words_ids(BAD_WORD_PHRASES, tokenizer)


# ── 3. Generation config ───────────────────────────────────────────────────
GENERATION_CONFIG = dict(
    num_beams            = 4,
    length_penalty       = 1.5,
    repetition_penalty   = 2.5,
    no_repeat_ngram_size = 3,
    early_stopping       = True,
    do_sample            = False,
    bad_words_ids        = BAD_WORDS_IDS,
)

# ── Garbage particles ──────────────────────────────────────────────────────
GARBAGE_PARTICLES = {
    'و', 'من', 'في', 'على', 'أن', 'وقد', 'وأكد', 'أعلنت',
    'ومجلس', 'وشرط', 'الحكومة', 'المذكورة', 'لاحقا', 'بحل',
    'عنهم', 'تطبق', 'القانونية', 'الرسمي', 'حكومي', 'رسمي',
    'رسميا', 'رسمية', 'الرسمية', 'خلسة', 'قانونيا', 'هنا',
    'بسنة', 'السنة', 'عمن', 'رأس', 'اكسب', 'طول',
    'اشهى', 'عربى', 'ريش', 'يعرفه', 'انتمائه', 'دوريات',
    'مسؤول', 'والمسؤول', 'والوزارات', 'رأيه',
}

# ── Label starters ─────────────────────────────────────────────────────────
LABEL_STARTERS = {
    'المادة', 'الإجراء', 'النتيجة', 'ملاحظة',
    'تنبيه', 'المتهم', 'الظنين', 'البند',
}


# ── 4. Post-processing pipeline ────────────────────────────────────────────

def fix_leading_waw(text: str) -> str:
    """
    Remove leading و (waw) that appears at the very start of output.
    e.g. "و يمكن لطرف..." → "يمكن لطرف..."
    This happens when the model generates a connector as the first token.
    """
    text = re.sub(r'^و\s+', '', text).strip()
    text = re.sub(r'^و،\s*', '', text).strip()
    return text


def remove_boilerplate_sentences(text: str) -> str:
    """
    Remove sentences that are structural template labels.
    Detects by structure — label word at start regardless of sentence length.
    """
    sentences = re.split(r'[.،؟!]\s*', text)
    clean = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        words = s.split()
        if not words:
            continue

        # Remove ANY sentence starting with a label word
        # (removed the length check — المادة يمكن للمواطن was slipping through)
        if words[0] in LABEL_STARTERS:
            continue

        # Remove "المادة [number]:" pattern
        if re.match(r'^المادة\s*[\d٠-٩]*\s*:', s):
            continue

        clean.append(s)

    result = ". ".join(clean).strip()
    return result + "." if result and not result.endswith((".", "،", "؟", "!")) else result


def fix_double_lam(text: str) -> str:
    """Fix double ل grammar error: يمكن ل للمواطن → يمكن للمواطن"""
    text = re.sub(r'\sل\s+ل', ' ل', text)
    text = re.sub(r'\sل\s+(?=[\u0600-\u06FF])', ' ل', text)
    # Fix لالكلمة → للكلمة (ل attached directly to ال)
    text = re.sub(r'ل(ال\w+)', r'ل\1', text)
    return text.strip()


def fix_attached_lam(text: str) -> str:
    """
    Fix لا + word that got incorrectly merged.
    e.g. "لايحرم" → "لا يحرم"
    """
    text = re.sub(r'لا([يتأن]\w+)', r'لا \1', text)
    return text


def remove_garbage_suffix(text: str) -> str:
    """Cut off trailing garbage sentences."""
    sentences = re.split(r'(?<=[.،؟!])\s*', text)
    clean = []
    for sentence in sentences:
        words = sentence.strip().split()
        if not words:
            continue
        # Remove sentences that are just a colon + word (": الحكومة", ":اشهى")
        if re.match(r'^:.*', sentence.strip()):
            break
        # Remove very short sentences (1-2 words) at end that are likely garbage
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


def remove_garbage_lines(text: str) -> str:
    """Remove lines with Latin characters, numbers, symbols, or too short."""
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


def remove_duplicate_sentences(text: str) -> str:
    """Remove sentences that overlap >50% with an already-seen sentence."""
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


def cap_output_length(result: str, input_text: str, max_ratio: float = 2.5) -> str:
    """Cap output at 2.5x input word count, cutting at nearest sentence boundary."""
    input_words  = len(input_text.split())
    output_words = result.split()
    max_words    = int(input_words * max_ratio)
    if len(output_words) <= max_words:
        return result
    truncated = " ".join(output_words[:max_words])
    last_period = max(
        truncated.rfind("."),
        truncated.rfind("،"),
        truncated.rfind("؟"),
        truncated.rfind("!")
    )
    if last_period > 0:
        return truncated[:last_period + 1]
    return truncated + "."


def is_garbage_output(result: str, input_text: str) -> bool:
    """Detect if the entire output is unusable garbage."""
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


def postprocess(result: str, input_text: str) -> str:
    """Run the full post-processing pipeline in order."""
    result = result.replace("<extra_id_0>", "").strip()
    result = fix_leading_waw(result)          # remove leading و
    result = remove_boilerplate_sentences(result)  # remove label sentences
    result = fix_double_lam(result)           # fix ل ل grammar
    result = fix_attached_lam(result)         # fix لايحرم → لا يحرم
    result = remove_garbage_suffix(result)    # cut trailing noise
    result = remove_garbage_lines(result)     # remove Latin/number lines
    result = remove_duplicate_sentences(result)  # remove repeated sentences
    result = cap_output_length(result, input_text)  # cap at 2.5x input
    if is_garbage_output(result, input_text):
        return "⚠️ لم يتمكن النموذج من تبسيط هذا النص. يرجى المحاولة بنص أطول أو أكثر وضوحاً."
    return result


# ── 5. Simplify function ───────────────────────────────────────────────────

def simplify(text: str) -> str:
    cleaned  = WHITESPACE_HANDLER(text)
    prompted = "تبسيط: " + cleaned

    inputs = tokenizer(
        prompted,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)

    input_token_len = inputs["input_ids"].shape[1]
    min_len = max(30, int(input_token_len * 0.5))
    max_len = int(input_token_len * 2.5)

    output = model.generate(
        **inputs,
        min_length     = min_len,
        max_new_tokens = max_len,
        **GENERATION_CONFIG,
    )

    raw    = tokenizer.decode(output[0], skip_special_tokens=True)
    result = postprocess(raw, text)
    return result


# ── 6. Save to HTML ────────────────────────────────────────────────────────

def save_html(pairs):
    with open("results_preview.html", "w", encoding="utf-8") as f:
        f.write("""<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <style>
        body        { font-family: Arial; font-size: 18px; direction: rtl;
                      padding: 20px; background: #f9f9f9; }
        h1          { text-align: center; color: #333; }
        .sample     { border: 1px solid #ccc; margin: 20px auto; padding: 20px;
                      border-radius: 8px; background: white; max-width: 900px; }
        .label      { font-weight: bold; color: #555; font-size: 14px; margin-top: 10px; }
        .original   { background: #fff8e1; padding: 10px; margin: 8px 0; border-radius: 4px; }
        .generated  { background: #e3f2fd; padding: 10px; margin: 8px 0; border-radius: 4px; }
        .fallback   { background: #fdecea; padding: 10px; margin: 8px 0; border-radius: 4px;
                      color: #c62828; }
        .stats      { font-size: 13px; color: #888; margin-top: 5px; }
        .ratio-good { color: green; font-weight: bold; }
        .ratio-bad  { color: red;   font-weight: bold; }
    </style>
</head>
<body>
<h1>نتائج التبسيط</h1>
""")
        for i, (original, generated) in enumerate(pairs):
            orig_words  = len(original.split())
            gen_words   = len(generated.split())
            ratio       = round(gen_words / orig_words * 100) if orig_words > 0 else 0
            ratio_class = "ratio-good" if 80 <= ratio <= 200 else "ratio-bad"
            is_fallback = generated.startswith("⚠️")
            gen_class   = "fallback" if is_fallback else "generated"
            f.write(f"""
    <div class="sample">
        <div class="label">📄 النص الأصلي ({i+1}):</div>
        <div class="original">{original}</div>
        <div class="stats">عدد الكلمات: {orig_words}</div>
        <div class="label">🤖 النص المبسّط:</div>
        <div class="{gen_class}">{generated}</div>
        <div class="stats">
            عدد الكلمات: {gen_words}
            <span class="{ratio_class}">({ratio}% من الأصل)</span>
        </div>
    </div>
""")
        f.write("</body></html>")
    print("✅ Saved to results_preview.html\n")


# ── 7. Interactive loop ────────────────────────────────────────────────────

print("=" * 60)
print("🤖 Arabic Text Simplifier")
print("=" * 60)
print("Type or paste Arabic text and press Enter.")
print("Type 'quit' to exit.\n")

pairs = []

while True:
    print("📝 Enter Arabic text (or 'quit' to exit):")
    text = input("> ").strip()

    if text.lower() == "quit":
        if pairs:
            save_html(pairs)
        print("👋 Goodbye!")
        break

    if not text:
        print("⚠️  Please enter some text.\n")
        continue

    print("\n⏳ Simplifying...")
    result = simplify(text)
    pairs.append((text, result))

    orig_words = len(text.split())
    gen_words  = len(result.split())
    ratio      = round(gen_words / orig_words * 100) if orig_words > 0 else 0

    print(f"\n🤖 Simplified Text:")
    print(result)
    print(f"\n📊 Stats: {orig_words} words → {gen_words} words ({ratio}% of original)")
    print()
    save_html(pairs)