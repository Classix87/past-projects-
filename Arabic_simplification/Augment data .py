"""
augment_data.py
---------------
Augments the Arabic legal simplification dataset using synonym replacement.
No external dependencies beyond pandas — works with your existing venv.

Strategy:
    For each training pair (original, simplified):
        → Generate N variations by replacing 2-3 words with synonyms
        → Keep the same simplified target for each variation
        → Add all variations to the training set

Result:
    Original train set:  ~3,400 pairs
    After augmentation:  ~10,000+ pairs

Only augments the TRAIN split — val and test are left untouched
so evaluation remains a fair measurement of real generalization.

Usage:
    python augment_data.py
"""

import os
import random
import re
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Must import pandas before torch (Windows hang prevention) ──────────────
random.seed(42)

# ── Config ─────────────────────────────────────────────────────────────────
TRAIN_PATH      = "data/train.csv"
VAL_PATH        = "data/val.csv"
TEST_PATH       = "data/test.csv"
OUTPUT_DIR      = "data"
VARIATIONS      = 3       # how many augmented versions per original pair
WORDS_TO_SWAP   = 2       # how many words to replace per variation
MIN_TEXT_LEN    = 5       # skip texts shorter than this (word count)


# ══════════════════════════════════════════════════════════════════════════
# ARABIC LEGAL SYNONYM DICTIONARY
# ══════════════════════════════════════════════════════════════════════════
# Each key is a word that may appear in legal text.
# Its value is a list of synonyms that can replace it without changing meaning.
# Words are in their base form — the replacer handles finding them in context.

SYNONYMS = {
    # Rights and permissions
    "يحق":       ["يجوز", "يحل", "يسمح", "يمكن"],
    "يجوز":      ["يحق", "يمكن", "يسمح", "يحل"],
    "يسمح":      ["يجوز", "يحق", "يُتاح", "يمكن"],
    "يمكن":      ["يجوز", "يحق", "يسمح"],

    # Prohibition
    "لا يجوز":   ["لا يحق", "لا يسمح", "يُحظر", "يُمنع"],
    "يُحظر":     ["لا يجوز", "لا يسمح", "يُمنع", "لا يحق"],
    "يُمنع":     ["يُحظر", "لا يجوز", "لا يسمح"],
    "محظور":     ["ممنوع", "غير مسموح", "غير جائز"],
    "ممنوع":     ["محظور", "غير مسموح", "غير جائز"],

    # Obligation
    "يلتزم":     ["يتعهد", "يجب عليه", "يتوجب عليه", "ملزم"],
    "يتعهد":     ["يلتزم", "يوعد", "يضمن"],
    "يجب":       ["ينبغي", "يتعين", "يلزم", "من الضروري"],
    "ينبغي":     ["يجب", "يتعين", "يلزم"],
    "يتعين":     ["يجب", "ينبغي", "يلزم"],
    "ملزم":      ["مُجبر", "مضطر", "واجب عليه"],

    # Contracts and agreements
    "عقد":       ["اتفاقية", "اتفاق", "وثيقة"],
    "اتفاقية":   ["عقد", "اتفاق", "معاهدة"],
    "اتفاق":     ["عقد", "اتفاقية", "تفاهم"],
    "إبرام":     ["توقيع", "إتمام", "إنجاز"],
    "توقيع":     ["إبرام", "تثبيت", "إتمام"],

    # Legal parties
    "صاحب العمل":  ["المشغّل", "رب العمل", "الجهة المشغّلة"],
    "المشغّل":     ["صاحب العمل", "رب العمل"],
    "العامل":      ["الموظف", "الأجير", "المستخدم"],
    "الموظف":      ["العامل", "الأجير", "المستخدم"],
    "البائع":      ["المورد", "المالك", "البائع"],
    "المشتري":     ["المستهلك", "الطرف المشتري", "الشاري"],
    "المدعي":      ["المشتكي", "الطرف المدعي", "صاحب الدعوى"],
    "المدعى عليه":["المتهم", "الطرف المدعى عليه"],
    "الطرف":       ["الجهة", "الشخص", "الجانب"],

    # Legal actions
    "فصل":        ["إنهاء خدمة", "طرد", "إخراج من العمل"],
    "إنهاء":      ["إيقاف", "إلغاء", "إنهاء"],
    "مطالبة":     ["مطالبة", "طلب", "مراجعة"],
    "تعويض":      ["غرامة", "تعويض مادي", "بدل"],
    "غرامة":      ["عقوبة مالية", "تعويض", "بدل مالي"],
    "عقوبة":      ["جزاء", "عقوبة قانونية", "غرامة"],
    "جزاء":       ["عقوبة", "غرامة", "إجراء قانوني"],

    # Courts and authorities
    "المحكمة":    ["الجهة القضائية", "هيئة القضاء", "المحكمة المختصة"],
    "المختصة":    ["الصالحة", "المعنية", "ذات الاختصاص"],
    "القضاء":     ["الجهة القضائية", "المحاكم", "السلطة القضائية"],
    "القاضي":     ["المحكم", "الحاكم", "رئيس المحكمة"],

    # Time expressions
    "خلال":       ["في غضون", "في مدة", "ضمن"],
    "مدة":        ["فترة", "وقت", "أجل"],
    "فترة":       ["مدة", "حقبة", "وقت"],
    "يوماً":      ["يوم", "يوماً كاملاً"],
    "شهراً":      ["شهر", "شهراً كاملاً"],
    "سنةً":       ["عام", "سنة كاملة"],
    "سنوياً":     ["كل عام", "في العام", "سنة بسنة"],

    # Conditions
    "شرط":        ["حالة", "ضرورة", "متطلب"],
    "بشرط":       ["على أن", "بحالة", "بضرورة"],
    "في حال":     ["إذا", "عند", "في حالة"],
    "إذا":        ["في حال", "عند", "متى"],
    "متى":        ["إذا", "عند", "في حال"],

    # Property
    "ملكية":      ["حيازة", "امتلاك", "تملّك"],
    "مالك":       ["صاحب", "حائز", "متملّك"],
    "حيازة":      ["ملكية", "امتلاك", "تملّك"],

    # General legal terms
    "قانوني":     ["شرعي", "نظامي", "قانوني"],
    "شرعي":       ["قانوني", "مشروع", "نظامي"],
    "مشروع":      ["قانوني", "شرعي", "نظامي"],
    "نظامي":      ["قانوني", "شرعي", "رسمي"],
    "رسمي":       ["نظامي", "قانوني", "معتمد"],
    "معتمد":      ["رسمي", "مصادق عليه", "مقبول"],
    "صحيح":       ["سليم", "صائب", "معتمد"],
    "باطل":       ["لاغٍ", "غير صالح", "منتهٍ"],
    "لاغٍ":       ["باطل", "غير معتمد", "منتهٍ"],
    "نافذ":       ["ساري", "فعّال", "مُطبَّق"],
    "ساري":       ["نافذ", "فعّال", "مستمر"],

    # Amounts and values
    "مبلغ":       ["قيمة", "مبلغ مالي", "مقدار"],
    "قيمة":       ["مبلغ", "ثمن", "مقدار"],
    "ثمن":        ["سعر", "قيمة", "مبلغ"],
    "أجر":        ["راتب", "مرتب", "مكافأة"],
    "راتب":       ["أجر", "مرتب", "مكافأة"],
    "مرتب":       ["راتب", "أجر", "دخل"],

    # Descriptors
    "مدفوع":      ["مسدَّد", "محوَّل", "مؤدَّى"],
    "مفصَّل":     ["موضَّح", "مبيَّن", "شامل"],
    "محدد":       ["معيَّن", "مقرَّر", "واضح"],
    "معلوم":      ["واضح", "محدد", "مبيَّن"],
    "كامل":       ["تام", "شامل", "وافٍ"],
    "مناسب":      ["ملائم", "لائق", "كافٍ"],
    "تعسفي":      ["غير مبرر", "ظالم", "غير مشروع"],

    # Connectors (replace carefully — only in legal context)
    "وفقاً":      ["استناداً", "طبقاً", "بموجب"],
    "استناداً":   ["وفقاً", "طبقاً", "بحسب"],
    "طبقاً":      ["وفقاً", "استناداً", "بموجب"],
    "بموجب":      ["وفقاً", "استناداً", "بحسب"],
    "بحسب":       ["وفقاً", "طبقاً", "استناداً"],
}


# ══════════════════════════════════════════════════════════════════════════
# AUGMENTATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def tokenize_arabic(text: str) -> list:
    """Split Arabic text into tokens, preserving punctuation separately."""
    return re.findall(r'[\u0600-\u06FF]+|[^\u0600-\u06FF\s]+|\s+', text)


def find_replaceable_positions(tokens: list) -> list:
    """Return indices of tokens that have synonyms available."""
    replaceable = []
    for i, token in enumerate(tokens):
        # Check single word
        if token.strip() in SYNONYMS:
            replaceable.append((i, i+1, token.strip()))
        # Check two-word phrases (e.g. "لا يجوز", "صاحب العمل")
        if i < len(tokens) - 2:
            phrase = token.strip() + " " + tokens[i+2].strip() if tokens[i+1] == ' ' else ""
            if phrase and phrase in SYNONYMS:
                replaceable.append((i, i+3, phrase))
    return replaceable


def augment_text(text: str, n_words: int = 2, seed: int = 0) -> str:
    """
    Generate one augmented version of the text by replacing
    n_words words/phrases with synonyms.
    Returns the augmented text, or the original if no replacements found.
    """
    random.seed(seed)
    tokens = tokenize_arabic(text)
    replaceable = find_replaceable_positions(tokens)

    if not replaceable:
        return None  # nothing to replace

    # Shuffle and pick up to n_words positions to replace
    random.shuffle(replaceable)
    chosen = replaceable[:n_words]

    # Sort by position descending so replacements don't shift indices
    chosen.sort(key=lambda x: x[0], reverse=True)

    for start, end, phrase in chosen:
        synonyms = SYNONYMS.get(phrase, [])
        if not synonyms:
            continue
        replacement = random.choice(synonyms)
        # Replace tokens at [start:end] with replacement
        tokens[start:end] = [replacement]

    return "".join(tokens).strip()


def generate_variations(text: str, n_variations: int, n_words: int) -> list:
    """
    Generate n_variations augmented versions of the text.
    Uses different random seeds for each variation.
    Returns list of unique augmented texts (may be fewer than n_variations
    if synonyms are exhausted).
    """
    variations = []
    seen = {text}  # don't include the original

    for i in range(n_variations * 3):  # try 3x to get enough unique ones
        augmented = augment_text(text, n_words=n_words, seed=i * 17 + 3)
        if augmented and augmented not in seen:
            variations.append(augmented)
            seen.add(augmented)
        if len(variations) >= n_variations:
            break

    return variations


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Arabic Legal Data Augmentation")
print("=" * 60)

# ── Load train set ─────────────────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_PATH)
print(f"\nOriginal train set: {len(train_df)} pairs")
print(f"Val and test sets will NOT be modified.\n")

# ── Augment ────────────────────────────────────────────────────────────────
augmented_rows = []
skipped        = 0
augmented      = 0

for idx, row in train_df.iterrows():
    text    = str(row["text"]).strip()
    summary = str(row["summary"]).strip()

    # Skip very short texts — not enough words to replace meaningfully
    if len(text.split()) < MIN_TEXT_LEN:
        skipped += 1
        continue

    variations = generate_variations(text, VARIATIONS, WORDS_TO_SWAP)

    for var in variations:
        augmented_rows.append({
            "text":    var,
            "summary": summary   # same simplified target
        })
        augmented += 1

    if idx % 500 == 0:
        print(f"   Processed {idx}/{len(train_df)} pairs... "
              f"({augmented} augmented so far)")

print(f"\n✅ Augmentation complete:")
print(f"   Original pairs:   {len(train_df)}")
print(f"   Skipped (short):  {skipped}")
print(f"   New pairs added:  {augmented}")

# ── Combine original + augmented ───────────────────────────────────────────
augmented_df = pd.DataFrame(augmented_rows)
combined_df  = pd.concat([train_df, augmented_df], ignore_index=True)

# Shuffle so augmented pairs are mixed in, not all at the end
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Drop any exact duplicates that slipped through
before = len(combined_df)
combined_df = combined_df.drop_duplicates(subset=["text", "summary"])
print(f"   Duplicates removed: {before - len(combined_df)}")
print(f"   Final train size:   {len(combined_df)} pairs")

# ── Save ───────────────────────────────────────────────────────────────────
combined_df.to_csv(f"{OUTPUT_DIR}/train.csv", index=False, encoding="utf-8-sig")

print(f"\n💾 Saved augmented train set to data/train.csv")
print(f"   Val and test sets unchanged.")
print(f"\n{'=' * 60}")
print(f"Summary:")
print(f"   Before augmentation: {len(train_df):,} train pairs")
print(f"   After augmentation:  {len(combined_df):,} train pairs")
print(f"   Increase:            {len(combined_df) - len(train_df):,} new pairs "
      f"({(len(combined_df)/len(train_df) - 1)*100:.0f}% more data)")
print(f"{'=' * 60}")
print(f"\nNext step: run train.py to retrain on the augmented dataset.")

input("\nPress Enter to exit...")