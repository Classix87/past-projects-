"""
cleanup.py
----------
Wipes all training output to start completely fresh.

Deletes:
    results/        (checkpoints)
    model/          (saved model weights)
    logs/           (training logs)

Keeps:
    data/           (your training data)
    venv/           (your Python environment)
    *.py            (your code)
"""

import os
import shutil

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DELETE_DIRS = [
    os.path.join(PROJECT_DIR, "results"),
    os.path.join(PROJECT_DIR, "model"),
    os.path.join(PROJECT_DIR, "logs"),
]

print("=" * 60)
print("Full Cleanup — Delete All Training Output")
print("=" * 60)

# ── Show what exists and how big it is ────────────────────────────
total_size = 0
found = []

for d in DELETE_DIRS:
    if os.path.exists(d):
        size = sum(
            os.path.getsize(os.path.join(dirpath, f))
            for dirpath, _, files in os.walk(d)
            for f in files
        )
        total_size += size
        found.append((d, size))
        print(f"\n   📁 {os.path.basename(d)}/  ({size / 1024**3:.2f} GB)")
    else:
        print(f"\n   ⬜ {os.path.basename(d)}/  (not found, skipping)")

if not found:
    print("\n✅ Nothing to delete — already clean.")
    input("\nPress Enter to exit...")
    exit(0)

print(f"\n   Total space to free: {total_size / 1024**3:.2f} GB")

# ── Confirm ────────────────────────────────────────────────────────
print(f"\n⚠️  This permanently deletes all model weights and training output.")
print(f"   Your data/ folder and .py files will NOT be touched.")
confirm = input("\nType 'yes' to confirm: ").strip().lower()

if confirm != "yes":
    print("❌ Cancelled — nothing deleted.")
    input("\nPress Enter to exit...")
    exit(0)

# ── Delete and recreate empty dirs ────────────────────────────────
print("\n🗑️  Deleting...")
for d, size in found:
    try:
        shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
        print(f"   ✅ Cleared {os.path.basename(d)}/")
    except Exception as e:
        print(f"   ❌ Failed on {os.path.basename(d)}/: {e}")

print(f"\n{'=' * 60}")
print(f"✅ Done — freed approx {total_size / 1024**3:.2f} GB")
print(f"   Ready for a fresh training run.")
print(f"{'=' * 60}")

input("\nPress Enter to exit...")


"""
Remove-Item -Recurse -Force results
Remove-Item -Recurse -Force model

"""
