#!/usr/bin/env python3
"""
Fix download for Twi datasets that have non-standard column formats.
Handles:
  - 'conversations' column (list of dicts with 'content' key)
  - Chat message columns (user/analysis/final)
  - Audio datasets (select only text columns)
"""

import json
import sys
from pathlib import Path

# Activate venv
venv_python = Path(__file__).parent.parent / ".venv" / "bin" / "python"

def main():
    from datasets import load_dataset

    data_dir = Path(__file__).parent.parent / "data" / "twi"
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. GooAQ Twi 2M — 'conversations' column ──
    out = data_dir / "gooaq_twi_2m.txt"
    if not out.exists() or out.stat().st_size == 0:
        print("↓ GooAQ Twi 2M (conversations format)...")
        ds = load_dataset("ghananlpcommunity/gooaq-twi-2m", split="train")
        ds = ds.select_columns(["conversations"])
        count = 0
        with open(out, "w", encoding="utf-8") as f:
            for row in ds:
                convos = row.get("conversations", [])
                if isinstance(convos, list):
                    for msg in convos:
                        if isinstance(msg, dict):
                            text = msg.get("value", msg.get("content", "")).strip()
                        elif isinstance(msg, str):
                            text = msg.strip()
                        else:
                            continue
                        if text and len(text) > 5:
                            f.write(text + "\n")
                            count += 1
        print(f"  ✓ gooaq_twi_2m.txt: {count:,} lines")
    else:
        print(f"  ✓ gooaq_twi_2m.txt: already exists, skipping")

    # ── 2. Code 170k Twi — 'conversations' column ──
    out = data_dir / "code_170k_twi.txt"
    if not out.exists() or out.stat().st_size == 0:
        print("↓ Code 170k Twi (conversations format)...")
        ds = load_dataset("ghananlpcommunity/Code-170k-twi", split="train")
        ds = ds.select_columns(["conversations"])
        count = 0
        with open(out, "w", encoding="utf-8") as f:
            for row in ds:
                convos = row.get("conversations", [])
                if isinstance(convos, list):
                    for msg in convos:
                        if isinstance(msg, dict):
                            text = msg.get("value", msg.get("content", "")).strip()
                        elif isinstance(msg, str):
                            text = msg.strip()
                        else:
                            continue
                        if text and len(text) > 5:
                            f.write(text + "\n")
                            count += 1
        print(f"  ✓ code_170k_twi.txt: {count:,} lines")
    else:
        print(f"  ✓ code_170k_twi.txt: already exists, skipping")

    # ── 3. Twi Reasoning 1k — user/analysis/final columns ──
    out = data_dir / "reasoning_1k.txt"
    if not out.exists() or out.stat().st_size == 0:
        print("↓ Twi Reasoning 1k (multi-column format)...")
        ds = load_dataset("ghananlpcommunity/twi-llm-reasoning-dataset-1k", split="train")
        ds = ds.select_columns(["user", "analysis", "final"])
        count = 0
        with open(out, "w", encoding="utf-8") as f:
            for row in ds:
                for col in ["user", "analysis", "final"]:
                    text = str(row.get(col, "")).strip()
                    if text and len(text) > 5:
                        f.write(text + "\n")
                        count += 1
        print(f"  ✓ reasoning_1k.txt: {count:,} lines")
    else:
        print(f"  ✓ reasoning_1k.txt: already exists, skipping")

    # ── 4. Asante Twi Bible — audio dataset, select text only ──
    out = data_dir / "bible_asante.txt"
    if not out.exists() or out.stat().st_size == 0:
        print("↓ Asante Twi Bible (audio+text, extracting text only)...")
        ds = load_dataset("ghananlpcommunity/asante-twi-bible-speech-text", split="train")
        # Select only text columns (skip audio)
        text_cols = [c for c in ds.column_names if c in ["text", "sentence", "transcription"]]
        if not text_cols:
            # Try to find any string column
            row0 = ds[0]
            text_cols = [c for c in ds.column_names if isinstance(row0.get(c), str)]
        if text_cols:
            ds = ds.select_columns(text_cols)
            count = 0
            with open(out, "w", encoding="utf-8") as f:
                for row in ds:
                    for col in text_cols:
                        text = str(row.get(col, "")).strip()
                        if text and len(text) > 3:
                            f.write(text + "\n")
                            count += 1
            print(f"  ✓ bible_asante.txt: {count:,} lines (columns: {text_cols})")
        else:
            print(f"  ✗ bible_asante: no text columns found in {ds.column_names}")
    else:
        print(f"  ✓ bible_asante.txt: already exists, skipping")

    print("\nDone!")


if __name__ == "__main__":
    main()
