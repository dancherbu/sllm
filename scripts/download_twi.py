#!/usr/bin/env python3
"""
Download and extract Ashanti Twi text datasets from HuggingFace.

All datasets are from the ghananlpcommunity org — free, no auth required.
Outputs plain UTF-8 .txt files (one sentence per line) into data/twi/.

Usage:
    python scripts/download_twi.py [--output data/twi/] [--dry-run]
"""

import argparse
import os
import sys
from pathlib import Path

def ensure_datasets():
    """Install datasets library if needed."""
    try:
        import datasets
        return True
    except ImportError:
        print("Installing 'datasets' library...")
        os.system(f"{sys.executable} -m pip install -q datasets")
        return True

DATASETS = [
    {
        "id": "ghananlpcommunity/twi_sentences_320k",
        "output": "sentences_320k.txt",
        "text_col": "sentence",
        "description": "320k monolingual Twi sentences",
    },
    {
        "id": "ghananlpcommunity/english-twi_sentence-pairs-4m",
        "output": "parallel_4m_twi_only.txt",
        "text_col": "twi",  # Extract Twi column only
        "description": "4M Twi-English pairs (Twi column only)",
    },
    {
        "id": "ghananlpcommunity/twi-sentiments-corpus-400k",
        "output": "sentiments_400k.txt",
        "text_col": "text",
        "description": "400k Twi sentiment sentences",
    },
    {
        "id": "ghananlpcommunity/twi-emotions-corpus-400k",
        "output": "emotions_400k.txt",
        "text_col": "text",
        "description": "400k Twi emotion sentences",
    },
    {
        "id": "ghananlpcommunity/asante-twi-bible-speech-text",
        "output": "bible_asante.txt",
        "text_col": "text",
        "description": "Asante Twi Bible transcriptions",
    },
    {
        "id": "ghananlpcommunity/gooaq-twi-2m",
        "output": "gooaq_twi_2m.txt",
        "text_col": "answer",  # QA — both question and answer are Twi
        "extra_cols": ["question"],
        "description": "2.95M Twi QA pairs (questions + answers)",
    },
    {
        "id": "ghananlpcommunity/Code-170k-twi",
        "output": "code_170k_twi.txt",
        "text_col": "answer",
        "extra_cols": ["question"],
        "description": "170k coding conversations in Twi",
    },
    {
        "id": "ghananlpcommunity/twi-llm-reasoning-dataset-1k",
        "output": "reasoning_1k.txt",
        "text_col": "output",
        "extra_cols": ["instruction"],
        "description": "Chain-of-thought reasoning in Twi",
    },
    {
        "id": "ghananlpcommunity/ghana-qa",
        "output": "ghana_qa_twi.txt",
        "text_col": "answer",
        "extra_cols": ["question"],
        "filter_col": None,  # Contains Twi, Ewe, Ga — we take all (mostly Twi)
        "description": "3.5M Ghana QA pairs",
    },
]


def download_dataset(ds_info: dict, output_dir: Path, dry_run: bool = False):
    """Download and extract text from a single dataset."""
    from datasets import load_dataset

    ds_id = ds_info["id"]
    output_file = output_dir / ds_info["output"]
    text_col = ds_info["text_col"]
    extra_cols = ds_info.get("extra_cols", [])
    desc = ds_info["description"]

    if output_file.exists():
        lines = sum(1 for _ in open(output_file))
        print(f"  ✓ {desc}: already exists ({lines:,} lines), skipping")
        return lines

    if dry_run:
        print(f"  → {desc}: would download {ds_id}")
        return 0

    print(f"  ↓ {desc}: downloading {ds_id}...")

    try:
        ds = load_dataset(ds_id, split="train")
    except Exception:
        try:
            ds = load_dataset(ds_id, split="train", trust_remote_code=True)
        except Exception as e:
            print(f"  ✗ Failed to download {ds_id}: {e}")
            return 0

    # Find the right text column
    cols = ds.column_names
    if text_col not in cols:
        # Try common alternatives
        for candidate in ["text", "sentence", "twi", "Twi", "content", "output", "answer"]:
            if candidate in cols:
                text_col = candidate
                break
        else:
            print(f"  ✗ No text column found in {ds_id}. Columns: {cols}")
            return 0

    # Select only text columns to avoid decoding audio/image features
    keep_cols = [c for c in [text_col] + extra_cols if c in cols]
    try:
        ds = ds.select_columns(keep_cols)
    except Exception:
        pass  # Older datasets lib may not have select_columns

    seen = set()
    line_count = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for row in ds:
            # Collect text from primary + extra columns
            texts = []
            for col in [text_col] + extra_cols:
                if col in row and row[col]:
                    val = str(row[col]).strip()
                    if val and val not in seen:
                        texts.append(val)
                        seen.add(val)

            for text in texts:
                # Basic cleaning
                text = text.replace("\r\n", "\n").replace("\r", "\n")
                for line in text.split("\n"):
                    line = line.strip()
                    if line and len(line) > 2:  # Skip very short lines
                        f.write(line + "\n")
                        line_count += 1

    print(f"  ✓ {desc}: {line_count:,} lines → {output_file.name}")
    return line_count


def main():
    parser = argparse.ArgumentParser(description="Download Ashanti Twi training data")
    parser.add_argument("--output", default="data/twi", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    args = parser.parse_args()

    # Resolve relative to project root
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        # Find project root (where Cargo.toml is)
        root = Path(__file__).parent.parent
        output_dir = root / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'DRY RUN — ' if args.dry_run else ''}Downloading Ashanti Twi datasets → {output_dir}")
    print(f"{'=' * 60}")

    ensure_datasets()

    total_lines = 0
    for ds_info in DATASETS:
        lines = download_dataset(ds_info, output_dir, args.dry_run)
        total_lines += lines

    print(f"{'=' * 60}")
    print(f"Total: {total_lines:,} lines of Ashanti Twi text")

    # Write manifest
    manifest = output_dir / "MANIFEST.txt"
    with open(manifest, "w") as f:
        f.write(f"Ashanti Twi Training Data\n")
        f.write(f"Total lines: {total_lines:,}\n")
        f.write(f"Files:\n")
        for ds_info in DATASETS:
            p = output_dir / ds_info["output"]
            if p.exists():
                size_mb = p.stat().st_size / (1024 * 1024)
                lines = sum(1 for _ in open(p))
                f.write(f"  {ds_info['output']}: {lines:,} lines, {size_mb:.1f} MB\n")


if __name__ == "__main__":
    main()
