#!/usr/bin/env python3
"""
Download English text datasets for sLLM training.

Sources:
  1. TinyStories (roneneldan/TinyStories) — 2M short stories
  2. Simple English Wikipedia — extracted articles

Usage:
    python scripts/download_english.py [--output data/english/] [--dry-run]
"""

import argparse
import os
import sys
import json
import bz2
import re
from pathlib import Path


def ensure_deps():
    """Install required libraries."""
    try:
        import datasets
    except ImportError:
        os.system(f"{sys.executable} -m pip install -q datasets")


def download_tinystories(output_dir: Path, dry_run: bool = False) -> int:
    """Download TinyStories dataset from HuggingFace."""
    from datasets import load_dataset

    out_file = output_dir / "tinystories" / "tinystories.txt"

    if out_file.exists():
        lines = sum(1 for _ in open(out_file))
        print(f"  ✓ TinyStories: already exists ({lines:,} lines), skipping")
        return lines

    if dry_run:
        print(f"  → TinyStories: would download roneneldan/TinyStories (~470 MB)")
        return 0

    print(f"  ↓ TinyStories: downloading roneneldan/TinyStories...")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("roneneldan/TinyStories", split="train", trust_remote_code=True)

    line_count = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for row in ds:
            text = row.get("text", "").strip()
            if text and len(text) > 10:
                f.write(text + "\n")
                line_count += 1

    print(f"  ✓ TinyStories: {line_count:,} stories → tinystories/tinystories.txt")
    return line_count


def download_simplewiki(output_dir: Path, dry_run: bool = False) -> int:
    """Download Simple English Wikipedia dump and extract articles."""
    import urllib.request

    out_file = output_dir / "simplewiki" / "simplewiki.txt"
    dump_url = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
    dump_file = output_dir / "simplewiki" / "simplewiki-dump.xml.bz2"

    if out_file.exists():
        lines = sum(1 for _ in open(out_file))
        print(f"  ✓ Simple Wikipedia: already exists ({lines:,} lines), skipping")
        return lines

    if dry_run:
        print(f"  → Simple Wikipedia: would download dump (~250 MB)")
        return 0

    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Download if not cached
    if not dump_file.exists():
        print(f"  ↓ Simple Wikipedia: downloading dump (~250 MB)...")
        urllib.request.urlretrieve(dump_url, dump_file)
        print(f"  ✓ Downloaded {dump_file.stat().st_size / (1024**2):.0f} MB")

    # Extract text from XML dump
    print(f"  ⚙ Extracting articles from XML dump...")
    line_count = 0
    in_text = False
    text_buf = []

    # Simple streaming XML extraction (no external deps)
    with bz2.open(dump_file, "rt", encoding="utf-8") as bf, \
         open(out_file, "w", encoding="utf-8") as out:

        for raw_line in bf:
            stripped = raw_line.strip()

            if "<text" in stripped:
                in_text = True
                # Extract content after <text ...>
                match = re.search(r"<text[^>]*>(.*)", stripped)
                if match:
                    text_buf.append(match.group(1))
                continue

            if "</text>" in stripped:
                in_text = False
                match = re.search(r"(.*)</text>", stripped)
                if match:
                    text_buf.append(match.group(1))
                # Process accumulated text
                article = " ".join(text_buf)
                text_buf = []

                # Clean wiki markup (basic)
                article = re.sub(r"\{\{[^}]*\}\}", "", article)  # templates
                article = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", article)  # links
                article = re.sub(r"'''?", "", article)  # bold/italic
                article = re.sub(r"&amp;", "&", article)
                article = re.sub(r"&lt;[^&]*&gt;", "", article)  # HTML tags
                article = re.sub(r"&[a-z]+;", "", article)  # HTML entities
                article = re.sub(r"<[^>]+>", "", article)  # remaining XML tags
                article = re.sub(r"==+\s*[^=]+\s*==+", "\n", article)  # headers
                article = re.sub(r"\n\s*\n+", "\n", article)  # blank lines

                # Skip redirects, stubs, and very short articles
                if article.startswith("#REDIRECT") or len(article) < 100:
                    continue

                for line in article.split("\n"):
                    line = line.strip()
                    if line and len(line) > 20 and not line.startswith(("*", "|", "!", "{")):
                        out.write(line + "\n")
                        line_count += 1
                continue

            if in_text:
                text_buf.append(stripped)

    # Clean up dump file to save space
    dump_file.unlink(missing_ok=True)

    print(f"  ✓ Simple Wikipedia: {line_count:,} lines → simplewiki/simplewiki.txt")
    return line_count


def main():
    parser = argparse.ArgumentParser(description="Download English training data")
    parser.add_argument("--output", default="data/english", help="Output directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        root = Path(__file__).parent.parent
        output_dir = root / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_deps()

    print(f"{'DRY RUN — ' if args.dry_run else ''}Downloading English datasets → {output_dir}")
    print("=" * 60)

    total = 0
    total += download_tinystories(output_dir, args.dry_run)
    total += download_simplewiki(output_dir, args.dry_run)

    print("=" * 60)
    print(f"Total: {total:,} lines of English text")


if __name__ == "__main__":
    main()
