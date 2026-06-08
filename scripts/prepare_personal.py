#!/usr/bin/env python3
"""
Extract and prepare personal code from ~/Projects/ for sLLM training.

Reads source files from symlinked project directories in data/personal/,
filters out generated/vendored code, and produces combined .txt corpora.

Usage:
    python scripts/prepare_personal.py [--input data/personal/] [--output data/personal/]
"""

import argparse
from pathlib import Path

# Directories to skip
SKIP_DIRS = {
    "node_modules", ".git", "target", "__pycache__", "dist", ".next",
    "build", ".tox", ".venv", "venv", "vendor", ".cargo", ".mypy_cache",
    "coverage", ".pytest_cache", ".turbo", ".cache", ".output",
    "backup", "migrations", "static", "public", "assets",
}

# File extensions to include (no SQL per user request)
INCLUDE_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".rs",
    ".css", ".html", ".sh", ".toml", ".yaml", ".yml",
    ".json",  # Only small config files
}

# Max file size (skip huge generated files)
MAX_FILE_SIZE = 100_000  # 100 KB


def should_skip_dir(name: str) -> bool:
    return name in SKIP_DIRS or name.startswith(".")


def should_include_file(path: Path) -> bool:
    if path.suffix not in INCLUDE_EXTENSIONS:
        return False
    if path.stat().st_size > MAX_FILE_SIZE:
        return False
    if path.suffix == ".json" and path.stat().st_size > 10_000:
        return False  # Skip large JSON files (package-lock, etc.)

    # Skip known generated files
    name = path.name.lower()
    if name in {"package-lock.json", "yarn.lock", "pnpm-lock.yaml",
                "cargo.lock", ".eslintcache", "tsconfig.tsbuildinfo"}:
        return False

    return True


def extract_project(project_dir: Path, output_dir: Path) -> dict:
    """Extract source files from a single project."""
    stats = {}

    for root_path, dirs, files in project_dir.walk():
        # Skip excluded directories
        dirs[:] = [d for d in dirs if not should_skip_dir(d)]

        for fname in files:
            fpath = root_path / fname
            if not should_include_file(fpath):
                continue

            ext = fpath.suffix
            lang = ext.lstrip(".")
            stats[lang] = stats.get(lang, 0) + 1

            # Append to per-language combined file
            combined = output_dir / f"personal_{lang}.txt"
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
                with open(combined, "a", encoding="utf-8") as out:
                    # Relative path as context
                    rel = fpath.relative_to(project_dir.resolve())
                    out.write(f"<|file|>{rel}\n{content}\n<|endfile|>\n")
            except Exception:
                pass  # Skip unreadable files

    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare personal code for training")
    parser.add_argument("--input", default="data/personal")
    parser.add_argument("--output", default="data/personal")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    input_dir = root / args.input if not Path(args.input).is_absolute() else Path(args.input)
    output_dir = root / args.output if not Path(args.output).is_absolute() else Path(args.output)

    # Clean previous combined files
    for f in output_dir.glob("personal_*.txt"):
        f.unlink()

    print(f"Extracting personal code from {input_dir}")
    print("=" * 60)

    total_stats = {}
    for project in sorted(input_dir.iterdir()):
        if not project.is_dir():
            continue

        # Resolve symlink
        real_path = project.resolve()
        if not real_path.exists():
            print(f"  ✗ {project.name}: symlink broken")
            continue

        stats = extract_project(real_path, output_dir)
        if stats:
            summary = ", ".join(f"{ext}({n})" for ext, n in sorted(stats.items(), key=lambda x: -x[1])[:3])
            total_files = sum(stats.values())
            print(f"  ✓ {project.name}: {total_files:,} files ({summary})")

            for ext, count in stats.items():
                total_stats[ext] = total_stats.get(ext, 0) + count

    print("=" * 60)
    print("Combined files created:")
    for f in sorted(output_dir.glob("personal_*.txt")):
        size_mb = f.stat().st_size / (1024 * 1024)
        lines = sum(1 for _ in open(f))
        lang = f.stem.replace("personal_", "")
        print(f"  {f.name}: {lines:,} lines, {size_mb:.1f} MB")

    total = sum(total_stats.values())
    print(f"\nTotal: {total:,} source files extracted")


if __name__ == "__main__":
    main()
