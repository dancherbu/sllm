#!/usr/bin/env python3
"""
sLLM Monitor — Training & Development Dashboard

Shows:
  - Training data status (downloaded, sizes, line counts)
  - Model training progress (if running)
  - Project development status (crate health, test results)
  - Disk/memory usage

Usage:
    python scripts/monitor.py              # One-shot status report
    python scripts/monitor.py --watch      # Live refreshing (every 5s)
    python scripts/monitor.py --json       # Machine-readable output
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


# ─── Color helpers ───────────────────────────────────────────────────────────

class C:
    """ANSI color codes."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    RED = "\033[31m"
    BG_DARK = "\033[48;5;235m"

    @staticmethod
    def ok(s): return f"{C.GREEN}✓{C.RESET} {s}"
    @staticmethod
    def warn(s): return f"{C.YELLOW}⚠{C.RESET} {s}"
    @staticmethod
    def err(s): return f"{C.RED}✗{C.RESET} {s}"
    @staticmethod
    def info(s): return f"{C.BLUE}ℹ{C.RESET} {s}"
    @staticmethod
    def progress(s): return f"{C.CYAN}↻{C.RESET} {s}"
    @staticmethod
    def header(s): return f"{C.BOLD}{C.MAGENTA}{'─' * 60}\n  {s}\n{'─' * 60}{C.RESET}"


def human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def count_lines(path: Path) -> int:
    """Fast line count using wc."""
    try:
        result = subprocess.run(
            ["wc", "-l", str(path)], capture_output=True, text=True, timeout=10
        )
        return int(result.stdout.split()[0])
    except Exception:
        return 0


def dir_size(path: Path) -> int:
    """Get directory size in bytes."""
    try:
        result = subprocess.run(
            ["du", "-sb", str(path)], capture_output=True, text=True, timeout=30
        )
        return int(result.stdout.split()[0])
    except Exception:
        return 0


# ─── Data status ─────────────────────────────────────────────────────────────

def check_data_status(project_root: Path) -> dict:
    """Check status of all training data directories."""
    data_dir = project_root / "data"
    status = {}

    phases = {
        "twi": {"label": "Phase 0: Ashanti Twi 🇬🇭", "ext": ".txt"},
        "english": {"label": "Phase 1: English", "ext": ".txt"},
        "code": {"label": "Phase 2: Code (Public)", "ext": None},
        "personal": {"label": "Phase 3: Your Code", "ext": ".txt"},
    }

    for phase_name, meta in phases.items():
        phase_dir = data_dir / phase_name
        if not phase_dir.exists():
            status[phase_name] = {"label": meta["label"], "status": "missing", "files": 0, "lines": 0, "size": 0}
            continue

        # Count files and lines
        txt_files = list(phase_dir.rglob("*.txt"))
        total_lines = 0
        total_size = 0
        file_details = []

        for f in txt_files:
            lines = count_lines(f)
            size = f.stat().st_size
            total_lines += lines
            total_size += size
            file_details.append({"name": f.name, "lines": lines, "size": size})

        status[phase_name] = {
            "label": meta["label"],
            "status": "ready" if total_lines > 0 else "empty",
            "files": len(txt_files),
            "lines": total_lines,
            "size": total_size,
            "details": sorted(file_details, key=lambda x: -x["lines"])[:5],
        }

    return status


# ─── Model status ────────────────────────────────────────────────────────────

def check_model_status(project_root: Path) -> dict:
    """Check for trained models."""
    models_dir = project_root / "models"
    models = []

    if models_dir.exists():
        for f in models_dir.glob("*.sllm"):
            models.append({
                "name": f.stem,
                "size": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            })

    return {"models": models, "count": len(models)}


# ─── Build status ────────────────────────────────────────────────────────────

def check_build_status(project_root: Path) -> dict:
    """Run cargo check and cargo test."""
    status = {"check": "unknown", "test": "unknown", "warnings": 0}

    try:
        result = subprocess.run(
            ["cargo", "check", "--workspace", "--message-format=short"],
            cwd=project_root, capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            warnings = result.stderr.count("warning:")
            status["check"] = "ok"
            status["warnings"] = warnings
        else:
            status["check"] = "error"
            status["errors"] = result.stderr[-500:] if result.stderr else ""
    except Exception as e:
        status["check"] = f"failed: {e}"

    try:
        result = subprocess.run(
            ["cargo", "test", "--workspace", "--", "--test-threads=8"],
            cwd=project_root, capture_output=True, text=True, timeout=120
        )
        # Parse test results
        for line in result.stdout.split("\n"):
            if "test result:" in line:
                status["test"] = line.strip()
                break
        if result.returncode != 0:
            status["test"] = "FAILED"
    except Exception as e:
        status["test"] = f"failed: {e}"

    return status


# ─── System status ───────────────────────────────────────────────────────────

def check_system_status() -> dict:
    """Check disk and memory usage."""
    status = {}

    try:
        result = subprocess.run(["free", "-b"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.split("\n"):
            if line.startswith("Mem:"):
                parts = line.split()
                status["ram_total"] = int(parts[1])
                status["ram_used"] = int(parts[2])
                status["ram_available"] = int(parts[6])
    except Exception:
        pass

    try:
        result = subprocess.run(["df", "-B1", "/home"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.split("\n")[1:]:
            if line.strip():
                parts = line.split()
                status["disk_total"] = int(parts[1])
                status["disk_used"] = int(parts[2])
                status["disk_available"] = int(parts[3])
    except Exception:
        pass

    try:
        result = subprocess.run(["uptime"], capture_output=True, text=True, timeout=5)
        status["uptime"] = result.stdout.strip()
    except Exception:
        pass

    return status


# ─── Training status ─────────────────────────────────────────────────────────

def check_training_status(project_root: Path) -> dict:
    """Check if training is currently running."""
    status = {"running": False, "progress": None}

    # Check for checkpoint files
    checkpoints = list((project_root / "models").glob("*.checkpoint.sllm")) if (project_root / "models").exists() else []
    if checkpoints:
        latest = max(checkpoints, key=lambda f: f.stat().st_mtime)
        status["latest_checkpoint"] = {
            "name": latest.name,
            "size": latest.stat().st_size,
            "time": datetime.fromtimestamp(latest.stat().st_mtime).isoformat(),
        }

    # Check if sllm-train is running
    try:
        result = subprocess.run(
            ["pgrep", "-af", "sllm-train"], capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip():
            status["running"] = True
            status["process"] = result.stdout.strip().split("\n")[0]
    except Exception:
        pass

    return status


# ─── Render ──────────────────────────────────────────────────────────────────

def render_dashboard(project_root: Path, as_json: bool = False):
    """Render the full dashboard."""
    data = check_data_status(project_root)
    models = check_model_status(project_root)
    system = check_system_status()
    training = check_training_status(project_root)

    if as_json:
        print(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "models": models,
            "system": system,
            "training": training,
        }, indent=2, default=str))
        return

    # Clear screen for watch mode
    print("\033[2J\033[H", end="")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{C.BOLD}╔══════════════════════════════════════════════════════════╗{C.RESET}")
    print(f"{C.BOLD}║          sLLM Monitor — {now}           ║{C.RESET}")
    print(f"{C.BOLD}╚══════════════════════════════════════════════════════════╝{C.RESET}")
    print()

    # ── Training Data ──
    print(C.header("TRAINING DATA"))
    total_lines = 0
    total_size = 0
    for phase_name, info in data.items():
        total_lines += info["lines"]
        total_size += info["size"]
        if info["status"] == "ready":
            bar_len = min(30, max(1, info["lines"] // 100000))
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"  {C.ok(info['label'])}")
            print(f"    {C.DIM}{bar}{C.RESET} {info['lines']:>12,} lines  {human_size(info['size']):>8}  ({info['files']} files)")
            if info.get("details"):
                for d in info["details"][:3]:
                    print(f"    {C.DIM}  └─ {d['name']}: {d['lines']:,} lines{C.RESET}")
        elif info["status"] == "empty":
            print(f"  {C.warn(info['label'])}: directory exists but empty")
        else:
            print(f"  {C.err(info['label'])}: not downloaded yet")

    print(f"\n  {C.BOLD}Total: {total_lines:,} lines ({human_size(total_size)}){C.RESET}")

    # ── Models ──
    print()
    print(C.header("MODELS"))
    if models["count"] == 0:
        print(f"  {C.info('No trained models yet. Run: sllm train')}")
    else:
        for m in models["models"]:
            print(f"  {C.ok(m['name'])} — {human_size(m['size'])} — {m['modified']}")

    # ── Training ──
    print()
    print(C.header("TRAINING"))
    if training["running"]:
        print(f"  {C.progress('Training in progress!')}")
        print(f"    Process: {training.get('process', 'unknown')}")
    else:
        print(f"  {C.DIM}  No training running{C.RESET}")

    if training.get("latest_checkpoint"):
        cp = training["latest_checkpoint"]
        cp_name = cp["name"]
        cp_size = human_size(cp["size"])
        print(f"  {C.info(f'Latest checkpoint: {cp_name} ({cp_size})')}")

    # ── System ──
    print()
    print(C.header("SYSTEM RESOURCES"))
    if "ram_total" in system:
        ram_pct = (system["ram_used"] / system["ram_total"]) * 100
        ram_bar_used = int(ram_pct / 100 * 30)
        ram_bar = f"{'█' * ram_bar_used}{'░' * (30 - ram_bar_used)}"
        print(f"  RAM:  {ram_bar} {human_size(system['ram_used'])} / {human_size(system['ram_total'])} ({ram_pct:.0f}%)")
        print(f"        {C.DIM}Available for sLLM: {human_size(system['ram_available'])}{C.RESET}")

    if "disk_total" in system:
        disk_pct = (system["disk_used"] / system["disk_total"]) * 100
        disk_bar_used = int(disk_pct / 100 * 30)
        disk_bar = f"{'█' * disk_bar_used}{'░' * (30 - disk_bar_used)}"
        print(f"  Disk: {disk_bar} {human_size(system['disk_used'])} / {human_size(system['disk_total'])} ({disk_pct:.0f}%)")
        print(f"        {C.DIM}Free: {human_size(system['disk_available'])}{C.RESET}")

    if "uptime" in system:
        print(f"  {C.DIM}{system['uptime']}{C.RESET}")

    print()


def render_with_build(project_root: Path):
    """Render dashboard including build status (slower)."""
    render_dashboard(project_root)

    print(C.header("BUILD STATUS"))
    build = check_build_status(project_root)

    if build["check"] == "ok":
        warn_str = f" ({build['warnings']} warnings)" if build["warnings"] > 0 else ""
        print(f"  {C.ok(f'cargo check{warn_str}')}")
    else:
        check_val = build["check"]
        print(f"  {C.err(f'cargo check: {check_val}')}") 

    test_str = build.get("test", "unknown")
    if "0 failed" in str(test_str) or "passed" in str(test_str):
        print(f"  {C.ok(test_str)}")
    elif test_str == "FAILED":
        print(f"  {C.err('Tests FAILED')}")
    else:
        print(f"  {C.info(test_str)}")

    print()


def main():
    parser = argparse.ArgumentParser(description="sLLM Training & Development Monitor")
    parser.add_argument("--watch", action="store_true", help="Live refreshing mode (every 5s)")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    parser.add_argument("--full", action="store_true", help="Include build/test status (slower)")
    parser.add_argument("--project", default=None, help="Project root (auto-detected)")
    args = parser.parse_args()

    # Find project root
    if args.project:
        project_root = Path(args.project)
    else:
        # Walk up from script location
        project_root = Path(__file__).parent.parent
        if not (project_root / "Cargo.toml").exists():
            project_root = Path.cwd()

    if not (project_root / "Cargo.toml").exists():
        print(f"Error: Could not find sLLM project root (no Cargo.toml)")
        sys.exit(1)

    if args.watch:
        try:
            while True:
                if args.full:
                    render_with_build(project_root)
                else:
                    render_dashboard(project_root)
                print(f"{C.DIM}Refreshing every 5s... (Ctrl+C to stop){C.RESET}")
                time.sleep(5)
        except KeyboardInterrupt:
            print("\nStopped.")
    elif args.full:
        render_with_build(project_root)
    else:
        render_dashboard(project_root, as_json=args.json)


if __name__ == "__main__":
    main()
