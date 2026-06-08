#!/usr/bin/env python3
"""
sLLM Monitor — Training & Development Dashboard

Shows:
  - Training data status (downloaded, sizes, line counts)
  - LIVE training progress (phase, epoch, perplexity, ETA, sample generations)
  - Model stats (associations, vocab, size)
  - System resources (RAM, Disk)
  - Build health (cargo check, tests)

Usage:
    python scripts/monitor.py              # One-shot status
    python scripts/monitor.py --watch      # Live refresh every 3s
    python scripts/monitor.py --json       # Machine-readable output
    python scripts/monitor.py --full       # Include build/test status
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta


# ─── Color helpers ───────────────────────────────────────────────────────────

class C:
    """ANSI color codes."""
    RESET    = "\033[0m"
    BOLD     = "\033[1m"
    DIM      = "\033[2m"
    GREEN    = "\033[32m"
    YELLOW   = "\033[33m"
    BLUE     = "\033[34m"
    MAGENTA  = "\033[35m"
    CYAN     = "\033[36m"
    RED      = "\033[31m"
    WHITE    = "\033[97m"
    ORANGE   = "\033[38;5;208m"

    @staticmethod
    def ok(s):       return f"{C.GREEN}✓{C.RESET} {s}"
    @staticmethod
    def warn(s):     return f"{C.YELLOW}⚠{C.RESET} {s}"
    @staticmethod
    def err(s):      return f"{C.RED}✗{C.RESET} {s}"
    @staticmethod
    def info(s):     return f"{C.BLUE}ℹ{C.RESET} {s}"
    @staticmethod
    def spin(s):     return f"{C.CYAN}↻{C.RESET} {s}"
    @staticmethod
    def converged(s):return f"{C.GREEN}★{C.RESET} {s}"
    @staticmethod
    def header(s):
        line = "─" * 62
        return f"{C.BOLD}{C.MAGENTA}{line}\n  {s}\n{line}{C.RESET}"


def human_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def human_duration(secs: float) -> str:
    if secs < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        return f"{secs/60:.1f}m"
    return f"{secs/3600:.1f}h"


def bar(value: float, total: float, width: int = 28, color: str = C.CYAN) -> str:
    """Render a progress bar."""
    if total <= 0:
        pct = 0.0
    else:
        pct = min(1.0, value / total)
    filled = int(pct * width)
    empty  = width - filled
    return f"{color}{'█' * filled}{C.DIM}{'░' * empty}{C.RESET}"


def sparkline(values: list, width: int = 16) -> str:
    """Render an ASCII sparkline for a list of values (e.g. perplexity history)."""
    if not values:
        return C.DIM + "─" * width + C.RESET
    blocks = " ▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    result = []
    for v in values[-width:]:
        idx = int((v - mn) / rng * (len(blocks) - 1))
        result.append(blocks[idx])
    # Pad to width
    while len(result) < width:
        result.insert(0, " ")
    return "".join(result)


def count_lines(path: Path) -> int:
    try:
        r = subprocess.run(["wc", "-l", str(path)], capture_output=True, text=True, timeout=10)
        return int(r.stdout.split()[0])
    except Exception:
        return 0


def dir_size(path: Path) -> int:
    try:
        r = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True, timeout=30)
        return int(r.stdout.split()[0])
    except Exception:
        return 0


# ─── Data status ─────────────────────────────────────────────────────────────

def check_data_status(project_root: Path) -> dict:
    data_dir = project_root / "data"
    status = {}
    phases = {
        "twi":      {"label": "Phase 0: Ashanti Twi 🇬🇭"},
        "english":  {"label": "Phase 1: English"},
        "code":     {"label": "Phase 2: Code (Public)"},
        "personal": {"label": "Phase 3: Your Code"},
    }
    for phase_name, meta in phases.items():
        phase_dir = data_dir / phase_name
        if not phase_dir.exists():
            status[phase_name] = {"label": meta["label"], "status": "missing", "files": 0, "lines": 0, "size": 0}
            continue
        txt_files = list(phase_dir.rglob("*.txt"))
        total_lines = 0
        total_size  = 0
        details = []
        for f in txt_files:
            lines = count_lines(f)
            size  = f.stat().st_size
            total_lines += lines
            total_size  += size
            details.append({"name": f.name, "lines": lines, "size": size})
        status[phase_name] = {
            "label":   meta["label"],
            "status":  "ready" if total_lines > 0 else "empty",
            "files":   len(txt_files),
            "lines":   total_lines,
            "size":    total_size,
            "details": sorted(details, key=lambda x: -x["lines"])[:3],
        }
    return status


# ─── Training progress ───────────────────────────────────────────────────────

def load_training_progress(project_root: Path) -> dict | None:
    """Load training_progress.json if it exists."""
    progress_file = project_root / "models" / "training_progress.json"
    if not progress_file.exists():
        return None
    try:
        with open(progress_file) as f:
            return json.load(f)
    except Exception:
        return None


def is_training_running(project_root: Path) -> tuple[bool, str]:
    """Return (running, pid/cmdline)."""
    try:
        r = subprocess.run(["pgrep", "-af", "sllm-train"], capture_output=True, text=True, timeout=5)
        if r.stdout.strip():
            line = r.stdout.strip().split("\n")[0]
            return True, line
    except Exception:
        pass
    # Check systemd service
    try:
        r = subprocess.run(
            ["systemctl", "is-active", "sllm-train"],
            capture_output=True, text=True, timeout=5
        )
        if r.stdout.strip() == "active":
            return True, "systemd: sllm-train.service"
    except Exception:
        pass
    return False, ""


# ─── Model status ────────────────────────────────────────────────────────────

def check_model_status(project_root: Path) -> dict:
    models_dir = project_root / "models"
    models = []
    if models_dir.exists():
        for f in sorted(models_dir.glob("*.sllm"), key=lambda x: x.stat().st_mtime, reverse=True):
            models.append({
                "name":     f.stem,
                "size":     f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            })
    return {"models": models, "count": len(models)}


# ─── System ──────────────────────────────────────────────────────────────────

def check_system_status() -> dict:
    status = {}
    try:
        r = subprocess.run(["free", "-b"], capture_output=True, text=True, timeout=5)
        for line in r.stdout.split("\n"):
            if line.startswith("Mem:"):
                parts = line.split()
                status["ram_total"]     = int(parts[1])
                status["ram_used"]      = int(parts[2])
                status["ram_available"] = int(parts[6])
    except Exception:
        pass
    try:
        r = subprocess.run(["df", "-B1", "/home"], capture_output=True, text=True, timeout=5)
        for line in r.stdout.split("\n")[1:]:
            if line.strip():
                parts = line.split()
                status["disk_total"]     = int(parts[1])
                status["disk_used"]      = int(parts[2])
                status["disk_available"] = int(parts[3])
    except Exception:
        pass
    try:
        r = subprocess.run(["uptime"], capture_output=True, text=True, timeout=5)
        status["uptime"] = r.stdout.strip()
    except Exception:
        pass
    return status


# ─── Build status ────────────────────────────────────────────────────────────

def check_build_status(project_root: Path) -> dict:
    status = {"check": "unknown", "test": "unknown", "warnings": 0}
    try:
        r = subprocess.run(
            ["cargo", "check", "--workspace", "--message-format=short"],
            cwd=project_root, capture_output=True, text=True, timeout=120
        )
        status["check"] = "ok" if r.returncode == 0 else "error"
        status["warnings"] = r.stderr.count("warning:")
        if r.returncode != 0:
            status["errors"] = r.stderr[-500:]
    except Exception as e:
        status["check"] = f"failed: {e}"
    try:
        r = subprocess.run(
            ["cargo", "test", "--workspace", "--", "--test-threads=8"],
            cwd=project_root, capture_output=True, text=True, timeout=120
        )
        for line in r.stdout.split("\n"):
            if "test result:" in line:
                status["test"] = line.strip()
                break
        if r.returncode != 0:
            status["test"] = "FAILED"
    except Exception as e:
        status["test"] = f"failed: {e}"
    return status


# ─── Render ──────────────────────────────────────────────────────────────────

STATUS_ICONS = {
    "tokenizing":    f"{C.CYAN}⚙ Tokenizing{C.RESET}",
    "training":      f"{C.CYAN}↻ Training{C.RESET}",
    "evaluating":    f"{C.YELLOW}⚖ Evaluating{C.RESET}",
    "consolidating": f"{C.MAGENTA}✂ Consolidating{C.RESET}",
    "converged":     f"{C.GREEN}★ CONVERGED{C.RESET}",
    "failed":        f"{C.RED}✗ FAILED{C.RESET}",
}

PHASE_STATUS_ICONS = {
    "pending":    f"{C.DIM}○ pending{C.RESET}",
    "active":     f"{C.CYAN}↻ active{C.RESET}",
    "evaluating": f"{C.YELLOW}⚖ eval{C.RESET}",
    "completed":  f"{C.GREEN}✓ done{C.RESET}",
    "converged":  f"{C.GREEN}★ converged{C.RESET}",
}


def render_training_panel(progress: dict, running: bool, proc: str):
    """Render the full training progress panel."""
    status_icon = STATUS_ICONS.get(progress.get("status", ""), f"? {progress.get('status','')}")
    elapsed = human_duration(progress.get("elapsed_secs", 0))
    updated = progress.get("updated_at", "")[:19].replace("T", " ")

    print(f"  Status  : {status_icon}   Elapsed: {C.BOLD}{elapsed}{C.RESET}   Updated: {C.DIM}{updated}{C.RESET}")
    if running:
        print(f"  Process : {C.DIM}{proc[:80]}{C.RESET}")
    print()

    # Model stats
    model = progress.get("model", {})
    assoc   = model.get("total_associations", 0)
    tokens  = model.get("total_tokens_trained", 0)
    vocab   = model.get("vocab_size", 0)
    msize   = model.get("model_size_bytes", 0)
    ppl     = model.get("overall_perplexity")

    print(f"  {C.BOLD}Model:{C.RESET} {model.get('name', '?')}  │  "
          f"Vocab: {C.CYAN}{vocab:,}{C.RESET}  │  "
          f"Assoc: {C.CYAN}{assoc:,}{C.RESET}  │  "
          f"Tokens: {C.CYAN}{tokens:,}{C.RESET}")
    if msize > 0:
        print(f"  Size  : {human_size(msize)}  │  "
              f"Overall Perplexity: {f'{ppl:.2f}' if ppl else C.DIM + 'not yet' + C.RESET}")
    print()

    # Per-phase breakdown
    phases = progress.get("phases", [])
    total  = len(phases)
    current_idx = progress.get("current_phase_index", 0)

    print(f"  {C.BOLD}Curriculum ({current_idx}/{total} phases):{C.RESET}")
    for i, phase in enumerate(phases):
        pstatus = PHASE_STATUS_ICONS.get(phase.get("status", "pending"), "?")
        name    = phase.get("name", f"Phase {i}")
        files   = phase.get("files_processed", 0)
        total_f = phase.get("total_files", 0)
        epoch   = phase.get("epoch", 0)
        max_ep  = phase.get("max_epochs", 1)
        ppl_val = phase.get("latest_perplexity")
        cov_val = phase.get("latest_coverage")
        hist    = phase.get("perplexity_history", [])
        sample  = phase.get("sample_generation", "")

        if phase.get("status") == "pending":
            print(f"    {C.DIM}  {i+1}. {name} — waiting{C.RESET}")
            continue

        # Progress bar
        pb = bar(files, total_f if total_f > 0 else 1, width=20)
        pct = f"{files/total_f*100:.1f}%" if total_f > 0 else "?"
        print(f"    {pstatus}  {C.BOLD}{name}{C.RESET}")
        print(f"       {pb} {files:,}/{total_f:,} files ({pct})  Epoch {epoch}/{max_ep}")

        # Metrics
        metrics = []
        if ppl_val is not None:
            metrics.append(f"Perplexity: {C.CYAN}{ppl_val:.2f}{C.RESET}")
        if cov_val is not None:
            metrics.append(f"Coverage: {C.CYAN}{cov_val*100:.1f}%{C.RESET}")
        if hist:
            spark = sparkline(hist, width=12)
            trend = "↓" if len(hist) > 1 and hist[-1] < hist[-2] else ("↑" if len(hist) > 1 and hist[-1] > hist[-2] else "→")
            color = C.GREEN if trend == "↓" else (C.RED if trend == "↑" else C.DIM)
            metrics.append(f"Trend: {color}{trend}{C.RESET} {C.DIM}[{spark}]{C.RESET}")
        if metrics:
            print(f"       " + "   ".join(metrics))

        # Sample generation (truncated)
        if sample:
            trunc = sample[:90].replace("\n", " ")
            print(f"       {C.DIM}Gen: \"{trunc}…\"{C.RESET}")
        print()


def render_dashboard(project_root: Path, as_json: bool = False):
    data       = check_data_status(project_root)
    models     = check_model_status(project_root)
    system     = check_system_status()
    tp         = load_training_progress(project_root)
    running, proc = is_training_running(project_root)

    if as_json:
        print(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "data": data, "models": models,
            "system": system, "training": tp, "running": running,
        }, indent=2, default=str))
        return

    # Clear screen
    print("\033[2J\033[H", end="")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    width = 64
    title = f"sLLM Monitor — {now}"
    pad   = (width - len(title) - 4) // 2
    print(f"{C.BOLD}╔{'═' * (width-2)}╗{C.RESET}")
    print(f"{C.BOLD}║{' ' * pad}  {title}  {' ' * pad}║{C.RESET}")
    print(f"{C.BOLD}╚{'═' * (width-2)}╝{C.RESET}")
    print()

    # ── Training Progress ── (most important, shown first when active)
    print(C.header("TRAINING PROGRESS"))
    if tp:
        render_training_panel(tp, running, proc)
    elif running:
        print(f"  {C.spin('Training is running but no progress file yet...')}")
        print(f"  {C.DIM}{proc[:80]}{C.RESET}")
        print()
    else:
        print(f"  {C.info('No training running.')}")
        print()
        print(f"  {C.BOLD}To start autonomous training:{C.RESET}")
        print(f"  {C.DIM}  cargo build --release -p sllm-train{C.RESET}")
        print(f"  {C.CYAN}  ./target/release/sllm-train --auto --data data/ --output models/brain.sllm --name sllm-v1{C.RESET}")
        print()
        print(f"  {C.BOLD}Or as a background service:{C.RESET}")
        print(f"  {C.DIM}  sudo cp deploy/sllm-train.service /etc/systemd/system/{C.RESET}")
        print(f"  {C.CYAN}  sudo systemctl enable --now sllm-train{C.RESET}")
        print()

    # ── Training Data ──
    print(C.header("TRAINING DATA"))
    total_lines = 0
    total_size  = 0
    for phase_name, info in data.items():
        total_lines += info["lines"]
        total_size  += info["size"]
        if info["status"] == "ready":
            b = bar(info["lines"], max(total_lines, 1), width=24)
            print(f"  {C.ok(info['label'])}")
            print(f"    {b} {info['lines']:>12,} lines  {human_size(info['size']):>8}  ({info['files']} files)")
            for d in info.get("details", [])[:3]:
                print(f"    {C.DIM}  └─ {d['name']}: {d['lines']:,} lines{C.RESET}")
        elif info["status"] == "empty":
            print(f"  {C.warn(info['label'])}: directory exists but empty")
        else:
            print(f"  {C.err(info['label'])}: not downloaded yet")
    print(f"\n  {C.BOLD}Total: {total_lines:,} lines ({human_size(total_size)}){C.RESET}")
    print()

    # ── Models ──
    print(C.header("TRAINED MODELS"))
    if models["count"] == 0:
        print(f"  {C.info('No trained models yet.')}")
    else:
        for m in models["models"]:
            print(f"  {C.ok(m['name'])}  {human_size(m['size']):>10}  {C.DIM}{m['modified']}{C.RESET}")
    print()

    # ── System ──
    print(C.header("SYSTEM RESOURCES"))
    if "ram_total" in system:
        ram_pct = system["ram_used"] / system["ram_total"]
        color   = C.RED if ram_pct > 0.85 else (C.YELLOW if ram_pct > 0.70 else C.GREEN)
        b = bar(system["ram_used"], system["ram_total"], color=color)
        print(f"  RAM : {b} {human_size(system['ram_used'])} / {human_size(system['ram_total'])} ({ram_pct*100:.0f}%)")
        print(f"        {C.DIM}Available for sLLM: {human_size(system['ram_available'])}{C.RESET}")
    if "disk_total" in system:
        disk_pct = system["disk_used"] / system["disk_total"]
        color    = C.RED if disk_pct > 0.90 else (C.YELLOW if disk_pct > 0.80 else C.GREEN)
        b = bar(system["disk_used"], system["disk_total"], color=color)
        print(f"  Disk: {b} {human_size(system['disk_used'])} / {human_size(system['disk_total'])} ({disk_pct*100:.0f}%)")
        print(f"        {C.DIM}Free: {human_size(system['disk_available'])}{C.RESET}")
    if "uptime" in system:
        print(f"  {C.DIM}{system['uptime']}{C.RESET}")
    print()


def render_with_build(project_root: Path):
    render_dashboard(project_root)
    print(C.header("BUILD STATUS"))
    build = check_build_status(project_root)
    if build["check"] == "ok":
        wstr = f" ({build['warnings']} warnings)" if build["warnings"] > 0 else ""
        print(f"  {C.ok(f'cargo check{wstr}')}")
    else:
        print(f"  {C.err(f'cargo check: {build[\"check\"]}')}")
    tstr = build.get("test", "unknown")
    if "0 failed" in str(tstr) or ("passed" in str(tstr) and "FAILED" not in str(tstr)):
        print(f"  {C.ok(tstr)}")
    elif "FAILED" in str(tstr):
        print(f"  {C.err('Tests FAILED')}")
    else:
        print(f"  {C.info(tstr)}")
    print()


def main():
    parser = argparse.ArgumentParser(description="sLLM Training & Development Monitor")
    parser.add_argument("--watch",   action="store_true", help="Live refresh every 3s")
    parser.add_argument("--json",    action="store_true", help="Machine-readable JSON output")
    parser.add_argument("--full",    action="store_true", help="Include build/test status (slower)")
    parser.add_argument("--project", default=None,        help="Project root (auto-detected)")
    args = parser.parse_args()

    if args.project:
        project_root = Path(args.project)
    else:
        project_root = Path(__file__).parent.parent
        if not (project_root / "Cargo.toml").exists():
            project_root = Path.cwd()

    if not (project_root / "Cargo.toml").exists():
        print("Error: Could not find sLLM project root (no Cargo.toml)")
        sys.exit(1)

    if args.watch:
        try:
            while True:
                render_with_build(project_root) if args.full else render_dashboard(project_root)
                print(f"{C.DIM}Refreshing every 3s… (Ctrl+C to stop){C.RESET}")
                time.sleep(3)
        except KeyboardInterrupt:
            print("\nStopped.")
    elif args.full:
        render_with_build(project_root)
    else:
        render_dashboard(project_root, as_json=args.json)


if __name__ == "__main__":
    main()
