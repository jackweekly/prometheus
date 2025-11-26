"""
Self-improvement harness: validates and applies patches proposed by the model or a research agent.

Usage:
  python self_improve.py --patch-file proposed.diff --apply
  cat proposed.diff | python self_improve.py --apply

Design:
- Restrict edits to allowed roots to avoid runaway changes.
- Dry-run by default; use --apply to actually apply via `git apply`.
- Intended to be driven by a model that writes unified diffs to a file; a human can review before apply.
"""

import argparse
import os
import re
import subprocess
import sys
from typing import List

ALLOWED_DIRS = ["training", "inference", "docs", "benchmarks", "data", "README.md", "Research.txt"]


def read_patch(patch_file: str = None) -> str:
    if patch_file:
        with open(patch_file, "r", encoding="utf-8") as f:
            return f.read()
    return sys.stdin.read()


def validate_patch_targets(patch: str, allowed_dirs: List[str]) -> bool:
    """
    Ensure the patch only touches allowed directories/files.
    """
    targets = re.findall(r"^\+\+\+\s+b/(.+)$", patch, flags=re.MULTILINE)
    for t in targets:
        if t == "/dev/null":
            continue
        if not any(t.startswith(ad.rstrip("/")) for ad in allowed_dirs):
            print(f"Disallowed target in patch: {t}")
            return False
    return True


def apply_patch(patch: str, dry_run: bool = True) -> bool:
    cmd = ["git", "apply", "--stat"] if dry_run else ["git", "apply"]
    res = subprocess.run(cmd, input=patch, text=True, capture_output=True)
    if res.returncode != 0:
        print("git apply failed:", res.stderr)
        return False
    print(res.stdout)
    return True


def main():
    parser = argparse.ArgumentParser(description="Self-improvement patch applier")
    parser.add_argument("--patch-file", type=str, help="Unified diff file to apply; defaults to stdin")
    parser.add_argument("--apply", action="store_true", help="Apply patch (default: dry-run)")
    args = parser.parse_args()

    patch = read_patch(args.patch_file)
    if not patch.strip():
        print("No patch provided.")
        return

    if not validate_patch_targets(patch, ALLOWED_DIRS):
        print("Patch validation failed.")
        return

    ok = apply_patch(patch, dry_run=not args.apply)
    if ok and not args.apply:
        print("Dry run succeeded. Re-run with --apply to commit changes to the working tree.")


if __name__ == "__main__":
    main()
