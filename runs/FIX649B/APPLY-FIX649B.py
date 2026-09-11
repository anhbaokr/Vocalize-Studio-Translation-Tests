# -*- coding: utf-8 -*-
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT.parent / "FIX649A"
DST_DIR = ROOT
SRC = DST_DIR / "hymt15_test_XIANXIA_WEAK300.py"

if not SRC_DIR.is_dir():
    raise SystemExit(f"Missing source run: {SRC_DIR}")
if SRC.exists():
    raise SystemExit(f"Refusing to overwrite existing: {SRC}")

shutil.copytree(SRC_DIR, DST_DIR, dirs_exist_ok=True)
src = SRC.read_text(encoding="utf-8")

start = src.find("def meaning_frame_coverage_check(target, frame):")
if start < 0:
    raise SystemExit("meaning_frame_coverage_check() was not found; no source change made.")

end = src.find("\n\n\n# Compact lexical-role vocabulary", start)
if end < 0:
    raise SystemExit("Coverage-check function boundary was not found; no source change made.")

old_block = src[start:end]
new_block = '''def meaning_frame_coverage_check(target, frame):
    """Check only high-confidence, authoritative slots; ordinary semantic equivalents remain free."""
    warnings, critical = [], []
    for slot, rows in (frame or {}).items():
        # MeaningFrame contains scalar metadata (e.g. version) as well as list slots.
        # Coverage must inspect only semantic slot lists and only dictionary rows.
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            cands = row.get("targetCandidates") or []
            if not cands or row.get("confidence") != "HIGH":
                continue
            authority = str(row.get("semanticAuthority", "SOFT")).upper()
            if authority != "STRONG":
                continue
            if not contains_any_candidate(target, cands):
                msg = f"MEANING_FRAME_COVERAGE_LOSS slot={slot} span={row.get('sourceSpan')} role={row.get('role')} expected={cands}"
                critical.append(msg)
    return warnings, critical
'''

src = src[:start] + new_block + src[end:]
src = src.replace("FIX649A", "FIX649B")
SRC.write_text(src, encoding="utf-8", newline="\n")

(DST_DIR / "FIX649B-CHANGELOG.txt").write_text(
    "FIX6.4.9-B — MeaningFrame runtime type-safety patch\n\n"
    "Root cause fixed:\n"
    "- MeaningFrame contains scalar metadata key `version`, but coverage-check code\n"
    "  iterated every frame value as if it were a list of dictionaries.\n"
    "- On the first row this caused: 'str' object has no attribute 'get'.\n\n"
    "Patch:\n"
    "- meaning_frame_coverage_check() now skips non-list frame slots.\n"
    "- It also skips non-dict rows defensively.\n"
    "- No semantic policy, model, prompt, source text, or target rewriting was changed.\n\n"
    "Verification required after applying:\n"
    "- python -m py_compile hymt15_test_XIANXIA_WEAK300.py\n"
    "- git diff --check\n"
    "- run the full 300-line runtime\n",
    encoding="utf-8",
)
print(f"Created FIX649B at: {DST_DIR}")
print(f"Patched: {SRC}")
