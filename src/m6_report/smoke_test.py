"""
smoke_test.py — M6 smoke test.

Generates PDF reports for 3 apps spanning low / medium / high risk.
Verifies each PDF is >5KB and has >=3 pages.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))


def _count_pages(pdf_path: str) -> int:
    """Count PDF pages using pypdf (lightweight)."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        return len(reader.pages)
    except ImportError:
        pass
    # Fallback: count '%%Page:' markers via string scan
    with open(pdf_path, "rb") as f:
        content = f.read()
    return content.count(b"/Type /Page")


def run_smoke_test() -> list[str]:
    """
    Run smoke test on 3 apps at low / medium / high risk.

    Returns list of generated PDF paths.
    """
    from m6_report.inference import list_all_app_ids, score_app
    from m6_report.generate_pdf import generate_audit_report

    t0 = time.time()
    print("=" * 60)
    print("M6 Smoke Test — PolicyGraphAudit-RT")
    print("=" * 60)

    # Score all apps and bucket into low / medium / high
    print("Scoring apps ...")
    scored: list[tuple[float, str]] = []
    for app_id in list_all_app_ids():
        try:
            r = score_app(app_id)
            if r["predictions"]:
                scored.append((r["risk_score"], app_id))
        except Exception:
            pass

    scored.sort(key=lambda x: x[0])
    total = len(scored)
    print(f"Found {total} apps with labeled predictions.")

    if total == 0:
        print("FAIL: no scoreable apps found.")
        return []

    # Pick one from each tertile
    low_idx    = 0
    mid_idx    = total // 2
    high_idx   = total - 1

    candidates = {
        "low":    scored[low_idx][1],
        "medium": scored[mid_idx][1],
        "high":   scored[high_idx][1],
    }
    print(f"\nSelected apps:")
    for bucket, app_id in candidates.items():
        rs = scored[[a for _, a in scored].index(app_id)][0]
        print(f"  {bucket.upper():6s}  risk={rs:.3f}  {app_id}")

    generated: list[str] = []
    failures: list[str] = []

    print()
    for bucket, app_id in candidates.items():
        print(f"Generating {bucket.upper()} report: {app_id}")
        try:
            path = generate_audit_report(app_id)
            size_kb = Path(path).stat().st_size / 1024
            pages   = _count_pages(path)

            ok_size  = size_kb > 5
            ok_pages = pages >= 3

            status = "PASS" if (ok_size and ok_pages) else "FAIL"
            print(f"  {status}  path={path}")
            print(f"         size={size_kb:.1f} KB (>5KB: {ok_size}),  "
                  f"pages={pages} (>=3: {ok_pages})")

            if ok_size and ok_pages:
                generated.append(path)
            else:
                failures.append(f"{app_id}: size={size_kb:.1f}KB pages={pages}")
        except Exception as exc:
            print(f"  FAIL  {app_id}: {exc}")
            failures.append(f"{app_id}: {exc}")

    elapsed = time.time() - t0
    print()
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Generated: {len(generated)}/3 PDFs")
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  - {f}")
    print("=" * 60)

    return generated


if __name__ == "__main__":
    paths = run_smoke_test()
    if paths:
        print("\nGenerated PDFs:")
        for p in paths:
            print(f"  {p}")
