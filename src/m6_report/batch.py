"""
batch.py — Batch PDF generation for M6.

Public API
----------
generate_all(skip_errors=True) -> list[str]
generate_sample(n=5) -> list[str]
"""

from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))


def generate_all(skip_errors: bool = True) -> list[str]:
    """Generate PDF audit reports for all apps that have labeled pairs."""
    from m6_report.inference import list_all_app_ids, score_app
    from m6_report.generate_pdf import generate_audit_report

    paths: list[str] = []
    app_ids = list_all_app_ids()
    print(f"Generating PDFs for {len(app_ids)} apps ...")
    for i, app_id in enumerate(app_ids, 1):
        try:
            if not score_app(app_id)["predictions"]:
                continue
            path = generate_audit_report(app_id)
            paths.append(path)
            print(f"  [{i}/{len(app_ids)}] OK  {app_id}  ({Path(path).stat().st_size//1024} KB)")
        except Exception as exc:
            if skip_errors:
                print(f"  [{i}/{len(app_ids)}] ERR {app_id}: {exc}")
            else:
                raise
    print(f"Generated {len(paths)} PDFs -> reports/audits/")
    return paths


def generate_sample(n: int = 5) -> list[str]:
    """Generate n sample PDFs spanning low / medium / high risk."""
    from m6_report.inference import list_all_app_ids, score_app
    from m6_report.generate_pdf import generate_audit_report

    print(f"Scoring apps for {n} representative samples ...")
    scored: list[tuple[float, str]] = []
    for app_id in list_all_app_ids():
        try:
            r = score_app(app_id)
            if r["predictions"]:
                scored.append((r["risk_score"], app_id))
        except Exception:
            pass

    if not scored:
        print("No scoreable apps found.")
        return []

    scored.sort(key=lambda x: x[0])
    total = len(scored)
    chosen = [scored[int(i*(total-1)/(n-1))][1] for i in range(n)] if total > n else [a for _,a in scored]

    paths: list[str] = []
    for app_id in chosen:
        try:
            path = generate_audit_report(app_id)
            paths.append(path)
            print(f"  OK  {app_id}  -> {path}")
        except Exception as exc:
            print(f"  ERR {app_id}: {exc}")
    return paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all","sample"], default="sample")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()
    (generate_all if args.mode == "all" else lambda: generate_sample(n=args.n))()
