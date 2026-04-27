"""
generate_pdf.py — M6 PDF audit report generator for PolicyGraphAudit-RT.

Public API
----------
generate_audit_report(appId, output_path=None) -> str
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[2]
_REPORTS_DIR = _ROOT / "reports" / "audits"
_MODEL_CARD_URL = "https://github.com/PolicyGraphAudit-RT/reports/blob/main/reports/m5_model_card.md"

# Risk thresholds and class ordering
_RISK_ORDER = ["UNDECLARED_COLLECTION", "POLICY_LABEL_MISMATCH", "OVER_DISCLOSURE", "CONSISTENT"]
_CLASS_BG = {
    "UNDECLARED_COLLECTION": "#fee2e2", "POLICY_LABEL_MISMATCH": "#fef3c7",
    "OVER_DISCLOSURE": "#dbeafe",       "CONSISTENT":            "#dcfce7",
}
_CLASS_FG = {
    "UNDECLARED_COLLECTION": "#991b1b", "POLICY_LABEL_MISMATCH": "#92400e",
    "OVER_DISCLOSURE": "#1e40af",       "CONSISTENT":            "#166534",
}


def _risk_category(score: float) -> tuple[str, str]:
    if score < 0.3:   return "LOW",    "#16a34a"
    if score < 0.6:   return "MEDIUM", "#d97706"
    return "HIGH", "#dc2626"


def generate_audit_report(appId: str, output_path: Optional[str] = None) -> str:
    """Generate a multi-page PDF compliance audit report. Returns output path."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, PageBreak,
        Table, TableStyle, HRFlowable, KeepTogether,
    )

    import sys; sys.path.insert(0, str(_ROOT / "src"))
    from m6_report.inference import score_app

    # -- Output path --
    if output_path is None:
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out = _REPORTS_DIR / f"{appId}.pdf"
    else:
        out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)

    # -- Inference --
    result    = score_app(appId)
    meta      = result["app_metadata"]
    preds     = result["predictions"]
    risk      = result["risk_score"]
    counts    = result["summary_counts"]
    rl, rc    = _risk_category(risk)
    today     = date.today().isoformat()
    app_title = meta.get("title", appId)

    sort_key = {c: i for i, c in enumerate(_RISK_ORDER)}
    preds_sorted = sorted(preds, key=lambda p: (sort_key.get(p["predicted_class"], 99), -p["confidence"]))
    high_risk = [p for p in preds_sorted if p["predicted_class"] == "UNDECLARED_COLLECTION"
                 or (p["confidence"] > 0.85 and p["predicted_class"] != "CONSISTENT")]

    # -- Colors --
    C      = colors.HexColor
    NAVY   = C("#0f172a"); WHITE = colors.white; GRAY = C("#64748b")
    DARK   = C("#1e293b"); LIGHT = C("#f8fafc"); BORDER = C("#e2e8f0")
    ACCENT = C("#2563eb")
    W = letter[0] - 1.7 * inch  # usable width

    # -- Styles --
    base = getSampleStyleSheet()
    def S(n, fn="Helvetica", fs=10, tc=None, sb=0, sa=6, ld=14, al=None, li=0, **kw):
        kw2 = dict(fontName=fn, fontSize=fs, textColor=tc or DARK, spaceBefore=sb,
                   spaceAfter=sa, leading=ld)
        if al is not None: kw2["alignment"] = al
        if li: kw2["leftIndent"] = li
        kw2.update(kw)
        return ParagraphStyle(n, parent=base["Normal"], **kw2)

    sty = {
        "h2":     S("h2",  fn="Helvetica-Bold", fs=13, sb=14, sa=5, ld=17),
        "body":   S("body",fs=10, sa=6, ld=15, al=TA_JUSTIFY),
        "bullet": S("bul", fs=10, sa=3, ld=14, li=16),
        "small":  S("sm",  fs=8,  tc=GRAY, sa=2, ld=11),
        "mono":   S("mo",  fn="Courier", fs=8, sa=2),
        "lbl":    S("lb",  fs=9,  sa=2),
        "disc":   S("di",  fs=9,  tc=GRAY, sa=5, ld=13, al=TA_JUSTIFY),
        "ft":     S("ft",  fs=7,  tc=GRAY, al=TA_CENTER),
        "ev_hdr": S("eh",  fn="Helvetica-Bold", fs=10, sb=10, sa=3),
        "ev_sub": S("es",  fs=9,  sa=2, ld=13, li=12),
    }

    def HR(color=BORDER, th=0.5, sb=6, sa=6):
        return HRFlowable(width="100%", thickness=th, color=color, spaceBefore=sb, spaceAfter=sa)

    def tbl(data, col_w, styles_extra=None):
        base_sty = [
            ("GRID", (0,0), (-1,-1), 0.4, BORDER),
            ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING", (0,0), (-1,-1), 8), ("RIGHTPADDING", (0,0), (-1,-1), 8),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
        ]
        if styles_extra: base_sty += styles_extra
        t = Table(data, colWidths=col_w)
        t.setStyle(TableStyle(base_sty))
        return t

    story = []

    # ===================== PAGE 1: COVER =====================
    hdr_p = [S("h1", fn="Helvetica-Bold", fs=18, tc=WHITE, sa=0, ld=22),
             S("h1r", fs=10, tc=C("#7dd3fc"), sa=0, ld=14, al=TA_RIGHT)]
    hdr = Table([[Paragraph("PolicyGraphAudit-RT", hdr_p[0]),
                  Paragraph("Compliance Audit Report<br/><font size='8' color='#94a3b8'>"
                            "Heterogeneous GNN Privacy Analysis</font>", hdr_p[1])]],
                colWidths=[W*.55, W*.45])
    hdr.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),NAVY),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(0,-1),14),("RIGHTPADDING",(-1,0),(-1,-1),14),
        ("TOPPADDING",(0,0),(-1,-1),14),("BOTTOMPADDING",(0,0),(-1,-1),14),
    ]))
    story += [hdr, Spacer(1,20),
              Paragraph(app_title, S("an", fn="Helvetica-Bold", fs=22, sa=4, ld=27)),
              Paragraph(appId, S("pk", fn="Courier", fs=9, tc=GRAY, sa=8)),
              HR()]

    info_rows = [["Developer", meta.get("developer","—")], ["Genre", meta.get("genreId","—")],
                 ["Installs", meta.get("installs_bucket","—")], ["Audit Date", today]]
    story.append(tbl(
        [[Paragraph(f"<b>{k}</b>", sty["lbl"]), Paragraph(v, sty["lbl"])] for k,v in info_rows],
        [W*.3, W*.7],
        [("BACKGROUND",(0,0),(-1,-1),LIGHT), ("ROWBACKGROUNDS",(0,0),(-1,-1),[WHITE,LIGHT])],
    ))
    story.append(Spacer(1,20))

    # Risk badge
    badge = Table([[
        Paragraph("Overall Risk Score", S("bl", fs=9, tc=WHITE, al=TA_CENTER, sa=0)),
        Paragraph(f"<b>{risk:.2f}</b>", S("bs", fn="Helvetica-Bold", fs=28, tc=WHITE, al=TA_CENTER, ld=34, sa=0)),
        Paragraph(f"<b>{rl}</b>", S("bcat", fn="Helvetica-Bold", fs=14, tc=WHITE, al=TA_CENTER, sa=0)),
    ]], colWidths=[W/3]*3)
    badge.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),C(rc)),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),14),("BOTTOMPADDING",(0,0),(-1,-1),14),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
    ]))
    story += [badge, Spacer(1,20)]

    # Count badges
    cls_order = ["CONSISTENT","POLICY_LABEL_MISMATCH","OVER_DISCLOSURE","UNDECLARED_COLLECTION"]
    cls_labels = ["Consistent","Mismatches","Over-disclosure","Undeclared"]
    cnt_cells = [Paragraph(
        f'<font color="{_CLASS_FG[c]}"><b>{counts[c]}</b></font><br/>{lb}',
        S(f"cnt{i}", fs=10, al=TA_CENTER, ld=14, sa=0))
        for i, (c, lb) in enumerate(zip(cls_order, cls_labels))]
    cnt_tbl = Table([cnt_cells], colWidths=[W/4]*4)
    cnt_tbl.setStyle(TableStyle(
        [("BACKGROUND",(i,0),(i,-1),C(_CLASS_BG[c])) for i,c in enumerate(cls_order)] +
        [("GRID",(0,0),(-1,-1),0.4,BORDER),("TOPPADDING",(0,0),(-1,-1),10),
         ("BOTTOMPADDING",(0,0),(-1,-1),10),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story.append(cnt_tbl)
    story.append(Spacer(1,12))

    pp = meta.get("privacyPolicy_url","")
    ft = "Research prototype · Ongoing research"
    if pp:
        ft += f'  |  Policy: <a href="{pp}" color="blue">{pp[:60]}{"..." if len(pp)>60 else ""}</a>'
    story += [Paragraph(ft, sty["ft"]), PageBreak()]

    # ===================== PAGE 2: EXEC SUMMARY =====================
    n_tot = len(preds); n_col = sum(1 for p in preds if p["evidence"]["has_label_collects"])
    n_sh  = sum(1 for p in preds if p["evidence"]["has_label_shares"])
    story += [Paragraph("Executive Summary", sty["h2"]), HR(color=ACCENT, th=1)]
    story.append(Paragraph(
        f"<b>{app_title}</b> declares {n_col} data type(s) collected and {n_sh} shared "
        f"in its Play Data Safety label, across {n_tot} data type(s) analyzed. "
        f"Our heterogeneous GNN audit model identifies "
        f"<b>{counts['UNDECLARED_COLLECTION']} potential undeclared collection(s)</b> "
        f"— data types implied by embedded SDK signals but absent from both label and policy — "
        f"and <b>{counts['POLICY_LABEL_MISMATCH']} policy-vs-label mismatch(es)</b>. "
        f"Overall risk score: <b>{risk:.2f} ({rl})</b>.", sty["body"]))
    story += [Spacer(1,10), Paragraph("<b>Top discrepancies by risk:</b>", sty["body"])]
    for p in preds_sorted[:3]:
        story.append(Paragraph(
            f"  - <b>{p['data_type']}</b>: {p['predicted_class'].replace('_',' ').title()} "
            f"(confidence {p['confidence']:.2f})", sty["bullet"]))
    story += [Spacer(1,10), Paragraph("<b>Class definitions:</b>", sty["body"])]
    defs = [("CONSISTENT","Label, policy, and SDK signals agree."),
            ("POLICY_LABEL_MISMATCH","Label discloses data not mentioned in policy."),
            ("OVER_DISCLOSURE","Policy mentions data not declared in label."),
            ("UNDECLARED_COLLECTION","SDK signals collection undisclosed in label and policy.")]
    for cls, defn in defs:
        story.append(Paragraph(
            f'  <font color="{_CLASS_FG[cls]}"><b>{cls}</b></font>: {defn}', sty["bullet"]))
    story.append(PageBreak())

    # ===================== PAGE 3+: DISCREPANCY TABLE =====================
    story += [Paragraph("Discrepancy Table", sty["h2"]), HR(color=ACCENT, th=1),
              Paragraph("Sorted by risk class (undeclared collection first). "
                        "Y/N: label=data-safety label, policy=policy text, sdk=SDK signal.",
                        sty["small"]), Spacer(1,6)]
    thead = [[Paragraph(f"<b>{h}</b>", sty["lbl"]) for h in
              ["Data Type","Class","Conf.","Evidence"]]]
    ts = [("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),WHITE),
          ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),
          ("GRID",(0,0),(-1,-1),0.4,BORDER),
          ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
          ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
          ("VALIGN",(0,0),(-1,-1),"TOP")]
    for ri, p in enumerate(preds_sorted, 1):
        ev = p["evidence"]; cn = p["predicted_class"]
        sdks = ev["sdks_involved"]
        sdk_str = ""
        if sdks:
            sdk_str = " (" + ", ".join(sdks[:3]) + (f" +{len(sdks)-3}" if len(sdks)>3 else "") + ")"
        evid = (f"label:{'Y' if ev['has_label_collects'] else 'N'} "
                f"policy:{'Y' if ev['has_policy_mentions'] else 'N'} "
                f"sdk:{'Y' if ev['has_sdk_collects'] else 'N'}" + sdk_str)
        thead.append([
            Paragraph(p["data_type"], sty["mono"]),
            Paragraph(f'<font color="{_CLASS_FG[cn]}"><b>{cn.replace("_"," ").title()}</b></font>', sty["lbl"]),
            Paragraph(f"{p['confidence']:.2f}", sty["lbl"]),
            Paragraph(evid, sty["small"]),
        ])
        ts.append(("BACKGROUND",(0,ri),(-1,ri),C(_CLASS_BG.get(cn,"#f8fafc"))))
    dt = Table(thead, colWidths=[W*.22, W*.28, W*.09, W*.41], repeatRows=1)
    dt.setStyle(TableStyle(ts))
    story += [dt, PageBreak()]

    # ===================== EVIDENCE TRAILS =====================
    story += [Paragraph("Evidence Trails", sty["h2"]), HR(color=ACCENT, th=1),
              Paragraph("Detailed evidence for UNDECLARED_COLLECTION or confidence > 0.85 "
                        "non-CONSISTENT findings.", sty["small"]), Spacer(1,8)]
    if not high_risk:
        story.append(Paragraph("No high-risk discrepancies detected.", sty["body"]))
    else:
        for p in high_risk:
            cn = p["predicted_class"]; ev = p["evidence"]
            sdks = ev["sdks_involved"]
            sdk_disp = (", ".join(sdks[:5]) + (f" (+{len(sdks)-5} more)" if len(sdks)>5 else "")) if sdks else "none"
            action_map = {
                "UNDECLARED_COLLECTION": f"Add '{p['data_type']}' to the Play Data Safety label under 'Data Collected' and update the privacy policy to disclose this collection.",
                "POLICY_LABEL_MISMATCH": f"Update the privacy policy to disclose '{p['data_type']}' collection, aligning it with the Play Data Safety label.",
                "OVER_DISCLOSURE": f"Review whether '{p['data_type']}' is genuinely collected. If yes, add to label; if no, remove from policy.",
            }
            block = [
                Paragraph(f'Discrepancy: <b>{p["data_type"]}</b> — '
                          f'<font color="{_CLASS_FG[cn]}"><b>{cn.replace("_"," ").title()}</b></font>'
                          f' (confidence {p["confidence"]:.2f})', sty["ev_hdr"]),
                HR(color=BORDER, th=0.4, sb=2, sa=4),
                Paragraph(f"  Label declares collection: <b>{'Yes' if ev['has_label_collects'] else 'No'}</b>", sty["ev_sub"]),
                Paragraph(f"  Label declares sharing: <b>{'Yes' if ev['has_label_shares'] else 'No'}</b>", sty["ev_sub"]),
                Paragraph(f"  Policy text mentions: <b>{'Yes' if ev['has_policy_mentions'] else 'No'}</b>", sty["ev_sub"]),
                Paragraph(f"  SDK collection signals: <b>{'Yes' if ev['has_sdk_collects'] else 'No'}</b> — {sdk_disp}", sty["ev_sub"]),
                Paragraph(f"  Recommended action: {action_map.get(cn, 'Review manually.')}", sty["ev_sub"]),
                Spacer(1,4),
            ]
            story.append(KeepTogether(block))
    story.append(PageBreak())

    # ===================== METHODOLOGY + DISCLAIMER =====================
    story += [Paragraph("Methodology", sty["h2"]), HR(color=ACCENT, th=1)]
    story.append(Paragraph(
        "PolicyGraphAudit-RT represents each Android app as a tri-partite heterogeneous graph "
        "connecting three layers: <b>Privacy Policy</b> text (OPP-115 segment classification), "
        "<b>Play Data Safety Labels</b> (declared data types and purposes), and "
        "<b>Runtime Evidence</b> (inferred SDK trackers via Exodus Privacy). "
        "A 2-layer HeteroConv R-GCN encodes all three layers jointly; an MLP classifier head "
        "predicts discrepancy classes for each (App, DataType) pair.", sty["body"]))
    story.append(Paragraph(
        "The model was trained under an edge-masking protocol (30% of label-determining edges removed) "
        f"to prevent circularity. It achieves <b>macro F1 = 0.956</b> on 39 held-out apps "
        f"(UNDECLARED_COLLECTION F1 = 0.974), trained on 252 apps and 3,202 labeled pairs. "
        f'Model card: <a href="{_MODEL_CARD_URL}" color="blue">{_MODEL_CARD_URL}</a>.', sty["body"]))
    story.append(Paragraph(
        "Known limitations: (1) Privacy labels are from a 2022 Play Store snapshot. "
        "(2) SDK presence is inferred via category-level priors, not measured from APK bytecode. "
        "(3) Policy classifier uses MiniLM + logistic regression (OPP-115 macro F1 = 0.70). "
        "(4) Labels are rule-derived weak supervision, not human-annotated. "
        "(5) English-language Android apps only.", sty["body"]))
    story += [Spacer(1,8), Paragraph("Disclaimer", sty["h2"]), HR(color=BORDER)]
    story.append(Paragraph(
        "This report is produced by a research prototype and is provided for informational "
        "and research purposes only. It does not constitute legal advice or a regulatory "
        "determination. Findings are based on automated weak-supervision labels from publicly "
        "available data and have not been reviewed by a privacy legal professional. "
        "Research prototype · Ongoing research.", sty["disc"]))

    # -- Footer callback --
    def _footer(cv, doc):
        cv.saveState()
        cv.setFont("Helvetica", 7)
        cv.setFillColor(GRAY)
        cv.drawString(0.85*inch, 0.4*inch, f"PolicyGraphAudit-RT  |  {app_title}  |  {today}")
        cv.drawRightString(letter[0]-0.85*inch, 0.4*inch,
                           f"Page {doc.page}  |  Research prototype · Ongoing research")
        cv.restoreState()

    doc = SimpleDocTemplate(str(out), pagesize=letter,
                            leftMargin=0.85*inch, rightMargin=0.85*inch,
                            topMargin=0.9*inch, bottomMargin=0.75*inch,
                            title=f"PolicyGraphAudit-RT — {app_title}",
                            author="Perplexity Computer",
                            subject="Privacy Compliance Audit Report")
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return str(out)
