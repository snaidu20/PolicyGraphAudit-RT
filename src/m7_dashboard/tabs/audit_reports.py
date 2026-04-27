"""Tab 6: Audit Reports — per-app PDF generation and sample previews."""

import os
import glob as glob_mod
from functools import lru_cache
from dash import html, dcc, callback, Input, Output, State
import pandas as pd

_LABELS_PATH = "" + os.environ.get("PGART_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))) + "/data/processed/discrepancy_labels_full.parquet"
_AUDITS_DIR = "" + os.environ.get("PGART_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))) + "/reports/audits"

# In production (pplx.app) the backend lives behind a /port/5000/ proxy prefix.
# Locally REQUESTS_PATHNAME_PREFIX is unset/'/' so links resolve to bare /reports/...
_URL_PREFIX = os.environ.get("REQUESTS_PATHNAME_PREFIX", "/").rstrip("/")


def _audit_url(fname: str) -> str:
    return f"{_URL_PREFIX}/reports/audits/{fname}"


@lru_cache(maxsize=1)
def _app_options():
    try:
        df = pd.read_parquet(_LABELS_PATH)
        apps = sorted(df["app_id"].unique().tolist())
    except Exception:
        apps = []
    return [{"label": a, "value": a} for a in apps]


def _existing_pdfs():
    """Return list of existing audit PDF paths."""
    try:
        pdfs = sorted(glob_mod.glob(os.path.join(_AUDITS_DIR, "*.pdf")))
        return pdfs
    except Exception:
        return []


def _try_generate_pdf(app_id: str):
    """Look up the pre-generated audit PDF for an app; return (path_or_none, message)."""
    if not app_id:
        return None, "Please select an app first."
    candidate = os.path.join(_AUDITS_DIR, f"{app_id}.pdf")
    if os.path.exists(candidate):
        return candidate, "Audit PDF available."
    return None, f"No pre-generated audit PDF available for {app_id}."


def _sample_preview_cards(pdfs):
    """Return preview cards for up to 6 sample PDFs."""
    if not pdfs:
        return [html.Div([
            html.Div("[No audit PDFs found in reports/audits/]",
                     style={"color": "#9ca3af", "fontSize": "0.85rem", "textAlign": "center"}),
        ], className="pdf-placeholder")]

    cards = []
    for pdf_path in pdfs[:6]:
        app_name = os.path.splitext(os.path.basename(pdf_path))[0]
        cards.append(html.Div([
            html.Div(app_name, style={"fontWeight": "600", "fontSize": "0.85rem",
                                       "marginBottom": "8px", "color": "#374151",
                                       "wordBreak": "break-all"}),
            html.A(
                "Download PDF",
                href=_audit_url(os.path.basename(pdf_path)),
                target="_blank",
                style={"color": "#2563eb", "fontSize": "0.8rem",
                        "textDecoration": "none", "fontWeight": "500"},
            ),
        ], className="card-sm"))
    return cards


def layout():
    opts = _app_options()
    default_app = opts[0]["value"] if opts else None
    existing_pdfs = _existing_pdfs()

    return html.Div([
        html.P("Audit Reports", className="section-header"),

        html.Div([
            html.Div(
                "Browse pre-generated per-app privacy audit PDFs for the 252 labeled apps in the dataset. "
                "Each report includes: discrepancy class breakdown, SDK risk summary, and "
                "policy-vs-practice comparison table.",
                style={"fontSize": "0.85rem", "color": "#374151", "marginBottom": "4px"},
            ),
            html.Div(
                "Pick an app below to fetch its audit PDF, or browse the samples further down.",
                style={"fontSize": "0.75rem", "color": "#6b7280"},
            ),
        ], className="callout-blue", style={"marginBottom": "16px"}),

        # App picker + generate
        html.Div([
            html.Div([
                html.Label("Select app to audit:", style={"fontWeight": "600",
                                                            "fontSize": "0.8rem",
                                                            "marginBottom": "6px",
                                                            "display": "block"}),
                dcc.Dropdown(
                    id="ar-app-picker",
                    options=opts,
                    value=default_app,
                    placeholder="Search for an app...",
                    style={"fontSize": "0.85rem", "marginBottom": "12px"},
                    clearable=False,
                ),
                html.Button(
                    "Get Audit PDF",
                    id="ar-generate-btn",
                    n_clicks=0,
                    style={
                        "background": "#2563eb",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "6px",
                        "padding": "9px 20px",
                        "fontSize": "0.85rem",
                        "fontWeight": "600",
                        "cursor": "pointer",
                        "fontFamily": "Inter, system-ui, sans-serif",
                        "letterSpacing": "0.01em",
                    },
                ),
                html.Div(id="ar-generate-status",
                         style={"marginTop": "12px", "fontSize": "0.82rem"}),
                html.Div(id="ar-download-link", style={"marginTop": "8px"}),
            ], className="card"),
        ], style={"marginBottom": "16px"}),

        # Sample PDFs
        html.P("Sample Audit Reports", className="section-header"),
        html.Div(
            f"Showing {min(6, len(existing_pdfs))} of {len(existing_pdfs)} pre-generated audit PDFs. "
            "Use the picker above to fetch any specific app's PDF.",
            style={"fontSize": "0.78rem", "color": "#6b7280", "marginBottom": "10px"},
        ),
        html.Div(
            _sample_preview_cards(existing_pdfs),
            className="grid-3",
            style={"marginBottom": "16px"},
        ),

        # What the report contains
        html.P("Report Contents", className="section-header"),
        html.Div([
            html.Div([
                html.Div("Each audit report contains:", className="callout-title"),
                html.Ul([
                    html.Li("App metadata (package name, genre, developer)"),
                    html.Li("Discrepancy summary table: all (App, DataType) pairs and their predicted class"),
                    html.Li("UNDECLARED_COLLECTION details: which SDKs imply undisclosed collection"),
                    html.Li("Policy-vs-label comparison: declared vs. practice for each data type"),
                    html.Li("Risk score and remediation suggestions"),
                    html.Li("Model confidence scores per prediction"),
                ], style={"margin": "0", "paddingLeft": "18px",
                           "fontSize": "0.85rem", "lineHeight": "1.8"}),
            ], className="callout-blue"),
        ]),

    ], className="tab-content")


def register_callbacks(app):
    @app.callback(
        Output("ar-generate-status", "children"),
        Output("ar-download-link", "children"),
        Input("ar-generate-btn", "n_clicks"),
        State("ar-app-picker", "value"),
        prevent_initial_call=True,
    )
    def generate_report(n_clicks, app_id):
        if not app_id:
            return "Please select an app first.", ""
        pdf_path, message = _try_generate_pdf(app_id)
        if pdf_path and os.path.exists(pdf_path):
            fname = os.path.basename(pdf_path)
            link = html.A(
                f"Download: {fname}",
                href=_audit_url(fname),
                target="_blank",
                style={"color": "#2563eb", "fontWeight": "600",
                        "fontSize": "0.82rem", "textDecoration": "none"},
            )
            return html.Span(message, style={"color": "#16a34a"}), link
        else:
            return html.Span(message, style={"color": "#f59e0b"}), ""
