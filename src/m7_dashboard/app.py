"""
PolicyGraphAudit-RT — M7 Interactive Plotly Dash Dashboard
===========================================================
6-tab dashboard: Overview | Graph Explorer | Discrepancy Atlas |
                 SDK Risk Leaderboard | Model Card | Audit Reports

Boot:
    cd /home/user/workspace/PolicyGraphAudit-RT
    python -m src.m7_dashboard.app

Production (gunicorn):
    REQUESTS_PATHNAME_PREFIX=/port/5000/ gunicorn \
        --workers 2 --bind 0.0.0.0:5000 src.m7_dashboard.app:server
"""

import os
import sys

# Make the project root importable
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dash import Dash, html, dcc, Input, Output

# MicroPlastiNet-style: configurable URL prefix for pplx.app deployment
_REQUESTS_PREFIX = os.environ.get("REQUESTS_PATHNAME_PREFIX", "/")

EXTERNAL_STYLESHEETS = [
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap",
]

app = Dash(
    __name__,
    title="PolicyGraphAudit-RT",
    requests_pathname_prefix=_REQUESTS_PREFIX,
    suppress_callback_exceptions=True,
    external_stylesheets=EXTERNAL_STYLESHEETS,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
    ],
)
server = app.server  # for gunicorn

# ---- Lazy tab imports to keep startup fast --------------------------------
from src.m7_dashboard.tabs import overview
from src.m7_dashboard.tabs import graph_explorer
from src.m7_dashboard.tabs import discrepancy_atlas
from src.m7_dashboard.tabs import sdk_leaderboard
from src.m7_dashboard.tabs import model_card
from src.m7_dashboard.tabs import audit_reports

# ---- Register per-tab callbacks -------------------------------------------
graph_explorer.register_callbacks(app)
sdk_leaderboard.register_callbacks(app)
audit_reports.register_callbacks(app)


# ---- Top-level layout -----------------------------------------------------

def _header():
    return html.Header([
        html.Div([
            html.Span("PolicyGraph", className="app-header-title"),
            html.Span("Audit-RT", className="app-header-title",
                      style={"color": "#2563eb"}),
        ]),
        html.Div([
            html.Span("Research prototype · Ongoing research", className="status-badge"),
        ], className="header-meta"),
    ], className="app-header")


def _footer():
    return html.Footer([
        html.Span("PolicyGraphAudit-RT · Research prototype · Ongoing research",
                  style={"color": "#9ca3af"}),
        html.Div([
            html.A("GitHub", href="https://github.com/snaidu20/PolicyGraphAudit-RT",
                   target="_blank", style={"marginRight": "16px"}),
            html.A("Methodology", href="#overview"),
        ]),
    ], className="app-footer")


_TABS_CONFIG = [
    {"label": "Overview",             "value": "overview"},
    {"label": "Graph Explorer",       "value": "graph_explorer"},
    {"label": "Discrepancy Atlas",    "value": "discrepancy_atlas"},
    {"label": "SDK Risk Leaderboard", "value": "sdk_leaderboard"},
    {"label": "Model Card",           "value": "model_card"},
    {"label": "Audit Reports",        "value": "reports"},
]

app.layout = html.Div([
    _header(),

    # Tab bar
    html.Div([
        dcc.Tabs(
            id="main-tabs",
            value="overview",
            children=[
                dcc.Tab(label=t["label"], value=t["value"],
                        className="custom-tab",
                        selected_className="custom-tab--selected")
                for t in _TABS_CONFIG
            ],
            className="dash-tabs--top",
            colors={"border": "#e5e7eb", "primary": "#2563eb", "background": "#ffffff"},
        ),
    ], className="tab-container"),

    # Tab content — rendered on demand
    html.Div(id="tab-content"),

    _footer(),
], style={"minHeight": "100vh", "background": "#fafafa"})


# ---- Main routing callback ------------------------------------------------

@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab):
    """Render the selected tab's layout. All errors are caught to keep other tabs alive."""
    try:
        if tab == "overview":
            return overview.layout()
        elif tab == "graph_explorer":
            return graph_explorer.layout()
        elif tab == "discrepancy_atlas":
            return discrepancy_atlas.layout()
        elif tab == "sdk_leaderboard":
            return sdk_leaderboard.layout()
        elif tab == "model_card":
            return model_card.layout()
        elif tab == "reports":
            return audit_reports.layout()
        else:
            return html.Div(f"Unknown tab: {tab}", className="tab-content")
    except Exception as exc:
        return html.Div([
            html.Div(f"Error rendering tab '{tab}':", style={"fontWeight": "700",
                                                              "color": "#dc2626",
                                                              "marginBottom": "8px"}),
            html.Pre(str(exc), style={"fontSize": "0.8rem", "color": "#374151",
                                       "background": "#fef2f2", "padding": "12px",
                                       "borderRadius": "6px", "overflow": "auto"}),
        ], className="tab-content")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("DASH_DEBUG", "false").lower() == "true"
    app.run(debug=debug, host="0.0.0.0", port=port)
