"""Tab 1: Overview — project narrative, architecture, dataset summary."""

from dash import html, dcc
import plotly.graph_objects as go

# Architecture diagram (M1→M7 pipeline)
def _build_arch_figure():
    modules = [
        ("M1", "Acquire", "Data\nAcquisition"),
        ("M2", "Policy\nGraph", "Policy\nGraph"),
        ("M3", "Runtime\nGraph", "Runtime\nGraph"),
        ("M4", "Fusion", "Graph\nFusion &\nLabels"),
        ("M5", "Model", "HeteroGNN\nClassifier"),
        ("M6", "Report", "Audit\nReport"),
        ("M7", "Dashboard", "This\nDashboard"),
    ]
    n = len(modules)
    xs = [i * 1.6 for i in range(n)]
    y = 1.0
    box_w, box_h = 1.0, 0.6

    shapes = []
    annotations = []
    # Arrows between boxes
    for i in range(n - 1):
        shapes.append(dict(
            type="line",
            x0=xs[i] + box_w / 2, y0=y,
            x1=xs[i + 1] - box_w / 2, y1=y,
            line=dict(color="#94a3b8", width=1.5),
            xref="x", yref="y",
        ))
        # Arrowhead
        shapes.append(dict(
            type="line",
            x0=xs[i + 1] - box_w / 2 - 0.08, y0=y + 0.06,
            x1=xs[i + 1] - box_w / 2, y1=y,
            line=dict(color="#94a3b8", width=1.5),
            xref="x", yref="y",
        ))
        shapes.append(dict(
            type="line",
            x0=xs[i + 1] - box_w / 2 - 0.08, y0=y - 0.06,
            x1=xs[i + 1] - box_w / 2, y1=y,
            line=dict(color="#94a3b8", width=1.5),
            xref="x", yref="y",
        ))
    # Boxes
    for i, (mid, short, label) in enumerate(modules):
        is_active = (mid == "M7")
        fill = "#dbeafe" if is_active else "#eff6ff"
        border = "#2563eb" if is_active else "#bfdbfe"
        shapes.append(dict(
            type="rect",
            x0=xs[i] - box_w / 2, y0=y - box_h / 2,
            x1=xs[i] + box_w / 2, y1=y + box_h / 2,
            fillcolor=fill,
            line=dict(color=border, width=1.5 if is_active else 1),
            xref="x", yref="y",
        ))
        annotations.append(dict(
            x=xs[i], y=y + 0.08,
            text=f"<b>{mid}</b>",
            showarrow=False, xref="x", yref="y",
            font=dict(size=11, color="#1d4ed8" if is_active else "#374151"),
        ))
        lines = label.split("\n")
        txt = "<br>".join(lines)
        annotations.append(dict(
            x=xs[i], y=y - 0.12,
            text=f"<span style='font-size:9px;color:#6b7280'>{txt}</span>",
            showarrow=False, xref="x", yref="y",
            font=dict(size=9, color="#6b7280"),
        ))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=[y] * n,
        mode="markers",
        marker=dict(size=0.1, color="rgba(0,0,0,0)"),
        hoverinfo="skip",
    ))
    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        height=140,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#fafafa",
        plot_bgcolor="#fafafa",
        xaxis=dict(visible=False, range=[-0.8, xs[-1] + 0.8]),
        yaxis=dict(visible=False, range=[0.25, 1.7]),
        showlegend=False,
    )
    return fig


# Comparison table vs prior topic-modeling baseline
_comparison_rows = [
    ("Method", "Prior topic-modeling baseline", "PolicyGraphAudit-RT (Ours)"),
    ("Representation", "t-SNE + KMeans on policy text", "Heterogeneous GNN over multi-source KG"),
    ("Runtime evidence", "None", "SDK tracker registry + Data Safety labels"),
    ("Labels", "Unsupervised clusters", "Weak-supervised from cross-source rules"),
    ("Discrepancy classes", "Binary (gap/no-gap)", "4-class (CONSISTENT / PLM / OVR / UNDECL)"),
    ("Macro F1 (our repro)", "0.2802", "0.9561 (masked eval)"),
    ("Edge-masking protocol", "N/A", "30% of label-determining edges masked"),
    ("Apps", "~200 (text only)", "268 (policy + label + runtime)"),
]


def _comparison_table():
    rows = []
    for i, (dim, them, us) in enumerate(_comparison_rows):
        bg = "#f9fafb" if i == 0 else ("white" if i % 2 == 0 else "#f9fafb")
        style = {"background": bg}
        rows.append(html.Tr([
            html.Td(dim, style={"fontWeight": "600", "fontSize": "0.78rem",
                                "color": "#6b7280", "padding": "7px 12px",
                                "borderBottom": "1px solid #f3f4f6"}),
            html.Td(them, style={"fontSize": "0.78rem", "padding": "7px 12px",
                                 "borderBottom": "1px solid #f3f4f6"}),
            html.Td(us, style={"fontSize": "0.78rem", "padding": "7px 12px",
                               "borderBottom": "1px solid #f3f4f6",
                               "color": "#2563eb", "fontWeight": "500"}),
        ], style=style))
    return html.Table([
        html.Thead(html.Tr([
            html.Th("Dimension", style={"padding": "8px 12px", "background": "#f3f4f6",
                                        "fontSize": "0.7rem", "fontWeight": "700",
                                        "textTransform": "uppercase", "letterSpacing": "0.05em"}),
            html.Th("Prior topic-modeling baseline", style={"padding": "8px 12px", "background": "#f3f4f6",
                                                     "fontSize": "0.7rem", "fontWeight": "700",
                                                     "textTransform": "uppercase", "letterSpacing": "0.05em"}),
            html.Th("This Work", style={"padding": "8px 12px", "background": "#dbeafe",
                                         "fontSize": "0.7rem", "fontWeight": "700",
                                         "textTransform": "uppercase", "letterSpacing": "0.05em",
                                         "color": "#2563eb"}),
        ])),
        html.Tbody(rows[1:]),  # skip header row used as sentinel
    ], style={"width": "100%", "borderCollapse": "collapse",
              "border": "1px solid #e5e7eb", "borderRadius": "8px", "overflow": "hidden"})


def layout():
    return html.Div([
        # Hero panel
        html.Div([
            html.H1("PolicyGraphAudit-RT", className="hero-title"),
            html.P(
                "A heterogeneous graph neural network that detects fine-grained privacy-policy "
                "discrepancies in Android apps by jointly encoding policy text, Play Store data-safety "
                "labels, and third-party SDK tracker evidence in a single multi-relational knowledge graph.",
                className="hero-thesis",
            ),
            html.Div([
                html.Span("268 fused graphs", className="hero-meta-item"),
                html.Span("·", style={"color": "#d1d5db"}),
                html.Span("3,202 labeled pairs", className="hero-meta-item"),
                html.Span("·", style={"color": "#d1d5db"}),
                html.Span("Macro F1 = 0.9561 (masked)", className="hero-meta-item",
                          style={"color": "#2563eb", "fontWeight": "600"}),
                html.Span("·", style={"color": "#d1d5db"}),
                html.Span("Research prototype · Ongoing research", className="hero-meta-item"),
            ], className="hero-meta"),
        ], className="hero-panel"),

        # Architecture diagram
        html.P("Pipeline Architecture", className="section-header"),
        html.Div([
            dcc.Graph(
                figure=_build_arch_figure(),
                config={"displayModeBar": False},
                style={"height": "140px"},
            ),
        ], className="card", style={"padding": "16px 8px 8px 8px"}),

        # Dataset stats
        html.P("Dataset At a Glance", className="section-header"),
        html.Div([
            html.Div([
                html.Div("268", className="stat-value"),
                html.Div("Fused Knowledge Graphs", className="stat-label"),
            ], className="stat-tile"),
            html.Div([
                html.Div("3,202", className="stat-value"),
                html.Div("Labeled (App, DataType) Pairs", className="stat-label"),
            ], className="stat-tile"),
            html.Div([
                html.Div("432", className="stat-value"),
                html.Div("Third-Party SDKs Tracked", className="stat-label"),
            ], className="stat-tile"),
            html.Div([
                html.Div("0.974", className="stat-value"),
                html.Div("UNDECL Collection F1", className="stat-label"),
            ], className="stat-tile"),
        ], className="grid-4", style={"marginBottom": "16px"}),

        # Source breakdown
        html.Div([
            html.Div([
                html.Div("Data Sources", className="callout-title"),
                html.Ul([
                    html.Li("OPP-115 Corpus — 10-category policy segment classifier"),
                    html.Li("Princeton PPC — 5,000 sampled privacy policies"),
                    html.Li("Google Play Data Safety Labels — 12.97M disclosure rows"),
                    html.Li("Exodus Privacy / Yale Privacy Lab — 432 tracker profiles"),
                    html.Li("TrackerControl X-Ray — 771 tracker domains"),
                ], style={"margin": "0", "paddingLeft": "18px", "fontSize": "0.85rem", "lineHeight": "1.8"}),
            ], className="callout-blue"),

            html.Div([
                html.Div("Label Distribution", className="callout-title"),
                html.Ul([
                    html.Li("OVER_DISCLOSURE — 1,659 pairs (51.8%)"),
                    html.Li("CONSISTENT — 561 pairs (17.5%)"),
                    html.Li("POLICY_LABEL_MISMATCH — 542 pairs (16.9%)"),
                    html.Li("UNDECLARED_COLLECTION — 440 pairs (13.7%)"),
                ], style={"margin": "0", "paddingLeft": "18px", "fontSize": "0.85rem", "lineHeight": "1.8"}),
            ], className="callout-green"),
        ], className="grid-2", style={"marginBottom": "16px"}),

        # Comparison vs prior topic-modeling baseline
        html.P("What is New vs. Prior Baselines", className="section-header"),
        html.Div([
            _comparison_table(),
        ], className="card", style={"padding": "0", "overflow": "hidden"}),

    ], className="tab-content")
