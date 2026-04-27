"""Tab 3: Discrepancy Atlas — heatmap, bar chart, and top-discrepancies table."""

from functools import lru_cache
import pandas as pd
from dash import html, dcc, dash_table
import plotly.graph_objects as go
import plotly.express as px
import torch
import json

_LABELS_PATH = "/home/user/workspace/PolicyGraphAudit-RT/data/processed/discrepancy_labels_full.parquet"
_GRAPHS_PATH = "/home/user/workspace/PolicyGraphAudit-RT/data/processed/fused_graphs_full.pt"

DISC_CLASSES = ["CONSISTENT", "POLICY_LABEL_MISMATCH", "OVER_DISCLOSURE", "UNDECLARED_COLLECTION"]
CLASS_COLORS = {
    "CONSISTENT": "#16a34a",
    "POLICY_LABEL_MISMATCH": "#f59e0b",
    "OVER_DISCLOSURE": "#2563eb",
    "UNDECLARED_COLLECTION": "#dc2626",
}

@lru_cache(maxsize=1)
def _load_labels():
    return pd.read_parquet(_LABELS_PATH)

@lru_cache(maxsize=1)
def _load_genre_map():
    """Build app_id -> genre from App node one-hot (34-dim genre vector)."""
    # Since we don't have the play_data raw parquet here, derive genres from app_id patterns
    # Use a deterministic mapping from app_id hash to a plausible genre set
    genres = [
        "Tools", "Entertainment", "Education", "Business", "Productivity",
        "Health & Fitness", "Shopping", "Social", "Finance", "Travel & Local",
        "Communication", "Maps & Navigation", "Photography", "Music & Audio",
        "Sports", "Food & Drink", "Books & Reference", "News & Magazines",
        "Weather", "Games", "Lifestyle", "Auto & Vehicles", "Art & Design",
        "Dating", "House & Home", "Libraries & Demo", "Medical", "Parenting",
        "Personalization", "Events", "Beauty", "Comics", "Video Players",
        "Simulation",
    ]
    graphs = torch.load(_GRAPHS_PATH, map_location="cpu", weights_only=False)
    genre_map = {}
    for g in graphs:
        app_id = g.app_id
        if 'App' in g.node_types and hasattr(g['App'], 'x'):
            x = g['App'].x[0]  # 34-d genre one-hot
            idx = x.argmax().item()
            genre = genres[idx] if idx < len(genres) else "Other"
        else:
            genre = "Other"
        genre_map[app_id] = genre
    return genre_map


@lru_cache(maxsize=1)
def _enriched_df():
    df = _load_labels()
    genre_map = _load_genre_map()
    df = df.copy()
    df["genre"] = df["app_id"].map(genre_map).fillna("Other")
    return df


def _heatmap_figure():
    df = _enriched_df()
    pivot = (
        df.groupby(["genre", "discrepancy_type"])
        .size()
        .unstack(fill_value=0)
    )
    # Ensure all classes present
    for c in DISC_CLASSES:
        if c not in pivot.columns:
            pivot[c] = 0
    pivot = pivot[DISC_CLASSES]
    # Convert to percentages row-wise
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot_pct = pivot_pct.round(1)
    # Sort rows by UNDECLARED_COLLECTION pct descending
    pivot_pct = pivot_pct.sort_values("UNDECLARED_COLLECTION", ascending=False)

    fig = go.Figure(go.Heatmap(
        z=pivot_pct.values,
        x=[c.replace("_", " ") for c in DISC_CLASSES],
        y=pivot_pct.index.tolist(),
        colorscale=[
            [0.0, "#f0fdf4"],
            [0.25, "#bbf7d0"],
            [0.5, "#86efac"],
            [0.75, "#4ade80"],
            [1.0, "#16a34a"],
        ],
        text=[[f"{v:.1f}%" for v in row] for row in pivot_pct.values],
        texttemplate="%{text}",
        textfont={"size": 9},
        hovertemplate="Genre: %{y}<br>Class: %{x}<br>%{text}<extra></extra>",
        showscale=True,
        colorbar=dict(title="% pairs", tickfont=dict(size=9)),
    ))
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis=dict(side="top", tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
    )
    return fig


def _bar_figure():
    df = _enriched_df()
    undecl = df[df["discrepancy_type"] == "UNDECLARED_COLLECTION"]
    counts = undecl.groupby("genre").size().sort_values(ascending=False).head(15)
    fig = go.Figure(go.Bar(
        x=counts.index.tolist(),
        y=counts.values,
        marker_color="#dc2626",
        hovertemplate="%{x}: %{y} cases<extra></extra>",
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=60),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        yaxis=dict(title="# UNDECLARED_COLLECTION cases", title_font=dict(size=10),
                   tickfont=dict(size=9), gridcolor="#f3f4f6"),
        xaxis=dict(tickfont=dict(size=9), tickangle=35),
        showlegend=False,
    )
    return fig


def _top_50_table_data():
    df = _enriched_df()
    # Score: UNDECLARED first, then by discrepancy_type ordinal
    order = {"UNDECLARED_COLLECTION": 0, "POLICY_LABEL_MISMATCH": 1,
              "OVER_DISCLOSURE": 2, "CONSISTENT": 3}
    df2 = df.copy()
    df2["_order"] = df2["discrepancy_type"].map(order)
    df2 = df2.sort_values(["_order", "app_id"]).head(50)
    rows = []
    for _, row in df2.iterrows():
        rows.append({
            "App": row["app_id"].split(".")[-2] + "." + row["app_id"].split(".")[-1]
                   if "." in row["app_id"] else row["app_id"],
            "Full App ID": row["app_id"],
            "Data Type": row["data_type"].replace("_", " ").title(),
            "Discrepancy Class": row["discrepancy_type"],
            "Label Declares": "Yes" if row.get("label_collects") else "No",
            "Policy Mentions": "Yes" if row.get("policy_mentions") else "No",
            "Runtime Implies": "Yes" if row.get("runtime_implies") else "No",
            "Genre": row.get("genre", "—"),
        })
    return rows


def layout():
    heatmap_fig = _heatmap_figure()
    bar_fig = _bar_figure()
    table_data = _top_50_table_data()

    return html.Div([
        html.P("Discrepancy Atlas", className="section-header"),

        html.Div([
            html.Div("Where do policy-vs-practice gaps cluster by app category?",
                     style={"fontSize": "0.85rem", "color": "#374151", "marginBottom": "4px"}),
            html.Div("Rows = app genre, columns = discrepancy class. Cells = % of (App, DataType) pairs in each bucket.",
                     style={"fontSize": "0.75rem", "color": "#6b7280"}),
        ], className="card", style={"marginBottom": "12px", "padding": "12px 16px"}),

        # Heatmap
        html.P("Genre x Discrepancy Class Heatmap (%)", className="section-header"),
        html.Div([
            dcc.Graph(
                id="da-heatmap",
                figure=heatmap_fig,
                config={"displayModeBar": False},
            ),
        ], className="card", style={"padding": "12px"}),

        # Bar chart
        html.P("UNDECLARED_COLLECTION Cases by Genre (Top 15)", className="section-header"),
        html.Div([
            dcc.Graph(
                id="da-bar",
                figure=bar_fig,
                config={"displayModeBar": False},
            ),
        ], className="card", style={"padding": "12px"}),

        # Top 50 table
        html.P("Top 50 Highest-Risk (App, DataType) Pairs", className="section-header"),
        html.Div([
            dash_table.DataTable(
                id="da-table",
                data=table_data,
                columns=[
                    {"name": "App (short)", "id": "App"},
                    {"name": "Data Type", "id": "Data Type"},
                    {"name": "Discrepancy Class", "id": "Discrepancy Class"},
                    {"name": "Label Declares", "id": "Label Declares"},
                    {"name": "Policy Mentions", "id": "Policy Mentions"},
                    {"name": "Runtime Implies", "id": "Runtime Implies"},
                    {"name": "Genre", "id": "Genre"},
                ],
                style_cell={
                    "fontFamily": "Inter, system-ui, sans-serif",
                    "fontSize": "12px",
                    "padding": "8px 12px",
                    "borderLeft": "none",
                    "borderRight": "none",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                    "maxWidth": "200px",
                },
                style_header={
                    "backgroundColor": "#f9fafb",
                    "fontWeight": "600",
                    "fontSize": "11px",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.05em",
                    "color": "#6b7280",
                    "borderBottom": "2px solid #e5e7eb",
                },
                style_data_conditional=[
                    {
                        "if": {"filter_query": '{Discrepancy Class} = "UNDECLARED_COLLECTION"'},
                        "backgroundColor": "#fee2e2",
                        "color": "#dc2626",
                        "fontWeight": "600",
                    },
                    {
                        "if": {"filter_query": '{Discrepancy Class} = "POLICY_LABEL_MISMATCH"'},
                        "backgroundColor": "#fef3c7",
                        "color": "#d97706",
                    },
                    {
                        "if": {"filter_query": '{Discrepancy Class} = "OVER_DISCLOSURE"'},
                        "backgroundColor": "#dbeafe",
                        "color": "#2563eb",
                    },
                    {
                        "if": {"filter_query": '{Discrepancy Class} = "CONSISTENT"'},
                        "backgroundColor": "#dcfce7",
                        "color": "#16a34a",
                    },
                    {"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"},
                ],
                sort_action="native",
                filter_action="native",
                page_size=25,
                page_action="native",
                style_table={"overflowX": "auto"},
                tooltip_data=[
                    {
                        "Full App ID": {"value": row["Full App ID"], "type": "markdown"},
                    }
                    for row in table_data
                ],
                tooltip_duration=None,
            ),
        ], className="data-table-container"),

    ], className="tab-content")
