"""Tab 4: SDK Risk Leaderboard — sortable SDK table with detail panel."""

import json
from functools import lru_cache
from dash import html, dcc, dash_table, callback, Input, Output
import torch
import pandas as pd

_SDK_PATH = "/home/user/workspace/PolicyGraphAudit-RT/data/processed/sdk_registry.json"
_GRAPHS_PATH = "/home/user/workspace/PolicyGraphAudit-RT/data/processed/fused_graphs_full.pt"
_LABELS_PATH = "/home/user/workspace/PolicyGraphAudit-RT/data/processed/discrepancy_labels_full.parquet"


@lru_cache(maxsize=1)
def _load_sdks():
    with open(_SDK_PATH) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_graphs():
    return torch.load(_GRAPHS_PATH, map_location="cpu", weights_only=False)


@lru_cache(maxsize=1)
def _load_labels():
    return pd.read_parquet(_LABELS_PATH)


@lru_cache(maxsize=1)
def _build_sdk_stats():
    """
    For each SDK: # apps containing it, # data types, # apps with UNDECLARED.
    """
    graphs = _load_graphs()
    labels_df = _load_labels()
    sdks = _load_sdks()

    # Map SDK name -> data from registry
    sdk_by_name = {s["name"].lower(): s for s in sdks}

    # Build sdk_name -> set of app_ids from CONTAINS_SDK edges
    sdk_apps = {}  # sdk_name -> set of app_ids
    for g in graphs:
        if "SDK" not in g.node_types:
            continue
        app_id = g.app_id
        nids = g["SDK"].node_ids if hasattr(g["SDK"], "node_ids") else []
        for nid in nids:
            # SDK node_ids look like: "SDK::tracker_name::N" or "TrackerName::N"
            parts = str(nid).split("::")
            # Try to find a recognizable SDK name
            sdk_key = parts[-2].lower() if len(parts) >= 2 else parts[0].lower()
            sdk_apps.setdefault(sdk_key, set()).add(app_id)

    # UNDECLARED counts per sdk_name
    undecl = labels_df[labels_df["discrepancy_type"] == "UNDECLARED_COLLECTION"]
    undecl_apps = set(undecl["app_id"].unique())

    rows = []
    for sdk in sdks:
        name = sdk["name"]
        key = name.lower()
        apps_set = sdk_apps.get(key, set())
        n_apps = len(apps_set)
        n_undecl = len(apps_set & undecl_apps)
        rows.append({
            "sdk_name": name,
            "owner": sdk.get("owner_company", "—"),
            "category": sdk.get("category", "—"),
            "n_apps": n_apps,
            "n_datatypes": len(sdk.get("collects_data_types", [])),
            "n_undecl": n_undecl,
            "has_yale": sdk.get("has_yale", False),
            "tracker_id": sdk.get("tracker_id", ""),
            "collects_data_types": sdk.get("collects_data_types", []),
            "purposes": sdk.get("purposes", []),
            "canonical_purpose": sdk.get("canonical_purpose", ""),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("n_apps", ascending=False).reset_index(drop=True)
    return df


def _table_records():
    df = _build_sdk_stats()
    records = []
    for _, row in df.iterrows():
        records.append({
            "SDK Name": row["sdk_name"],
            "Owner": row["owner"],
            "Category": row["category"],
            "# Apps": int(row["n_apps"]),
            "# Data Types": int(row["n_datatypes"]),
            "# Apps w/ UNDECL": int(row["n_undecl"]),
            "Yale Profile": "Yes" if row["has_yale"] else "No",
        })
    return records


def layout():
    records = _table_records()
    df = _build_sdk_stats()

    return html.Div([
        html.P("SDK Risk Leaderboard", className="section-header"),

        html.Div([
            html.Div("Which third-party SDKs are implicated in the most undisclosed data collection across 268 apps?",
                     style={"fontSize": "0.85rem", "color": "#374151", "marginBottom": "4px"}),
            html.Div("Click a row to see SDK details. Sorted by # apps containing the SDK.",
                     style={"fontSize": "0.75rem", "color": "#6b7280"}),
        ], className="card", style={"marginBottom": "12px", "padding": "12px 16px"}),

        html.Div([
            # Table
            html.Div([
                dash_table.DataTable(
                    id="sdk-table",
                    data=records,
                    columns=[
                        {"name": "SDK Name", "id": "SDK Name"},
                        {"name": "Owner", "id": "Owner"},
                        {"name": "Category", "id": "Category"},
                        {"name": "# Apps", "id": "# Apps", "type": "numeric"},
                        {"name": "# Data Types", "id": "# Data Types", "type": "numeric"},
                        {"name": "# Apps w/ UNDECL", "id": "# Apps w/ UNDECL", "type": "numeric"},
                        {"name": "Yale Profile", "id": "Yale Profile"},
                    ],
                    style_cell={
                        "fontFamily": "Inter, system-ui, sans-serif",
                        "fontSize": "12px",
                        "padding": "8px 10px",
                        "overflow": "hidden",
                        "textOverflow": "ellipsis",
                        "maxWidth": "160px",
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
                        {"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"},
                        {
                            "if": {"filter_query": '{# Apps w/ UNDECL} > 0'},
                            "color": "#dc2626",
                        },
                        {
                            "if": {"column_id": "# Apps w/ UNDECL", "filter_query": '{# Apps w/ UNDECL} > 0'},
                            "fontWeight": "700",
                            "backgroundColor": "#fee2e2",
                        },
                    ],
                    sort_action="native",
                    filter_action="native",
                    page_size=20,
                    page_action="native",
                    row_selectable="single",
                    selected_rows=[0],
                    style_table={"overflowX": "auto"},
                ),
            ], className="data-table-container"),

            # Detail panel
            html.Div([
                html.Div("SDK Detail Panel", style={"fontWeight": "700", "fontSize": "0.8rem",
                                                     "textTransform": "uppercase",
                                                     "letterSpacing": "0.05em",
                                                     "color": "#6b7280",
                                                     "marginBottom": "12px"}),
                html.Div(id="sdk-detail-panel",
                         children=[html.Span("Select a row to see SDK details.",
                                             style={"color": "#9ca3af", "fontSize": "0.8rem"})]),
            ], className="node-inspector"),
        ], className="sidebar-layout"),

    ], className="tab-content")


def register_callbacks(app):
    @app.callback(
        Output("sdk-detail-panel", "children"),
        Input("sdk-table", "selected_rows"),
        Input("sdk-table", "data"),
    )
    def show_sdk_detail(selected_rows, data):
        if not selected_rows or not data:
            return html.Span("Select a row to see SDK details.",
                             style={"color": "#9ca3af", "fontSize": "0.8rem"})
        row = data[selected_rows[0]]
        sdk_name = row.get("SDK Name", "")
        df = _build_sdk_stats()
        matches = df[df["sdk_name"] == sdk_name]
        if matches.empty:
            return html.Span("No detail found.", style={"color": "#9ca3af"})
        sdk = matches.iloc[0]

        detail_rows = [
            ("Name", sdk["sdk_name"]),
            ("Owner", sdk["owner"] or "—"),
            ("Category", sdk["category"] or "—"),
            ("Purpose", sdk.get("canonical_purpose", "—") or "—"),
            ("# Apps Using It", str(int(sdk["n_apps"]))),
            ("# Apps w/ UNDECL", str(int(sdk["n_undecl"]))),
            ("Yale Privacy Lab Profile", "Yes" if sdk["has_yale"] else "No"),
            ("Tracker ID (Exodus)", str(sdk.get("tracker_id", "—"))),
        ]
        items = []
        for k, v in detail_rows:
            items.append(html.Div([
                html.Span(k, className="attr-key"),
                html.Span(str(v), className="attr-val"),
            ], className="attr-row"))

        # Data types collected
        dts = sdk.get("collects_data_types", [])
        if dts:
            items.append(html.Div([
                html.Div("Data Types Collected:", style={"fontWeight": "600",
                                                          "fontSize": "0.75rem",
                                                          "color": "#6b7280",
                                                          "marginTop": "10px",
                                                          "marginBottom": "4px"}),
                html.Div([
                    html.Span(dt.replace("_", " ").title(),
                              style={"background": "#dbeafe", "color": "#2563eb",
                                     "borderRadius": "4px", "padding": "2px 7px",
                                     "fontSize": "0.7rem", "marginRight": "4px",
                                     "display": "inline-block", "marginBottom": "4px"})
                    for dt in dts
                ]),
            ]))

        # Exodus link
        tid = sdk.get("tracker_id", "")
        if tid:
            items.append(html.Div(
                html.A(f"View on Exodus Privacy (tracker #{tid})",
                       href=f"https://reports.exodus-privacy.eu.org/en/trackers/{tid}/",
                       target="_blank",
                       style={"fontSize": "0.75rem", "color": "#2563eb"}),
                style={"marginTop": "10px"}
            ))

        return items
