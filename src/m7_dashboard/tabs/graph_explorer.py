"""Tab 2: Graph Explorer — interactive per-app heterogeneous graph viewer."""

import json
from functools import lru_cache
from dash import html, dcc, callback, Input, Output, State
import plotly.graph_objects as go
import networkx as nx
import torch

# --- Data loading -----------------------------------------------------------

_DATA_PATH = "/home/user/workspace/PolicyGraphAudit-RT/data/processed/fused_graphs_full.pt"

@lru_cache(maxsize=1)
def _load_graphs():
    return torch.load(_DATA_PATH, map_location="cpu", weights_only=False)

def _get_graph_map():
    """Return {app_id: graph_index} dict."""
    graphs = _load_graphs()
    return {g.app_id: i for i, g in enumerate(graphs)}

def _dropdown_options():
    graphs = _load_graphs()
    opts = [{"label": g.app_id, "value": g.app_id} for g in graphs]
    opts.sort(key=lambda x: x["label"])
    return opts

# --- Node type color map ---------------------------------------------------

NODE_COLORS = {
    "Policy": "#2563eb",         # blue
    "PolicySegment": "#93c5fd",  # light blue
    "DataType": "#0d9488",       # teal
    "Purpose": "#f97316",        # orange
    "ThirdParty": "#7c3aed",     # purple
    "PrivacyLabel": "#16a34a",   # green
    "App": "#374151",            # dark grey
    "SDK": "#dc2626",            # red
    "Endpoint": "#6b7280",       # grey
}

NODE_COLOR_DEFAULT = "#9ca3af"


# --- Graph rendering -------------------------------------------------------

def _build_graph_figure(app_id: str):
    graphs = _load_graphs()
    gmap = _get_graph_map()
    if app_id not in gmap:
        return go.Figure()

    g_data = graphs[gmap[app_id]]

    # Build networkx graph for layout
    G = nx.Graph()
    node_labels = {}  # global_id -> label str
    node_colors_list = []
    node_sizes = []
    node_hover = []
    node_type_list = []
    node_id_map = {}  # (ntype, local_idx) -> global_id
    gid = 0

    for ntype in g_data.node_types:
        ndata = g_data[ntype]
        n_nodes = ndata.x.shape[0] if hasattr(ndata, 'x') else 0
        nids = ndata.node_ids if hasattr(ndata, 'node_ids') else list(range(n_nodes))
        color = NODE_COLORS.get(ntype, NODE_COLOR_DEFAULT)
        for li in range(n_nodes):
            nid_str = nids[li] if isinstance(nids, list) else str(nids[li].item())
            short_label = nid_str.split("::")[-1][:30]
            G.add_node(gid, ntype=ntype, label=short_label, full_id=nid_str)
            node_labels[gid] = short_label
            node_colors_list.append(color)
            node_type_list.append(ntype)
            node_hover.append(f"<b>{ntype}</b><br>{nid_str[:60]}")
            node_id_map[(ntype, li)] = gid
            gid += 1

    # Add edges
    for etype in g_data.edge_types:
        src_type, rel, dst_type = etype
        edata = g_data[etype]
        if not hasattr(edata, 'edge_index'):
            continue
        ei = edata.edge_index
        for k in range(ei.shape[1]):
            s = node_id_map.get((src_type, ei[0, k].item()))
            d = node_id_map.get((dst_type, ei[1, k].item()))
            if s is not None and d is not None:
                G.add_edge(s, d, rel=rel)

    if G.number_of_nodes() == 0:
        return go.Figure()

    # Layout — spring with deterministic seed, fewer iterations for perf
    pos = nx.spring_layout(G, seed=42, iterations=30, k=1.5)

    # Degree-based sizes
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1

    # Build edge traces
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.7, color="#d1d5db"),
        hoverinfo="skip",
        name="edges",
    )

    # Node traces per type (for legend)
    traces = [edge_trace]
    nodes_by_type = {}
    for nid in G.nodes():
        nt = G.nodes[nid]["ntype"]
        nodes_by_type.setdefault(nt, []).append(nid)

    for nt, nids_list in nodes_by_type.items():
        color = NODE_COLORS.get(nt, NODE_COLOR_DEFAULT)
        nx_list = [pos[n][0] for n in nids_list]
        ny_list = [pos[n][1] for n in nids_list]
        sizes = [8 + 12 * (degrees.get(n, 0) / max(max_deg, 1)) for n in nids_list]
        hover = [G.nodes[n]["full_id"] for n in nids_list]
        customdata = [json.dumps({"nid": n, "ntype": nt, "full_id": G.nodes[n]["full_id"]})
                      for n in nids_list]
        traces.append(go.Scatter(
            x=nx_list, y=ny_list,
            mode="markers+text",
            marker=dict(size=sizes, color=color, line=dict(width=1, color="white")),
            text=[G.nodes[n]["label"][:15] for n in nids_list],
            textposition="top center",
            textfont=dict(size=8, color="#374151"),
            hovertext=hover,
            hoverinfo="text",
            customdata=customdata,
            name=nt,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, font=dict(size=10)),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        hovermode="closest",
        clickmode="event",
    )
    return fig


def _graph_stats(app_id: str):
    """Return list of stat items: # nodes by type, # edges by type."""
    graphs = _load_graphs()
    gmap = _get_graph_map()
    if app_id not in gmap:
        return []
    g = graphs[gmap[app_id]]
    rows = []
    for nt in g.node_types:
        n = g[nt].x.shape[0] if hasattr(g[nt], 'x') else 0
        rows.append(html.Span(
            f"{nt}: {n}",
            style={"background": "#f3f4f6", "borderRadius": "4px",
                   "padding": "2px 8px", "fontSize": "0.75rem",
                   "marginRight": "6px", "display": "inline-block", "marginBottom": "4px"},
        ))
    rows.append(html.Br())
    for et in g.edge_types:
        src, rel, dst = et
        ei = g[et].edge_index if hasattr(g[et], 'edge_index') else None
        n = ei.shape[1] if ei is not None else 0
        rows.append(html.Span(
            f"{rel}: {n}",
            style={"background": "#dbeafe", "borderRadius": "4px",
                   "padding": "2px 8px", "fontSize": "0.75rem", "color": "#2563eb",
                   "marginRight": "6px", "display": "inline-block", "marginBottom": "4px"},
        ))
    return rows


# --- Layout ----------------------------------------------------------------

def layout():
    opts = _dropdown_options()
    default_app = opts[1]["value"] if len(opts) > 1 else (opts[0]["value"] if opts else None)

    return html.Div([
        html.P("Graph Explorer", className="section-header"),
        html.Div([
            html.Div([
                html.Label("Select app:", style={"fontWeight": "600", "fontSize": "0.8rem",
                                                  "marginBottom": "6px", "display": "block"}),
                dcc.Dropdown(
                    id="ge-app-picker",
                    options=opts,
                    value=default_app,
                    placeholder="Search for an app...",
                    style={"fontSize": "0.85rem"},
                    clearable=False,
                ),
            ], className="card", style={"marginBottom": "12px"}),
        ]),

        html.Div([
            # Graph area
            html.Div([
                html.Div(id="ge-stats-row",
                         style={"marginBottom": "8px", "minHeight": "28px"}),
                dcc.Graph(
                    id="ge-graph",
                    config={"displayModeBar": True, "modeBarButtonsToRemove": ["toImage"]},
                    style={"height": "480px"},
                ),
            ], className="card"),

            # Node inspector
            html.Div([
                html.Div("Node Inspector", style={"fontWeight": "700", "fontSize": "0.8rem",
                                                   "marginBottom": "10px",
                                                   "textTransform": "uppercase",
                                                   "letterSpacing": "0.05em",
                                                   "color": "#6b7280"}),
                html.Div("Click a node to inspect its attributes.",
                         id="ge-node-inspector",
                         style={"fontSize": "0.8rem", "color": "#9ca3af"}),
            ], className="node-inspector"),
        ], className="sidebar-layout"),

        # Legend
        html.Div([
            html.Div([
                html.Div(style={"width": "10px", "height": "10px", "borderRadius": "50%",
                                 "background": color, "display": "inline-block",
                                 "marginRight": "5px"}),
                html.Span(ntype, style={"fontSize": "0.75rem", "color": "#374151"}),
            ], style={"display": "inline-flex", "alignItems": "center",
                       "marginRight": "14px", "marginBottom": "4px"})
            for ntype, color in NODE_COLORS.items()
        ], style={"padding": "10px 0"}),

    ], className="tab-content")


# --- Callbacks -------------------------------------------------------------

def register_callbacks(app):
    @app.callback(
        Output("ge-graph", "figure"),
        Output("ge-stats-row", "children"),
        Input("ge-app-picker", "value"),
    )
    def update_graph(app_id):
        if not app_id:
            return go.Figure(), []
        return _build_graph_figure(app_id), _graph_stats(app_id)

    @app.callback(
        Output("ge-node-inspector", "children"),
        Input("ge-graph", "clickData"),
    )
    def inspect_node(click_data):
        if not click_data:
            return "Click a node to inspect its attributes."
        points = click_data.get("points", [])
        if not points:
            return "Click a node to inspect its attributes."
        pt = points[0]
        custom = pt.get("customdata")
        if not custom:
            return "Click a node to inspect its attributes."
        try:
            info = json.loads(custom)
        except Exception:
            return str(custom)
        rows = [
            html.Div([
                html.Span("Type", className="attr-key"),
                html.Span(info.get("ntype", "—"), className="attr-val"),
            ], className="attr-row"),
            html.Div([
                html.Span("ID", className="attr-key"),
                html.Span(info.get("full_id", "—"), className="attr-val"),
            ], className="attr-row"),
        ]
        # Neighbours
        graphs = _load_graphs()
        gmap = _get_graph_map()
        ntype = info.get("ntype", "")
        full_id = info.get("full_id", "")
        app_id = full_id.split("::")[1] if "::" in full_id and ntype == "App" else None

        rows.append(html.Div(
            html.Span("(Select a different node to see neighbours)",
                      style={"color": "#9ca3af", "fontSize": "0.75rem"}),
            style={"paddingTop": "8px"}
        ))
        return rows
