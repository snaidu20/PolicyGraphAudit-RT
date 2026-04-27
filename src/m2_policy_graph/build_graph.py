"""
build_graph.py — Constructs a typed NetworkX MultiDiGraph from a privacy policy.

Graph schema (see configs/schema.yaml):
  Nodes:
    - Policy          (1 per document)
    - PolicySegment   (1 per qualifying paragraph)
    - DataType        (canonical vocabulary nodes, shared across all policies)
    - Purpose         (canonical vocabulary nodes)
    - ThirdParty      (company name nodes)

  Edges:
    - (Policy)        -[HAS_SEGMENT]->  (PolicySegment)
    - (PolicySegment) -[MENTIONS]->     (DataType)
    - (PolicySegment) -[FOR_PURPOSE]->  (Purpose)
    - (PolicySegment) -[SHARED_WITH]->  (ThirdParty)   [only for 3rd-party segments]

Public API:
  build_policy_graph(policy_text, policy_id) -> (MultiDiGraph, stats_dict)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from m2_policy_graph.classifier import OPP115Classifier, _get_singleton
from m2_policy_graph.extract import (
    extract_data_types,
    extract_purposes,
    extract_third_parties,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# OPP-115 category that triggers ThirdParty edge creation
_THIRD_PARTY_CATEGORY = "Third Party Sharing/Collection"

# Probability threshold for secondary category match
_THIRD_PARTY_CAT_PROB_THRESHOLD = 0.25

# Regex for sharing-language heuristic (used as fallback if classifier uncertain)
_SHARING_PATTERN = re.compile(
    r"\b(shar(?:e|ed|ing)|disclos(?:e|ed|ing)|transfer(?:red|ring)?|provid(?:e|ed|ing) to"
    r"|send(?:ing)? to|with third.{0,6}part|partner|vendor|provider|processor"
    r"|third.party|use[sd]?\s+\w+\s+to|such as|includ(?:e|ing)|tools? such|platform such"
    r"|social network|analytics tool|advertis|may share|we share|we use)\b",
    re.IGNORECASE,
)

# Minimum paragraph length in characters
_MIN_PARA_LEN = 50


# ---------------------------------------------------------------------------
# Node / edge type constants (mirroring schema.yaml names)
# ---------------------------------------------------------------------------

NT_POLICY = "Policy"
NT_SEGMENT = "PolicySegment"
NT_DATA_TYPE = "DataType"
NT_PURPOSE = "Purpose"
NT_THIRD_PARTY = "ThirdParty"

ET_HAS_SEGMENT = "HAS_SEGMENT"
ET_MENTIONS = "MENTIONS"
ET_FOR_PURPOSE = "FOR_PURPOSE"
ET_SHARED_WITH = "SHARED_WITH"


# ---------------------------------------------------------------------------
# Helper — split policy into paragraphs
# ---------------------------------------------------------------------------

def _split_paragraphs(text: str, min_len: int = _MIN_PARA_LEN) -> List[str]:
    """
    Split policy text into paragraphs on double-newlines.
    Drop empty paragraphs and those shorter than min_len characters.
    """
    raw = re.split(r"\n{2,}", text)
    paras = []
    for p in raw:
        p_clean = p.strip()
        if p_clean and len(p_clean) >= min_len:
            paras.append(p_clean)
    return paras


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_policy_graph(
    policy_text: str,
    policy_id: str,
    classifier: Optional[OPP115Classifier] = None,
) -> Tuple[nx.MultiDiGraph, Dict[str, Any]]:
    """
    Build a typed MultiDiGraph for a single privacy policy.

    Parameters
    ----------
    policy_text : str
        Full text of the privacy policy (markdown or plain text).
    policy_id : str
        Unique identifier for this policy (e.g. domain name or file stem).
    classifier : OPP115Classifier, optional
        Pre-loaded classifier.  Loaded from disk on first call if None.

    Returns
    -------
    G : nx.MultiDiGraph
        Typed graph with node_type attributes on every node.
    stats : dict
        {n_segments, n_data_types, n_purposes, n_third_parties,
         edge_counts: {HAS_SEGMENT, MENTIONS, FOR_PURPOSE, SHARED_WITH}}
    """
    if classifier is None:
        classifier = _get_singleton()

    G = nx.MultiDiGraph()

    # ------------------------------------------------------------------
    # Policy root node
    # ------------------------------------------------------------------
    policy_node_id = f"policy:{policy_id}"
    G.add_node(
        policy_node_id,
        node_type=NT_POLICY,
        doc_id=policy_id,
        length_chars=len(policy_text),
    )

    # ------------------------------------------------------------------
    # Paragraph splitting
    # ------------------------------------------------------------------
    paragraphs = _split_paragraphs(policy_text, min_len=_MIN_PARA_LEN)

    # Counters for stats
    data_type_nodes: set[str] = set()
    purpose_nodes: set[str] = set()
    third_party_nodes: set[str] = set()

    edge_counts: Dict[str, int] = {
        ET_HAS_SEGMENT: 0,
        ET_MENTIONS: 0,
        ET_FOR_PURPOSE: 0,
        ET_SHARED_WITH: 0,
    }

    for pos, para in enumerate(paragraphs):
        # ----------------------------------------------------------------
        # Create PolicySegment node
        # ----------------------------------------------------------------
        seg_id = f"segment:{policy_id}:{pos}"

        # Run OPP-115 classifier
        cat_probs = classifier.predict_categories(para)
        top_cat = cat_probs[0][0] if cat_probs else "Other"
        top_cat_prob = cat_probs[0][1] if cat_probs else 0.0

        G.add_node(
            seg_id,
            node_type=NT_SEGMENT,
            policy_id=policy_id,
            position=pos,
            text_snippet=para[:200],
            opp115_category=top_cat,
            opp115_prob=round(top_cat_prob, 4),
            opp115_all=[(c, round(p, 4)) for c, p in cat_probs],
        )

        # Policy → Segment edge
        G.add_edge(
            policy_node_id,
            seg_id,
            edge_type=ET_HAS_SEGMENT,
        )
        edge_counts[ET_HAS_SEGMENT] += 1

        # ----------------------------------------------------------------
        # Extract data types
        # ----------------------------------------------------------------
        data_types = extract_data_types(para)
        for dt in data_types:
            dt_node_id = f"datatype:{dt}"
            if not G.has_node(dt_node_id):
                G.add_node(dt_node_id, node_type=NT_DATA_TYPE, name=dt)
            G.add_edge(seg_id, dt_node_id, edge_type=ET_MENTIONS)
            edge_counts[ET_MENTIONS] += 1
            data_type_nodes.add(dt_node_id)

        # ----------------------------------------------------------------
        # Extract purposes
        # ----------------------------------------------------------------
        purposes = extract_purposes(para)
        for pu in purposes:
            pu_node_id = f"purpose:{pu}"
            if not G.has_node(pu_node_id):
                G.add_node(pu_node_id, node_type=NT_PURPOSE, name=pu)
            G.add_edge(seg_id, pu_node_id, edge_type=ET_FOR_PURPOSE)
            edge_counts[ET_FOR_PURPOSE] += 1
            purpose_nodes.add(pu_node_id)

        # ----------------------------------------------------------------
        # Extract third parties
        # Add SHARED_WITH edge if:
        #   (a) top category is "Third Party Sharing/Collection", OR
        #   (b) any category in top-2 includes it with prob >= threshold, OR
        #   (c) the segment contains explicit sharing language (heuristic fallback)
        # ----------------------------------------------------------------
        third_parties = extract_third_parties(para)
        if third_parties:
            # Check OPP-115 third-party signal
            is_tp_by_classifier = top_cat == _THIRD_PARTY_CATEGORY or any(
                cat == _THIRD_PARTY_CATEGORY and prob >= _THIRD_PARTY_CAT_PROB_THRESHOLD
                for cat, prob in cat_probs[:3]
            )
            # Heuristic fallback: explicit sharing language
            is_tp_by_heuristic = bool(_SHARING_PATTERN.search(para))
            is_third_party_seg = is_tp_by_classifier or is_tp_by_heuristic

            for tp in third_parties:
                tp_node_id = f"thirdparty:{tp.lower()}"
                if not G.has_node(tp_node_id):
                    G.add_node(tp_node_id, node_type=NT_THIRD_PARTY, name=tp)
                third_party_nodes.add(tp_node_id)
                if is_third_party_seg:
                    G.add_edge(seg_id, tp_node_id, edge_type=ET_SHARED_WITH)
                    edge_counts[ET_SHARED_WITH] += 1

    # ------------------------------------------------------------------
    # Build stats dict
    # ------------------------------------------------------------------
    stats: Dict[str, Any] = {
        "n_segments": len(paragraphs),
        "n_data_types": len(data_type_nodes),
        "n_purposes": len(purpose_nodes),
        "n_third_parties": len(third_party_nodes),
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "edge_counts": edge_counts,
    }

    return G, stats


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sample = """
We collect your email address, phone number, and name when you register.

We use GPS location and device ID for analytics and app functionality.

Your data may be shared with Google Analytics and Facebook for advertising
and marketing purposes. We use Firebase Crashlytics for crash logs and
diagnostics.

You can opt out of receiving promotional emails at any time.
    """

    G, stats = build_policy_graph(sample, "demo_policy")
    print("Stats:", stats)
    print("Nodes:")
    for node, data in G.nodes(data=True):
        print(f"  {node}  [{data['node_type']}]")
    print("Edges:")
    for u, v, data in G.edges(data=True):
        print(f"  {u} -[{data['edge_type']}]-> {v}")
