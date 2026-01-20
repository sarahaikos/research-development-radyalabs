import itertools
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "SFIA_Full_Framework.csv"

TAGS = ["DATA", "SOFT", "DEVO", "CYBR", "CLOU"]

def load_data(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

def extract_tags(tags_value: str) -> set[str]:
    if not isinstance(tags_value, str) or not tags_value.strip():
        return set()
    tags = set()
    for raw_tag in tags_value.split(","):
        tag = raw_tag.strip().upper()
        if "_" in tag:
            tag = tag.split("_", 1)[0]
            if tag in TAGS:
                tags.add(tag)
    return tags


def analyze_code_tag_mapping(df: pd.DataFrame) -> dict:
    # map each code to the set of tags it appears with
    code_to_tags: dict[str, set[str]] = defaultdict(set)
    
    for _, row in df.iterrows():
        code = row.get("Code", "")
        tags = row.get("Tags", "")
        if not code or pd.isna(code) or not isinstance(code, str):
            continue  
        code = code.strip()
        if not code:
            continue
        tags = extract_tags(tags)
        if tags:
            code_to_tags[code].update(tags)
    
    # define exclusive and inclusive codes
    exclusive_codes: dict[str, list[str]] = {fam: [] for fam in TAGS}
    inclusive_codes: dict[tuple, list[str]] = defaultdict(list)
    
    for code, tags_set in code_to_tags.items():
        tags_list = sorted(list(tags_set))
        if len(tags_list) == 1:
            exclusive_codes[tags_list[0]].append(code)
        else:
            key = tuple(tags_list)
            inclusive_codes[key].append(code)
    
    return exclusive_codes, dict(inclusive_codes)


def print_code_analysis(exclusive_codes: dict, inclusive_codes: dict) -> None:
    lines: list[str] = []

    def add(line: str = "") -> None:
        print(line)
        lines.append(line)

    add("SFIA Code Analysis by Tags")
    add("=" * 80)
    
    for fam in TAGS:
        codes = sorted(exclusive_codes.get(fam, []))
        add(f"\n{fam} ONLY ({len(codes)} codes):")
        if codes:
            add(f"  {', '.join(codes)}")
        else:
            add("  (none)")
    
    sorted_inclusive = sorted(
        inclusive_codes.items(),
        key=lambda x: (len(x[0]), x[0])
    )
    
    for tags_tuple, codes in sorted_inclusive:
        tags_str = " + ".join(tags_tuple)
        add(f"\n{tags_str} ({len(codes)} codes):")
        add(f"  {', '.join(sorted(codes))}")

    report_path = BASE_DIR / "sfia_code_analysis.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def create_inclusive_graph(exclusive_codes: dict, inclusive_codes: dict) -> None:
    G = nx.Graph()

    tag_colors = {
        "DATA": "#084594",
        "SOFT": "#2171b5",
        "DEVO": "#4292c6",
        "CYBR": "#6baed6",
        "CLOU": "#9ecae1",
    }
    for tag in TAGS:
        G.add_node(tag, node_type="tag", color=tag_colors[tag], size=1200)

    # combination nodes (only for inclusive codes: length > 1)
    combo_label_map: dict[tuple[str, ...], str] = {}
    for tags_tuple, codes in inclusive_codes.items():
        if len(tags_tuple) <= 1:
            continue
        combo_label = "+".join(tags_tuple)
        combo_label_map[tags_tuple] = combo_label
        count = len(codes)
        G.add_node(combo_label, node_type="combo", size=400 + 40 * count, count=count)
        for tag in tags_tuple:
            if tag in TAGS:
                G.add_edge(tag, combo_label, weight=count, edge_type="tag_combo")

        # add inclusive code nodes for this combination (layer 4)
        for code in codes:
            if not G.has_node(code):
                G.add_node(code, node_type="code_inclusive", color="#98b3d4", size=260)
            G.add_edge(combo_label, code, weight=1, edge_type="combo_code")

    # exclusive code group nodes
    exclusive_group_nodes: list[str] = []
    for tag, codes in exclusive_codes.items():
        if codes:
            codes_sorted = sorted(codes)
            count = len(codes_sorted)
            group_label = f"{tag}_exclusive"
            exclusive_group_nodes.append(group_label)
            G.add_node(
                group_label,
                node_type="exclusive_group",
                color="#bed4ba",
                size=600 + 40 * count,
                count=count,
                tag=tag,
                codes=codes_sorted,
            )
            G.add_edge(tag, group_label, weight=count, edge_type="tag_exclusive_group")

    # 4-layer layout:
    # x = 0.0 → tags
    # x = 1.0 → exclusive codes
    # x = 2.0 → combinations
    # x = 3.0 → inclusive codes
    tag_nodes = [n for n, attrs in G.nodes(data=True) if attrs.get("node_type") == "tag"]
    combo_nodes = [n for n, attrs in G.nodes(data=True) if attrs.get("node_type") == "combo"]
    code_inclusive_nodes = [n for n, attrs in G.nodes(data=True) if attrs.get("node_type") == "code_inclusive"]
    exclusive_group_nodes = [n for n, attrs in G.nodes(data=True) if attrs.get("node_type") == "exclusive_group"]

    if not G.nodes or (not combo_nodes and not code_inclusive_nodes and not exclusive_group_nodes):
        print("No codes to plot.")
        return

    def layered_positions(nodes: list[str], x_value: float, max_height: float = 10.0) -> dict[str, tuple[float, float]]:
        if not nodes:
            return {}
        if len(nodes) == 1:
            return {nodes[0]: (x_value, max_height / 2)}
        step = max_height / (len(nodes) - 1) if len(nodes) > 1 else 0
        return {node: (x_value, i * step) for i, node in enumerate(sorted(nodes))}

    max_nodes = max(len(tag_nodes), len(combo_nodes), len(code_inclusive_nodes), len(exclusive_group_nodes), 1)
    max_height = max(8.0, max_nodes * 0.5)
    
    pos = {}
    pos.update(layered_positions(tag_nodes, x_value=0.0, max_height=max_height))
    pos.update(layered_positions(exclusive_group_nodes, x_value=1.0, max_height=max_height))
    pos.update(layered_positions(combo_nodes, x_value=2.0, max_height=max_height))
    pos.update(layered_positions(code_inclusive_nodes, x_value=3.0, max_height=max_height))

    fig_height = max(9, max_height * 0.8)
    plt.figure(figsize=(16, fig_height))

    # edges
    tag_combo_edges = [
        (u, v, d)
        for u, v, d in G.edges(data=True)
        if d.get("edge_type") == "tag_combo"
    ]
    combo_code_edges = [
        (u, v, d)
        for u, v, d in G.edges(data=True)
        if d.get("edge_type") == "combo_code"
    ]
    tag_exclusive_group_edges = [
        (u, v, d)
        for u, v, d in G.edges(data=True)
        if d.get("edge_type") == "tag_exclusive_group"
    ]

    if tag_combo_edges:
        widths = [0.5 + d.get("weight", 1) * 0.3 for _, _, d in tag_combo_edges]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v) for u, v, _ in tag_combo_edges],
            width=widths,
            edge_color="#4292c6",
            alpha=0.7,
        )

    if combo_code_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v) for u, v, _ in combo_code_edges],
            width=0.8,
            edge_color="#9ecae1",
            alpha=0.6,
        )

    if tag_exclusive_group_edges:
        widths = [0.8 + d.get("weight", 1) * 0.2 for _, _, d in tag_exclusive_group_edges]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v) for u, v, _ in tag_exclusive_group_edges],
            width=widths,
            edge_color="#bed4ba", 
            alpha=0.6,
            style="dashed",
        )
    
    # nodes
    if tag_nodes:
        tag_colors_list = [G.nodes[n]["color"] for n in tag_nodes]
        tag_sizes = [G.nodes[n]["size"] for n in tag_nodes]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=tag_nodes,
            node_color=tag_colors_list,
            node_size=tag_sizes,
            alpha=0.95,
            node_shape="s",
        )
        nx.draw_networkx_labels(
            G,
            pos,
            {n: n for n in tag_nodes},
            font_size=14,
            font_weight="bold",
        )

    if combo_nodes:
        combo_sizes = [G.nodes[n]["size"] for n in combo_nodes]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=combo_nodes,
            node_color="#c6dbef",
            node_size=combo_sizes,
            alpha=0.9,
        )
        combo_labels = {
            n: f"{n}\n({G.nodes[n].get('count', 0)} codes)" for n in combo_nodes
        }
        nx.draw_networkx_labels(
            G,
            pos,
            combo_labels,
            font_size=8,
            font_weight="normal",
        )

    if code_inclusive_nodes:
        code_inclusive_colors = [G.nodes[n].get("color", "#98b3d4") for n in code_inclusive_nodes]
        code_inclusive_sizes = [G.nodes[n].get("size", 220) for n in code_inclusive_nodes]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=code_inclusive_nodes,
            node_color=code_inclusive_colors,
            node_size=code_inclusive_sizes,
            alpha=0.8,
        )
        nx.draw_networkx_labels(
            G,
            pos,
            {n: n for n in code_inclusive_nodes},
            font_size=6,
            font_color="black",
        )

    if exclusive_group_nodes:
        exclusive_group_colors = [G.nodes[n].get("color", "#bed4ba") for n in exclusive_group_nodes]
        exclusive_group_sizes = [G.nodes[n].get("size", 600) for n in exclusive_group_nodes]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=exclusive_group_nodes,
            node_color=exclusive_group_colors,
            node_size=exclusive_group_sizes,
            alpha=0.85,
        )
        exclusive_group_labels = {}
        for n in exclusive_group_nodes:
            tag = G.nodes[n].get("tag", "")
            count = G.nodes[n].get("count", 0)
            codes_list = G.nodes[n].get("codes", [])
            codes_str = ", ".join(codes_list)
            exclusive_group_labels[n] = f"{tag} Exclusive\n({count} codes)\n{codes_str}"

        nx.draw_networkx_labels(
            G,
            pos,
            exclusive_group_labels,
            font_size=7,
            font_weight="bold",
            font_color="black",
        )

    plt.title(
        "SFIA Code Relationships (4-Layer: Tags → Exclusive Code Groups → Combinations → Inclusive Codes)\n"
        "Layer1: Tags | Layer2: Exclusive Code Groups (one circle per tag) | Layer3: Tag Combinations | Layer4: Inclusive Codes\n"
        "Dark blue = inclusive codes (via combinations) | Light blue circles = exclusive code groups (direct to tags, dashed edges)",
        fontsize=11,
        fontweight="bold",
    )
    plt.axis("off")
    plt.tight_layout()

    graph_path = BASE_DIR / "sfia_inclusive_graph.png"
    plt.savefig(graph_path, dpi=300)
    plt.show()

def build_code_tag_matrix(df: pd.DataFrame) -> pd.DataFrame:
    from collections import defaultdict

    code_to_tags: dict[str, set[str]] = defaultdict(set)

    for _, row in df.iterrows():
        code_raw = row.get("Code", "")
        code = str(code_raw).strip()
        if not code:
            continue
        tags = extract_tags(row.get("Tags", ""))
        if tags:
            code_to_tags[code].update(tags)

    codes = sorted(code_to_tags.keys())
    matrix = pd.DataFrame(0, index=codes, columns=TAGS, dtype=int)

    for code, fams in code_to_tags.items():
        for fam in fams:
            if fam in matrix.columns:
                matrix.loc[code, fam] = 1

    matrix_path = BASE_DIR / "sfia_code_tag_matrix.csv"
    matrix.to_csv(matrix_path, index_label="Code")

    return matrix


def build_code_category_tag_summary(df: pd.DataFrame) -> pd.DataFrame:
    from collections import defaultdict

    info_by_code: dict[str, dict] = {}
    tags_by_code: dict[str, set[str]] = defaultdict(set)

    for _, row in df.iterrows():
        code_raw = row.get("Code", "")
        code = str(code_raw).strip()
        if not code:
            continue

        if code not in info_by_code:
            info_by_code[code] = {
                "Category": row.get("Category", ""),
                "Subcategory": row.get("Subcategory", ""),
                "Skill": row.get("Skill", ""),
            }

        fams = extract_tags(row.get("Tags", ""))
        if fams:
            tags_by_code[code].update(fams)

    records: list[dict] = []
    for code, meta in info_by_code.items():
        fams = sorted(tags_by_code.get(code, set()))
        records.append(
            {
                "Code": code,
                "Category": meta.get("Category", ""),
                "Subcategory": meta.get("Subcategory", ""),
                "Skill": meta.get("Skill", ""),
                "tags": ", ".join(fams),
            }
        )

    summary_df = pd.DataFrame(records).sort_values(["Category", "Subcategory", "Code"])

    summary_path = BASE_DIR / "sfia_code_category_tag_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return summary_df


def create_category_tag_heatmap(summary_df: pd.DataFrame) -> None:
    tmp = summary_df.copy()
    tmp["tagsList"] = tmp["tags"].apply(
        lambda s: [f.strip() for f in s.split(",")] if isinstance(s, str) and s.strip() else []
    )
    tmp = tmp.explode("tagsList")
    tmp = tmp[tmp["tagsList"].isin(TAGS)]

    if tmp.empty:
        print("No category/tag data to plot.")
        return

    counts = (
        tmp.groupby(["Category", "tagsList"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=TAGS, fill_value=0)
    )

    plt.figure(figsize=(10, max(4, 0.4 * len(counts))))
    sns.heatmap(
        counts,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "Number of codes"},
        linewidths=0.5,
    )
    plt.title("Codes per Category and Tag", fontsize=12, fontweight="bold")
    plt.xlabel("tag")
    plt.ylabel("Category")
    plt.tight_layout()

    cat_heatmap_path = BASE_DIR / "sfia_category_tag_heatmap.png"
    plt.savefig(cat_heatmap_path, dpi=300)
    plt.show()


def main() -> None:
    df = load_data()
    df = df[df["Code"].notna() & (df["Code"].astype(str).str.strip() != "")]
    
    exclusive_codes, inclusive_codes = analyze_code_tag_mapping(df)
    print_code_analysis(exclusive_codes, inclusive_codes)
    code_tag_matrix = build_code_tag_matrix(df)
    code_category_summary = build_code_category_tag_summary(df)
    create_category_tag_heatmap(code_category_summary)    
    create_inclusive_graph(exclusive_codes, inclusive_codes)


if __name__ == "__main__":
    main()
