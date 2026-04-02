import os
import tempfile

import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components


# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title="Plant Knowledge Graph",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌿 Plant Trait & Taxonomy Knowledge Graph")
st.markdown(
    """
Explore the new knowledge graph structure:

**Genus → Species → SpeciesOrgan → Organ / TraitCategory / TraitValue**

Use the cascading filters in the sidebar to focus on a smaller and more interpretable subgraph.
"""
)


# ==========================================
# 2. File Paths
# ==========================================
NODES_FILE = "KG_nodes_with_organs.csv"
EDGES_FILE = "KG_edges_with_organs.csv"


# ==========================================
# 3. Data Loading & Caching
# ==========================================
@st.cache_data
def load_data(nodes_path, edges_path):
    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"Nodes file not found: {nodes_path}")
    if not os.path.exists(edges_path):
        raise FileNotFoundError(f"Edges file not found: {edges_path}")

    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    # Standardize column types
    for col in ["node_id", "node_type", "display_name"]:
        if col in nodes_df.columns:
            nodes_df[col] = nodes_df[col].astype(str)

    for col in ["source", "target", "relation"]:
        if col in edges_df.columns:
            edges_df[col] = edges_df[col].astype(str)

    # Fill missing optional columns
    if "display_name" not in nodes_df.columns:
        nodes_df["display_name"] = nodes_df["node_id"]

    if "value_type" not in nodes_df.columns:
        nodes_df["value_type"] = ""

    # Global degree for node sizing
    temp_G = nx.from_pandas_edgelist(
        edges_df, source="source", target="target", create_using=nx.DiGraph()
    )
    global_degree = dict(temp_G.degree())

    node_type_dict = pd.Series(nodes_df["node_type"].values, index=nodes_df["node_id"]).to_dict()
    display_name_dict = pd.Series(nodes_df["display_name"].values, index=nodes_df["node_id"]).to_dict()
    value_type_dict = pd.Series(nodes_df["value_type"].fillna("").values, index=nodes_df["node_id"]).to_dict()

    def get_nodes_by_type(node_type):
        return sorted(
            nodes_df.loc[nodes_df["node_type"] == node_type, "node_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    all_genera = get_nodes_by_type("Genus")
    all_species = get_nodes_by_type("Species")
    all_organs = get_nodes_by_type("Organ")
    all_species_organs = get_nodes_by_type("SpeciesOrgan")
    all_trait_categories = get_nodes_by_type("TraitCategory")
    all_trait_values = get_nodes_by_type("TraitValue")

    return (
        nodes_df,
        edges_df,
        global_degree,
        node_type_dict,
        display_name_dict,
        value_type_dict,
        all_genera,
        all_species,
        all_organs,
        all_species_organs,
        all_trait_categories,
        all_trait_values,
    )


try:
    (
        df_nodes,
        df_edges,
        global_degree_dict,
        node_type_dict,
        display_name_dict,
        value_type_dict,
        all_genera,
        all_species,
        all_organs,
        all_species_organs,
        all_trait_categories,
        all_trait_values,
    ) = load_data(NODES_FILE, EDGES_FILE)

except Exception as e:
    st.error(f"Failed to load graph files: {e}")
    st.stop()


# ==========================================
# 4. Helper Functions
# ==========================================
def edge_targets(edges_df, sources, relations):
    """Return targets reached from sources via given relation(s)."""
    if not sources:
        return set()
    if isinstance(relations, str):
        relations = [relations]
    mask = edges_df["source"].isin(sources) & edges_df["relation"].isin(relations)
    return set(edges_df.loc[mask, "target"].astype(str))


def edge_sources(edges_df, targets, relations):
    """Return sources that point to targets via given relation(s)."""
    if not targets:
        return set()
    if isinstance(relations, str):
        relations = [relations]
    mask = edges_df["target"].isin(targets) & edges_df["relation"].isin(relations)
    return set(edges_df.loc[mask, "source"].astype(str))


def safe_sorted(items):
    return sorted(list(set(items)))


def filter_edges_between_nodes(edges_df, active_nodes):
    mask = edges_df["source"].isin(active_nodes) & edges_df["target"].isin(active_nodes)
    return edges_df.loc[mask].copy()


# ==========================================
# 5. Cascading Sidebar Filtering UI
# ==========================================
with st.sidebar:
    st.header("🔍 Cascading Filters")
    st.markdown("Selections constrain downstream options based on the new KG structure.")

    # ----- LEVEL 1: GENUS -----
    sel_genera = st.multiselect(
        "1. Select Genus",
        options=all_genera,
        default=[]
    )

    # Available species from selected genera
    if sel_genera:
        available_species = safe_sorted(edge_targets(df_edges, set(sel_genera), "has_species"))
    else:
        available_species = all_species

    # ----- LEVEL 2: SPECIES -----
    sel_species = st.multiselect(
        "2. Select Species",
        options=available_species,
        default=[]
    )

    # Base species set for downstream filtering
    species_scope = set(sel_species) if sel_species else set(available_species)

    # SpeciesOrgan candidates from species
    available_species_organs = safe_sorted(edge_targets(df_edges, species_scope, "has_organ"))

    # Available organs from speciesorgans
    if available_species_organs:
        available_organs = safe_sorted(edge_targets(df_edges, set(available_species_organs), "instance_of"))
    else:
        available_organs = all_organs

    # ----- LEVEL 3: ORGAN -----
    sel_organs = st.multiselect(
        "3. Select Organ",
        options=available_organs,
        default=[]
    )

    # Constrain speciesorgan nodes by organ if selected
    speciesorgan_scope = set(available_species_organs)
    if sel_organs:
        so_from_organs = edge_sources(df_edges, set(sel_organs), "instance_of")
        speciesorgan_scope = speciesorgan_scope.intersection(so_from_organs)

    # Available trait categories from speciesorgan scope
    if speciesorgan_scope:
        available_trait_categories = safe_sorted(edge_targets(df_edges, speciesorgan_scope, "has_trait_category"))
    else:
        available_trait_categories = all_trait_categories

    # ----- LEVEL 4: TRAIT CATEGORY -----
    sel_trait_categories = st.multiselect(
        "4. Select Trait Category",
        options=available_trait_categories,
        default=[]
    )

    # Constrain speciesorgan nodes by trait category if selected
    if sel_trait_categories:
        so_from_categories = edge_sources(df_edges, set(sel_trait_categories), "has_trait_category")
        speciesorgan_scope = speciesorgan_scope.intersection(so_from_categories)

    # Available trait values from speciesorgan scope
    if speciesorgan_scope:
        available_trait_values = safe_sorted(edge_targets(df_edges, speciesorgan_scope, "has_trait_value"))
    else:
        available_trait_values = all_trait_values

    # If categories are selected, only keep values linked to those categories
    if sel_trait_categories:
        values_for_categories = edge_sources(df_edges, set(sel_trait_categories), "value_of_category")
        available_trait_values = safe_sorted(set(available_trait_values).intersection(values_for_categories))

    # ----- LEVEL 5: TRAIT VALUE -----
    sel_trait_values = st.multiselect(
        "5. Select Trait Value",
        options=available_trait_values,
        default=[]
    )

    st.markdown("---")
    st.header("⚙️ View Settings")

    show_edge_labels = st.checkbox("Show Edge Labels", value=False)
    show_value_nodes = st.checkbox("Show TraitValue Nodes", value=True)
    show_trait_category_nodes = st.checkbox("Show TraitCategory Nodes", value=True)
    show_speciesorgan_nodes = st.checkbox("Show SpeciesOrgan Nodes", value=True)

    physics_enabled = st.checkbox("Enable Physics Layout", value=True)

    max_edges = st.slider(
        "Maximum edges to render",
        min_value=300,
        max_value=6000,
        value=2500,
        step=100
    )


# ==========================================
# 6. Build Filtered Subgraph
# ==========================================
has_any_selection = any([
    len(sel_genera) > 0,
    len(sel_species) > 0,
    len(sel_organs) > 0,
    len(sel_trait_categories) > 0,
    len(sel_trait_values) > 0,
])

if not has_any_selection:
    st.info("👈 Please select at least one filter from the sidebar to display a subgraph.")
    st.stop()

# ----- Step A: species constraint from genus -----
if sel_genera:
    genus_species = edge_targets(df_edges, set(sel_genera), "has_species")
else:
    genus_species = set(all_species)

if sel_species:
    active_species = genus_species.intersection(set(sel_species))
else:
    active_species = genus_species

# ----- Step B: speciesorgan constraint from species -----
active_speciesorgans = edge_targets(df_edges, active_species, "has_organ")

# constrain by selected organs
if sel_organs:
    so_from_organs = edge_sources(df_edges, set(sel_organs), "instance_of")
    active_speciesorgans = active_speciesorgans.intersection(so_from_organs)

# constrain by selected trait categories
if sel_trait_categories:
    so_from_categories = edge_sources(df_edges, set(sel_trait_categories), "has_trait_category")
    active_speciesorgans = active_speciesorgans.intersection(so_from_categories)

# constrain by selected trait values
if sel_trait_values:
    so_from_values = edge_sources(df_edges, set(sel_trait_values), "has_trait_value")
    active_speciesorgans = active_speciesorgans.intersection(so_from_values)

# Recompute active species from surviving speciesorgan nodes
if active_speciesorgans:
    active_species = edge_sources(df_edges, active_speciesorgans, "has_organ")
else:
    active_species = set()

# Recompute active genera from surviving species
if active_species:
    active_genera = edge_sources(df_edges, active_species, "has_species")
else:
    active_genera = set()

# If genus was selected, keep only selected genera
if sel_genera:
    active_genera = active_genera.intersection(set(sel_genera))

# ----- Step C: derive organs / categories / values from surviving speciesorgans -----
active_organs = edge_targets(df_edges, active_speciesorgans, "instance_of")
active_trait_categories = edge_targets(df_edges, active_speciesorgans, "has_trait_category")
active_trait_values = edge_targets(df_edges, active_speciesorgans, "has_trait_value")

# If user selected deeper filters, keep only those
if sel_organs:
    active_organs = active_organs.intersection(set(sel_organs))

if sel_trait_categories:
    active_trait_categories = active_trait_categories.intersection(set(sel_trait_categories))

if sel_trait_values:
    active_trait_values = active_trait_values.intersection(set(sel_trait_values))

# If categories are selected, keep only values that belong to those categories
if active_trait_categories and active_trait_values:
    values_for_active_categories = edge_sources(df_edges, active_trait_categories, "value_of_category")
    active_trait_values = active_trait_values.intersection(values_for_active_categories)

# ----- Step D: assemble final active node set -----
active_nodes = set()
active_nodes.update(active_genera)
active_nodes.update(active_species)

if show_speciesorgan_nodes:
    active_nodes.update(active_speciesorgans)

active_nodes.update(active_organs)

if show_trait_category_nodes:
    active_nodes.update(active_trait_categories)

if show_value_nodes:
    active_nodes.update(active_trait_values)

# Build edge subset
df_edges_subset = filter_edges_between_nodes(df_edges, active_nodes)

# Optional cleanup by relation type if some node layers are hidden
allowed_relations = {"has_species", "has_organ", "instance_of", "has_trait_category", "has_trait_value", "value_of_category", "belongs_to_organ"}

if not show_speciesorgan_nodes:
    allowed_relations.discard("has_organ")
    allowed_relations.discard("instance_of")
    allowed_relations.discard("has_trait_category")
    allowed_relations.discard("has_trait_value")

if not show_trait_category_nodes:
    allowed_relations.discard("has_trait_category")
    allowed_relations.discard("value_of_category")
    allowed_relations.discard("belongs_to_organ")

if not show_value_nodes:
    allowed_relations.discard("has_trait_value")
    allowed_relations.discard("value_of_category")

df_edges_subset = df_edges_subset[df_edges_subset["relation"].isin(allowed_relations)].copy()

# Safety limit
if len(df_edges_subset) > max_edges:
    st.warning(
        f"⚠️ Subgraph too large ({len(df_edges_subset)} edges). "
        f"Showing a random sample of {max_edges} edges to keep rendering responsive."
    )
    df_edges_subset = df_edges_subset.sample(n=max_edges, random_state=42)

if len(df_edges_subset) == 0:
    st.warning("No edges matched the current filter combination.")
    st.stop()


# ==========================================
# 7. Graph Building and Rendering
# ==========================================
G = nx.from_pandas_edgelist(
    df_edges_subset,
    source="source",
    target="target",
    edge_attr="relation",
    create_using=nx.DiGraph()
)

# Add any isolated selected nodes not present after edge sampling
for node_id in active_nodes:
    if node_id not in G.nodes:
        G.add_node(node_id)

net = Network(
    height="780px",
    width="100%",
    bgcolor="#111111",
    font_color="white",
    directed=True,
    notebook=False,
    cdn_resources="in_line"
)

# Node colors
color_map = {
    "Genus": "#ff4d4d",          # red
    "Species": "#4da6ff",        # blue
    "Organ": "#ffd24d",          # yellow
    "SpeciesOrgan": "#b366ff",   # purple
    "TraitCategory": "#00c2a8",  # teal
    "TraitValue": "#4dff88",     # green
}

# Node shapes
shape_map = {
    "Genus": "dot",
    "Species": "dot",
    "Organ": "triangle",
    "SpeciesOrgan": "box",
    "TraitCategory": "diamond",
    "TraitValue": "ellipse",
}

# Node sizing
for node_id in G.nodes:
    node_type = node_type_dict.get(node_id, "Unknown")
    display_name = display_name_dict.get(node_id, node_id)
    value_type = value_type_dict.get(node_id, "")
    global_deg = global_degree_dict.get(node_id, 1)

    color = color_map.get(node_type, "#cccccc")
    shape = shape_map.get(node_type, "dot")

    if node_type == "Genus":
        size = 24
    elif node_type == "Species":
        size = 18
    elif node_type == "Organ":
        size = 20
    elif node_type == "SpeciesOrgan":
        size = 16
    elif node_type == "TraitCategory":
        size = 18
    elif node_type == "TraitValue":
        size = min(12 + global_deg * 0.35, 38)
    else:
        size = 12

    tooltip = f"""
    <b>{display_name}</b><br>
    ID: {node_id}<br>
    Type: {node_type}<br>
    Value Type: {value_type}<br>
    Global Degree: {global_deg}
    """

    net.add_node(
        node_id,
        label=str(display_name),
        title=tooltip,
        color=color,
        shape=shape,
        size=size
    )

# Edges
for source, target, data in G.edges(data=True):
    relation = data.get("relation", "")
    net.add_edge(
        source,
        target,
        title=relation,
        label=relation if show_edge_labels else "",
        arrows="to"
    )

# Physics / layout
if physics_enabled:
    net.force_atlas_2based(
        gravity=-45,
        central_gravity=0.01,
        spring_length=130,
        spring_strength=0.06,
        damping=0.75
    )
else:
    net.toggle_physics(False)

# Slightly nicer interaction options
net.set_options("""
const options = {
  "interaction": {
    "hover": true,
    "navigationButtons": true,
    "keyboard": true
  },
  "edges": {
    "smooth": {
      "type": "dynamic"
    },
    "color": {
      "inherit": false
    },
    "font": {
      "size": 12,
      "align": "middle"
    }
  },
  "nodes": {
    "font": {
      "size": 14
    }
  }
}
""")

# Save and render HTML
with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
    tmp_path = tmp_file.name

net.save_graph(tmp_path)

with open(tmp_path, "r", encoding="utf-8") as f:
    html_data = f.read()

components.html(html_data, height=800, scrolling=False)


# ==========================================
# 8. Sidebar Stats
# ==========================================
with st.sidebar:
    st.markdown("---")
    st.header("📊 Current View Stats")
    st.write(f"**Nodes Displayed:** {len(G.nodes)}")
    st.write(f"**Edges Displayed:** {len(G.edges)}")

    node_types_in_view = (
        pd.Series([node_type_dict.get(n, "Unknown") for n in G.nodes])
        .value_counts()
        .to_dict()
    )

    st.markdown("**Node Types in Current View**")
    for k, v in node_types_in_view.items():
        st.write(f"- {k}: {v}")


# ==========================================
# 9. Main Page Summary Table
# ==========================================
st.markdown("### Current Subgraph Summary")

summary_rows = [
    ["Genus", len([n for n in G.nodes if node_type_dict.get(n) == "Genus"])],
    ["Species", len([n for n in G.nodes if node_type_dict.get(n) == "Species"])],
    ["SpeciesOrgan", len([n for n in G.nodes if node_type_dict.get(n) == "SpeciesOrgan"])],
    ["Organ", len([n for n in G.nodes if node_type_dict.get(n) == "Organ"])],
    ["TraitCategory", len([n for n in G.nodes if node_type_dict.get(n) == "TraitCategory"])],
    ["TraitValue", len([n for n in G.nodes if node_type_dict.get(n) == "TraitValue"])],
]

summary_df = pd.DataFrame(summary_rows, columns=["Node Type", "Count"])
st.dataframe(summary_df, use_container_width=True)

st.markdown("### Edge Types in Current View")
edge_summary_df = (
    df_edges_subset["relation"]
    .value_counts()
    .reset_index()
)
edge_summary_df.columns = ["Relation", "Count"]
st.dataframe(edge_summary_df, use_container_width=True)
