import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

# 1. Page Configuration
st.set_page_config(
    page_title="Plant Knowledge Graph",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌿 Plant Trait & Taxonomy Knowledge Graph")
st.markdown("Use the cascading sidebar filters to explore specific sub-graphs. Selections strictly limit downstream options to ensure valid relationships.")

# 2. Data Loading & Caching
@st.cache_data
def load_data():
    #nodes_df = pd.read_csv("KG_nodes_all.csv")
    nodes_df = pd.read_csv("KG_nodes_clean_nodes.csv")
    #edges_df = pd.read_csv("KG_edges_all.csv")
    edges_df = pd.read_csv("KG_edges_clean_edges.csv")
    
    # Pre-calculate global degree for accurate node sizing
    temp_G = nx.from_pandas_edgelist(edges_df, 'source', 'target', create_using=nx.DiGraph())
    global_degree = dict(temp_G.degree())
    
    # Extract unique lists for all entities
    all_genera = sorted(nodes_df[nodes_df['label'] == 'Genus']['node_id'].dropna().tolist())
    all_species = sorted(nodes_df[nodes_df['label'] == 'Species']['node_id'].dropna().tolist())
    all_traits = sorted(nodes_df[nodes_df['label'] == 'Trait']['node_id'].dropna().tolist())
    
    return nodes_df, edges_df, global_degree, all_genera, all_species, all_traits

df_nodes, df_edges, global_degree_dict, all_genera, all_species, all_traits = load_data()

# 3. Cascading Sidebar Filtering UI
with st.sidebar:
    st.header("🔍 Cascading Filters")
    st.markdown("Selections limit downstream options.")
    
    # --- LEVEL 1: GENUS ---
    default_genera = [g for g in ['aloe', 'arenaria'] if g in all_genera]
    sel_genera = st.multiselect("1. Select Genus (Parent):", options=all_genera, default=default_genera)
    
    # --- LEVEL 2: SPECIES (Dependent on Genus) ---
    if sel_genera:
        valid_sp_edges = df_edges[(df_edges['source'].isin(sel_genera)) & (df_edges['relation'] == 'has_species')]
        available_species = sorted(valid_sp_edges['target'].unique().tolist())
    else:
        available_species = all_species
        
    sel_species = st.multiselect("2. Select Species (Child):", options=available_species)
    
    # --- LEVEL 3: TRAIT (Dependent on Species or Genus) ---
    if sel_species:
        valid_tr_edges = df_edges[(df_edges['source'].isin(sel_species)) & (df_edges['relation'] == 'has_trait')]
        available_traits = sorted(valid_tr_edges['target'].unique().tolist())
    elif sel_genera:
        valid_tr_edges = df_edges[(df_edges['source'].isin(available_species)) & (df_edges['relation'] == 'has_trait')]
        available_traits = sorted(valid_tr_edges['target'].unique().tolist())
    else:
        available_traits = all_traits
        
    sel_traits = st.multiselect("3. Select Trait (Morphology):", options=available_traits)

# 4. Strict Sub-graph Extraction Logic (Top-Down & Bottom-Up)
if sel_genera or sel_species or sel_traits:
    
    # Base active sets
    active_genera = set(sel_genera) if sel_genera else set(all_genera)
    active_species = set(sel_species) if sel_species else set(available_species)
    active_traits = set(sel_traits) if sel_traits else set(available_traits)
    
    # Bottom-Up Constraint: If Traits are selected, strictly limit Species to those possessing the Trait
    if sel_traits:
        sp_with_traits = set(df_edges[(df_edges['target'].isin(active_traits)) & (df_edges['relation'] == 'has_trait')]['source'])
        active_species = active_species.intersection(sp_with_traits)
        
    # Bottom-Up Constraint: If Species were limited (by user or by trait constraint), limit Genera
    if sel_species or sel_traits:
        gen_with_species = set(df_edges[(df_edges['target'].isin(active_species)) & (df_edges['relation'] == 'has_species')]['source'])
        active_genera = active_genera.intersection(gen_with_species)

    # Final Edge Mask: Only keep edges between active, strictly filtered entities
    mask_species = (df_edges['relation'] == 'has_species') & (df_edges['source'].isin(active_genera)) & (df_edges['target'].isin(active_species))
    mask_traits = (df_edges['relation'] == 'has_trait') & (df_edges['source'].isin(active_species)) & (df_edges['target'].isin(active_traits))

    df_edges_subset = df_edges[mask_species | mask_traits]
else:
    df_edges_subset = pd.DataFrame()

# 5. Safety Limit
MAX_EDGES = 2500
if len(df_edges_subset) > MAX_EDGES:
    st.warning(f"⚠️ **Sub-graph too large!** ({len(df_edges_subset)} edges). Showing a random sample of {MAX_EDGES} to prevent freezing.")
    df_edges_subset = df_edges_subset.sample(n=MAX_EDGES, random_state=42)
elif len(df_edges_subset) == 0:
    st.info("👈 Please select options from the sidebar or ensure your selection has valid connections.")

# 6. Graph Building and Rendering
if len(df_edges_subset) > 0:
    G = nx.from_pandas_edgelist(df_edges_subset, 'source', 'target', edge_attr='relation', create_using=nx.DiGraph())
    node_label_dict = pd.Series(df_nodes['label'].values, index=df_nodes['node_id']).to_dict()
    
    net = Network(height='750px', width='100%', bgcolor='#1a1a1a', font_color='white', directed=True)
    
    color_map = {
        'Genus': '#ff4d4d',   # Vibrant Red
        'Species': '#4da6ff', # Light Blue
        'Trait': '#4dff88'    # Neon Green
    }
    
    for node_id in G.nodes:
        node_type = node_label_dict.get(node_id, 'Unknown')
        color = color_map.get(node_type, '#ffffff')
        global_deg = global_degree_dict.get(node_id, 1)
        
        # Calculate size based on global degree to maintain context
        if node_type == 'Genus':
            size = 15 + (global_deg * 0.5)
        elif node_type == 'Species':
            size = 10 
        elif node_type == 'Trait':
            size = 15 + (global_deg * 0.5)
        else:
            size = 10
            
        size = min(size, 60) # Size cap
        net.add_node(node_id, label=node_id, title=f"Type: {node_type} | Global Connections: {global_deg}", color=color, size=size)
        
    for source, target, data in G.edges(data=True):
        net.add_edge(source, target, title=data.get('relation', ''))

    # Advanced Physics for strict sub-graphs
    net.force_atlas_2based(
        gravity=-60,
        central_gravity=0.015,
        spring_length=120,
        spring_strength=0.08,
        damping=0.6
    )
    
    path_to_html = "graph_interactive.html"
    net.save_graph(path_to_html)

    with open(path_to_html, 'r', encoding='utf-8') as f:
        html_data = f.read()
        
    components.html(html_data, height=760, scrolling=False)

    with st.sidebar:
        st.markdown("---")
        st.header("📊 Current View Stats")
        st.write(f"**Nodes Displayed:** {len(G.nodes)}")
        st.write(f"**Edges Displayed:** {len(G.edges)}")
