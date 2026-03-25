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
st.markdown("Use the cascading sidebar filters to explore specific sub-graphs. Now fully supports Numerical Traits and anatomical backbone relations.")

# 2. Data Loading & Caching
@st.cache_data
def load_data():
    nodes_df = pd.read_csv("KG_nodes_normalized.csv")
    edges_df = pd.read_csv("KG_edges_normalized.csv")
    
    # Pre-calculate global degree for accurate node sizing
    temp_G = nx.from_pandas_edgelist(edges_df, 'source', 'target', create_using=nx.DiGraph())
    global_degree = dict(temp_G.degree())
    
    # Extract unique lists for all entities
    all_genera = sorted(nodes_df[nodes_df['label'] == 'Genus']['node_id'].dropna().astype(str).tolist())
    all_species = sorted(nodes_df[nodes_df['label'] == 'Species']['node_id'].dropna().astype(str).tolist())
    
    # Combine both Categorical and Numerical traits for the filter
    trait_labels = ['TraitValue', 'NumericalTrait', 'Trait'] 
    all_traits = sorted(nodes_df[nodes_df['label'].isin(trait_labels)]['node_id'].dropna().astype(str).tolist())
    
    return nodes_df, edges_df, global_degree, all_genera, all_species, all_traits

df_nodes, df_edges, global_degree_dict, all_genera, all_species, all_traits = load_data()

# 3. Cascading Sidebar Filtering UI
with st.sidebar:
    st.header("🔍 Cascading Filters")
    st.markdown("Selections strictly limit downstream options.")
    
    # --- LEVEL 1: GENUS ---
    default_genera = [g for g in ['aloe', 'arenaria'] if g in all_genera]
    sel_genera = st.multiselect("1. Select Genus:", options=all_genera, default=default_genera)
    
    # --- LEVEL 2: SPECIES ---
    if sel_genera:
        # Check both directions to be safe with new taxonomy relations ('has_species' or 'belongs_to')
        valid_sp_edges = df_edges[(df_edges['source'].isin(sel_genera)) & (df_edges['relation'].isin(['has_species', 'belongs_to']))]
        sp_targets = valid_sp_edges['target'].tolist()
        sp_sources = df_edges[(df_edges['target'].isin(sel_genera)) & (df_edges['relation'] == 'belongs_to')]['source'].tolist()
        available_species = sorted(list(set(sp_targets + sp_sources)))
    else:
        available_species = all_species
        
    sel_species = st.multiselect("2. Select Species:", options=available_species)
    
    # --- LEVEL 3: TRAIT ---
    # Catch all trait relations including the newly added specific organ sizes
    trait_relations = ['has_trait', 'has_leaf_size', 'has_organ_size']
    if sel_species:
        valid_tr_edges = df_edges[(df_edges['source'].isin(sel_species)) & (df_edges['relation'].isin(trait_relations))]
        available_traits = sorted(valid_tr_edges['target'].unique().astype(str).tolist())
    elif sel_genera:
        valid_tr_edges = df_edges[(df_edges['source'].isin(available_species)) & (df_edges['relation'].isin(trait_relations))]
        available_traits = sorted(valid_tr_edges['target'].unique().astype(str).tolist())
    else:
        available_traits = all_traits
        
    sel_traits = st.multiselect("3. Select Traits/Sizes:", options=available_traits)
    
    st.markdown("---")
    st.header("⚙️ View Settings")
    show_edge_labels = st.checkbox("Show Edge Labels", value=False)
    include_anatomy = st.checkbox("Include Anatomy Backbone", value=True, help="Automatically shows 'leaf', 'plant' ontology backbone nodes if injected in data.")

# 4. Strict Sub-graph Extraction Logic (Dynamic for all active nodes)
if sel_genera or sel_species or sel_traits:
    
    # Base active sets
    active_genera = set(sel_genera) if sel_genera else set(all_genera)
    active_species = set(sel_species) if sel_species else set(available_species)
    active_traits = set(sel_traits) if sel_traits else set(available_traits)
    
    # Bottom-Up Constraint: If Traits are selected, strictly limit Species
    if sel_traits:
        sp_with_traits = set(df_edges[(df_edges['target'].isin(active_traits)) & (df_edges['relation'].isin(trait_relations))]['source'])
        active_species = active_species.intersection(sp_with_traits)
        
    # Bottom-Up Constraint: If Species were limited, strictly limit Genera
    if sel_species or sel_traits:
        gen_with_species_fwd = set(df_edges[(df_edges['target'].isin(active_species)) & (df_edges['relation'] == 'has_species')]['source'])
        gen_with_species_rev = set(df_edges[(df_edges['source'].isin(active_species)) & (df_edges['relation'] == 'belongs_to')]['target'])
        gen_with_species = gen_with_species_fwd.union(gen_with_species_rev)
        active_genera = active_genera.intersection(gen_with_species)

    # Combine all actively selected nodes
    active_nodes = active_genera.union(active_species).union(active_traits)
    
    # Add anatomy and biological backbone if toggled and present in the CSV
    if include_anatomy:
        anatomy_labels = ['Organism', 'AnatomicalStructure', 'BiologicalFunction']
        anatomy_nodes = set(df_nodes[df_nodes['label'].isin(anatomy_labels)]['node_id'].astype(str).tolist())
        active_nodes = active_nodes.union(anatomy_nodes)

    # Final Edge Mask: Keep ANY edge where BOTH source and target are in our active nodes pool
    mask_all = (df_edges['source'].astype(str).isin(active_nodes)) & (df_edges['target'].astype(str).isin(active_nodes))
    df_edges_subset = df_edges[mask_all]
else:
    df_edges_subset = pd.DataFrame()

# 5. Safety Limit
MAX_EDGES = 3000
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
    
    # Enhanced Color Mapping for the new node types
    color_map = {
        'Genus': '#ff4d4d',               # Vibrant Red
        'Species': '#4da6ff',             # Light Blue
        'TraitValue': '#4dff88',          # Neon Green (Categorical)
        'NumericalTrait': '#ffb366',      # Orange (Numerical size data)
        'Organism': '#9933ff',            # Purple (Backbone)
        'AnatomicalStructure': '#ffff66', # Yellow (Backbone)
        'BiologicalFunction': '#ff66b3',  # Pink (Backbone)
        'Trait': '#4dff88'                # Fallback for old data
    }
    
    # Enhanced Shape Mapping to visually separate dimensions
    shape_map = {
        'Genus': 'dot',
        'Species': 'dot',
        'TraitValue': 'dot',
        'NumericalTrait': 'square',       # Differentiate numerical traits with a square
        'Organism': 'star',               # Star for root backbone (e.g., Plant)
        'AnatomicalStructure': 'triangle',# Triangle for organs (e.g., Leaf)
        'BiologicalFunction': 'diamond',  # Diamond for functions
        'Trait': 'dot'
    }
    
    for node_id in G.nodes:
        node_type = node_label_dict.get(node_id, 'Unknown')
        color = color_map.get(node_type, '#ffffff')
        shape = shape_map.get(node_type, 'dot')
        global_deg = global_degree_dict.get(node_id, 1)
        
        # Calculate size based on global degree and specific node importance
        if node_type == 'Genus':
            size = 15 + (global_deg * 0.5)
        elif node_type == 'Species':
            size = 10 
        elif node_type in ['TraitValue', 'NumericalTrait', 'Trait']:
            size = 15 + (global_deg * 0.5)
        elif node_type in ['Organism', 'AnatomicalStructure', 'BiologicalFunction']:
            size = 30  # Make backbone ontology nodes much larger and fixed
        else:
            size = 10
            
        size = min(size, 60) # Size cap
        
        # Add Node to PyVis
        net.add_node(
            node_id, 
            label=str(node_id), 
            title=f"Type: {node_type} | Global Connections: {global_deg}", 
            color=color, 
            size=size,
            shape=shape
        )
        
    for source, target, data in G.edges(data=True):
        edge_title = data.get('relation', '')
        edge_label = edge_title if show_edge_labels else ""
        net.add_edge(source, target, title=edge_title, label=edge_label)

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
