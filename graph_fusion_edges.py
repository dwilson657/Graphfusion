import numpy as np
from scipy.spatial import KDTree

def calculate_spatial_edges(nodes, max_distance_threshold=0.8):
    """
    Calculates spatial graph edges between nodes using an R-radius search.
    
    Parameters:
    - nodes (numpy.ndarray): Array of shape (N, 3) representing [x, y, z] node coordinates.
    - max_distance_threshold (float): The maximum physical distance allowed between connected nodes.
    
    Returns:
    - numpy.ndarray: Array of shape (2, num_edges) containing source and target node indices.
    """
    # Build a fast Spatial KDTree to handle 3D coordinate lookup
    tree = KDTree(nodes)
    
    # Find all pairs of nodes within our maximum distance threshold
    # pairs is returned as a set of tuples: {(node_i, node_j), ...}
    pairs = tree.query_pairs(r=max_distance_threshold)
    
    # Convert the undirected pairs into a directed edge list for our graph
    sources = []
    targets = []
    
    for i, j in pairs:
        # Edge from i to j
        sources.append(i)
        targets.append(j)
        
        # Edge from j to i (graphs are bidirectional in plasma modeling)
        sources.append(j)
        targets.append(i)
        
    # Stack arrays together to create the canonical (2, E) shape for GNNs
    edge_index = np.vstack([sources, targets])
    
    return edge_index

# --- Task-Botting Execution & Verification ---
if __name__ == "__main__":
    # 1. Regenerate our mesh nodes from step 1 for testing
    from graph_fusion_mesh import generate_toroidal_mesh
    mesh_nodes = generate_toroidal_mesh(major_R=3.0, minor_r=1.0, num_u=30, num_v=50)
    
    # 2. Compute edge network layout
    DISTANCE_THRESHOLD = 0.55  # Custom threshold to link close points
    edges = calculate_spatial_edges(mesh_nodes, max_distance_threshold=DISTANCE_THRESHOLD)
    
    print("=== Spatial Edge Network Computed ===")
    print(f"Total Nodes Evaluated: {mesh_nodes.shape[0]}")
    print(f"Total Directed Edges Generated: {edges.shape[1]} (Shape: {edges.shape})")
    print(f"Average Connections Per Node: {edges.shape[1] / mesh_nodes.shape[0]:.1f}")
    print("\nFirst 5 Edge Pairs [Source Node ID -> Target Node ID]:")
    print(edges[:, :5])
