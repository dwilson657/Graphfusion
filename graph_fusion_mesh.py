import numpy as np

def generate_toroidal_mesh(major_R, minor_r, num_u=40, num_v=40):
    """
    Generates a 3D toroidal coordinate system for a Tokamak simulation.
    
    Parameters:
    - major_R (float): Distance from the center of the torus to the center of the tube.
    - minor_r (float): Radius of the inner tube (plasma containment cross-section).
    - num_u (int): Number of sampling points along the tube's cross-section (poloidal angle).
    - num_v (int): Number of sampling points around the ring of the torus (toroidal angle).
    
    Returns:
    - numpy.ndarray: Array of shape (num_u * num_v, 3) containing [x, y, z] node coordinates.
    """
    # u represents the poloidal angle (0 to 2*pi) around the minor cross-section
    u = np.linspace(0, 2 * np.pi, num_u, endpoint=False)
    
    # v represents the toroidal angle (0 to 2*pi) around the major ring
    v = np.linspace(0, 2 * np.pi, num_v, endpoint=False)
    
    # Create a coordinate grid mapping out the surface space
    u_grid, v_grid = np.meshgrid(u, v)
    
    # Parametric equations for a torus mapping to 3D Cartesian coordinates
    x = (major_R + minor_r * np.cos(u_grid)) * np.cos(v_grid)
    y = (major_R + minor_r * np.cos(u_grid)) * np.sin(v_grid)
    z = minor_r * np.sin(u_grid)
    
    # Flatten the grids and stack them to create an (N, 3) node coordinate array
    nodes = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    
    return nodes

# --- Task-Botting Execution & Verification ---
if __name__ == "__main__":
    # Example dimensions matching standard experimental proportions
    MAJOR_RADIUS = 3.0   # Center to tube center
    MINOR_RADIUS = 1.0   # Core plasma cross-section radius
    
    # Generate mesh grid nodes
    mesh_nodes = generate_toroidal_mesh(MAJOR_RADIUS, MINOR_RADIUS, num_u=30, num_v=50)
    
    print("=== Toroidal Mesh Initialized ===")
    print(f"Total Graph Nodes Generated: {mesh_nodes.shape[0]} (Shape: {mesh_nodes.shape})")
    print("\nFirst 5 Node Coordinates [X, Y, Z]:")
    print(mesh_nodes[:5])
