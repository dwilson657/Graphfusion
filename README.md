# GraphFusion

GraphFusion is an experimental spatiotemporal Graph Neural Network (GNN) architecture designed to simulate Tokamak plasma behavior and magnetic confinement fields. This framework applies structural principles inspired by DeepMind's GraphCast architecture to model high-dimensional fluid dynamics and plasma profiles inside a closed toroidal geometry.

## 🦾 Project Architecture

The simulation pipeline follows a modular Encoder-Processor-Decoder sequence:

1. **Toroidal Mesh Generation (`graph_fusion_mesh.py`)**  
   Generates a discrete 3D spatial coordinate grid (`X, Y, Z`) mapping the parametric dimensions of a Tokamak vacuum vessel based on core major and minor radii.
   
2. **Spatial Edge Calculation (`graph_fusion_edges.py`)**  
   Constructs a bidirectional graph network by computing spatial proximity connections using an R-radius neighbor search with a fast `scipy.spatial.KDTree`.

3. **Message-Passing GNN (`graph_fusion_layer.py`)**  
   Implements a custom PyTorch message-passing neural network layer. It aggregates localized node interactions (plasma features) across spatial edges to compute latent state transitions.

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- NumPy
- SciPy
- PyTorch

### Core Workflow
The components link together sequentially to initialize the graph structure:
```python
from graph_fusion_mesh import generate_toroidal_mesh
from graph_fusion_edges import calculate_spatial_edges

# 1. Generate 3D coordinates inside the torus
nodes = generate_toroidal_mesh(major_R=3.0, minor_r=1.0, num_u=30, num_v=50)

# 2. Establish spatial graph network edges
edge_index = calculate_spatial_edges(nodes, max_distance_threshold=0.55)
```

## 🎯 Project Roadmap
- Integrate temporal steps to model plasma evolution over time.
- Implement specialized loss functions to enforce physical constraints (e.g., conservation of magnetic flux).
- Deploy and train scaling configurations utilizing Google Cloud Platform computing environments.
# Graphfusion
