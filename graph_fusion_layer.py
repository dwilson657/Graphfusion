import torch
import torch.nn as nn

class PlasmaMessagePassingLayer(nn.Module):
    """
    A PyTorch Message Passing Layer optimized for Tokamak plasma simulations.
    Models spatial interactions across the toroidal grid.
    """
    def __init__(self, node_features_in, edge_features_in, hidden_dim):
        super(PlasmaMessagePassingLayer, self).__init__()
        
        # MLP to compute messages based on source node, target node, and edge features
        # Input size: (node_in + node_in + edge_in)
        self.message_mlp = nn.Sequential(
            nn.Linear(node_features_in * 2 + edge_features_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # MLP to update node states using their old state and aggregated messages
        # Input size: (node_in + hidden_dim)
        self.update_mlp = nn.Sequential(
            nn.Linear(node_features_in + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Parameters:
        - x (torch.Tensor): Node features matrix of shape (num_nodes, node_features_in)
        - edge_index (torch.Tensor): Graph edge matrix of shape (2, num_edges)
        - edge_attr (torch.Tensor): Spatial edge features matrix of shape (num_edges, edge_features_in)
        
        Returns:
        - torch.Tensor: Updated node hidden states matrix of shape (num_nodes, hidden_dim)
        """
        row, col = edge_index[0], edge_index[1]
        
        # 1. Gather features for every edge connection
        x_source = x[row]       # Features of source nodes for each edge
        x_target = x[col]       # Features of target nodes for each edge
        
        # 2. Concatenate features along the edge dimension to compute messages
        # Shape: (num_edges, node_features_in + node_features_in + edge_features_in)
        edge_inputs = torch.cat([x_source, x_target, edge_attr], dim=-1)
        messages = self.message_mlp(edge_inputs)
        
        # 3. Aggregate messages at target nodes using sum reduction
        num_nodes = x.size(0)
        aggregated_messages = torch.zeros(num_nodes, messages.size(-1), device=x.device)
        
        # Scatter_add accumulates the messages into target node rows matching the 'col' indices
        aggregated_messages.scatter_add_(0, col.unsqueeze(-1).expand_as(messages), messages)
        
        # 4. Combine old node states with aggregated messages to compute updated states
        node_inputs = torch.cat([x, aggregated_messages], dim=-1)
        updated_nodes = self.update_mlp(node_inputs)
        
        return updated_nodes

# --- Task-Botting Execution & Verification ---
if __name__ == "__main__":
    torch.manual_seed(42) # Lock weights for verification
    
    # Mock Graph Setup matching our spatial outputs
    NUM_NODES = 1500      # 30 u-points * 50 v-points
    NUM_EDGES = 12000     # Average spatial connections
    
    # Feature Dimensions
    NODE_FEAT_DIM = 4     # e.g., [temperature, density, pressure, velocity_z]
    EDGE_FEAT_DIM = 3     # e.g., [dx, dy, dz] spatial offset vectors
    HIDDEN_DIM = 16       # Latent space embedding size
    
    # Create Mock Tensors
    mock_x = torch.randn(NUM_NODES, NODE_FEAT_DIM)
    mock_edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES))
    mock_edge_attr = torch.randn(NUM_EDGES, EDGE_FEAT_DIM)
    
    # Initialize Layer
    gnn_layer = PlasmaMessagePassingLayer(NODE_FEAT_DIM, EDGE_FEAT_DIM, HIDDEN_DIM)
    output_states = gnn_layer(mock_x, mock_edge_index, mock_edge_attr)
    
    print("=== PyTorch Message-Passing Layer Initialized ===")
    print(f"Input Node Tensor Shape:  {mock_x.shape}")
    print(f"Output Node Hidden Shape: {output_states.shape} (Successfully projected to Hidden Dim)")
