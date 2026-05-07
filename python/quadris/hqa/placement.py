import numpy as np
import random
import networkx as nx
from enum import Enum
from dataclasses import dataclass

class InitialPlacement(str, Enum):
    RANDOM = "random"
    SPECTRAL_CLUSTERING = "spectral_clustering"

@dataclass
class PlacementConfig:
    policy: InitialPlacement
    interaction_tensor: np.ndarray
    num_virtual_qubits: int
    core_caps: np.ndarray
    seed: int | None = None

def generate_initial_placement(config: PlacementConfig) -> np.ndarray:
    """
    Generate an initial mapping of virtual qubits to physical cores based on
    a specific placement policy.
    """
    if config.policy == InitialPlacement.RANDOM:
        return _random_placement(config)
    elif config.policy == InitialPlacement.SPECTRAL_CLUSTERING:
        return _spectral_placement(config)
    else:
        raise ValueError(f"Unknown initial placement policy: {config.policy}")

def _random_placement(config: PlacementConfig) -> np.ndarray:
    """
    Randomly assign virtual qubits to physical cores, respecting capacities.
    """
    part = []
    for c_idx, cap in enumerate(config.core_caps):
        part.extend([c_idx] * cap)
    part = part[:config.num_virtual_qubits]
    
    if config.seed:
        random.seed(config.seed)
        
    random.shuffle(part)
    
    return np.array(part, dtype=np.int32)

def _spectral_placement(config: PlacementConfig) -> np.ndarray:
    """
    Use spectral graph partitioning on the first 20 layers of the circuit
    to group highly-interacting qubits onto the same cores.
    """
    if config.num_virtual_qubits == 0:
        return np.array([], dtype=np.int32)
        
    num_cores = len(config.core_caps)
    
    # Extract lookahead interaction graph (first 20 layers)
    G = nx.Graph()
    G.add_nodes_from(range(config.num_virtual_qubits))
    
    BASELINE_GRAPH_WEIGHT = 1e-6
    for i in range(config.num_virtual_qubits):
        for j in range(i + 1, config.num_virtual_qubits):
            G.add_edge(i, j, weight=BASELINE_GRAPH_WEIGHT)
            
    LOOKAHEAD_HORIZON = 20
    
    for row in config.interaction_tensor:
        layer = row[0]
        if layer >= LOOKAHEAD_HORIZON:
            continue
            
        u = int(row[1])
        v = int(row[2])
        w = row[3]
        
        if u != v and w > 0:
            G[u][v]['weight'] += w
                
    # If the graph has no edges in the lookahead, fallback to random
    if G.number_of_edges() == 0:
        print("Fallback to random: No edges in lookahead")
        return _random_placement(config)
        
    # Spectral clustering using the Fiedler vector (2nd smallest Laplacian eigenvector)
    # The algebraic connectivity vector gives a 1D embedding of the graph where
    # closely connected nodes have similar values.
    try:
        fiedler_vec = nx.fiedler_vector(G, weight='weight', seed=config.seed)
    except nx.NetworkXError as e:
        # Fiedler vector fails if graph is completely disconnected or identical
        # We can fall back to random
        print(f"Fallback to random because nx.fiedler_vector raised: {e}")
        return _random_placement(config)
        
    # Sort qubits by their Fiedler embedding value
    sorted_qubits = np.argsort(fiedler_vec)
    
    # Pack them greedily into cores
    placement = np.full(config.num_virtual_qubits, -1, dtype=np.int32)
    
    current_core = 0
    capacity_remaining = config.core_caps[current_core]
    
    for q in sorted_qubits:
        while capacity_remaining == 0:
            current_core += 1
            if current_core >= num_cores:
                raise ValueError("Ran out of core capacity during spectral placement.")
            capacity_remaining = config.core_caps[current_core]
            
        placement[q] = current_core
        capacity_remaining -= 1
        
    return placement
