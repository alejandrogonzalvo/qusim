/// Sparse edge list: for each layer, a list of (q1, q2, weight, gate_type) tuples.
pub type ActiveGates = Vec<Vec<(usize, usize, f64, usize)>>;

/// Sparse interaction tensor storing only non-zero gate interactions.
/// Replaces the dense Array3 representation for O(|E|) iteration instead of O(N²).
pub struct InteractionTensor {
    active_gates: ActiveGates,
    num_layers: usize,
    num_qubits: usize,
}

impl InteractionTensor {
    /// Build directly from flat sparse JSON data: each entry is [layer, q1, q2, weight, gate_type].
    pub fn from_sparse(gs_sparse: &[[f64; 5]], num_layers: usize, num_qubits: usize) -> Self {
        let mut active_gates = vec![Vec::new(); num_layers];
        for edge in gs_sparse {
            let layer = edge[0] as usize;
            let u = edge[1] as usize;
            let v = edge[2] as usize;
            let w = edge[3];
            let gate_type = edge[4] as usize;
            if layer < num_layers && u < num_qubits && v < num_qubits && w > 0.0 {
                active_gates[layer].push((u, v, w, gate_type));
            }
        }
        Self {
            active_gates,
            num_layers,
            num_qubits,
        }
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Returns the sparse edge list for all layers.
    pub fn active_gates(&self) -> &ActiveGates {
        &self.active_gates
    }

    /// Returns the sparse edge list for a single layer.
    pub fn layer_gates(&self, layer: usize) -> &[(usize, usize, f64, usize)] {
        &self.active_gates[layer]
    }
}
