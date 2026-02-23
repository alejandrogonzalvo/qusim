use ndarray::{s, ArrayView2, ArrayView3, Axis};

/// A domain-specific wrapper around the 3D interaction tensor.
/// Shape: (num_layers, num_qubits, num_qubits).
pub struct InteractionTensor<'a> {
    view: ArrayView3<'a, f64>,
}

impl<'a> InteractionTensor<'a> {
    pub fn new(view: ArrayView3<'a, f64>) -> Self {
        Self { view }
    }

    pub fn num_layers(&self) -> usize {
        self.view.dim().0
    }

    pub fn num_qubits(&self) -> usize {
        self.view.dim().1
    }

    #[inline]
    pub fn weight(&self, layer: usize, q1: usize, q2: usize) -> f64 {
        self.view[[layer, q1, q2]]
    }

    /// Returns a 2D view of a single layer.
    pub fn current_layer(&self, layer_idx: usize) -> ArrayView2<'_, f64> {
        self.view.index_axis(Axis(0), layer_idx)
    }

    /// Returns a sub-tensor from `current_layer..` as a new ArrayView3.
    pub fn future_view(&self, current_layer: usize) -> ArrayView3<'_, f64> {
        self.view.slice(s![current_layer.., .., ..])
    }
}
