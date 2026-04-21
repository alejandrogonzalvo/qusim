use ndarray::Array2;

/// Reconstruct a placements matrix `(num_layers+1, num_qubits)` from the
/// initial virt→core snapshot and the teleportation event log.
///
/// `gate_idx` in events is mapped to a DAG layer via linear interpolation:
///   `layer = round(gate_idx * num_layers / total_ts_gates)`
///
/// The qubit's new core takes effect from `layer + 1` onwards because
/// `placements[l]` represents the placement *before* layer `l`'s gates
/// execute; the teleport completes during layer `layer`, so the new core
/// is first visible at `placements[layer + 1]`.
pub fn reconstruct_placements(
    initial_virt_to_core: &[i32],
    teleport_events: &[(i32, i32, i32, i32)],  // (vqubit, from_core, to_core, gate_idx)
    total_ts_gates: usize,
    num_layers: usize,
) -> Array2<i32> {
    let num_qubits = initial_virt_to_core.len();
    let rows = num_layers + 1;
    let mut placements = Array2::<i32>::zeros((rows, num_qubits));

    // Fill row 0 from initial snapshot
    for (q, &core) in initial_virt_to_core.iter().enumerate() {
        placements[[0, q]] = core;
    }

    // Propagate forward: each layer starts as a copy of the previous
    for l in 1..rows {
        for q in 0..num_qubits {
            placements[[l, q]] = placements[[l - 1, q]];
        }
    }

    // Apply teleport events sorted by gate_idx: new core takes effect at layer+1.
    // Sorting is required so that multiple events on the same qubit are applied
    // in chronological order; reverse ordering would cause later writes to be
    // overwritten by earlier ones, silently producing wrong placements.
    let mut sorted_events = teleport_events.to_vec();
    sorted_events.sort_unstable_by_key(|&(_, _, _, gate_idx)| gate_idx);
    for &(vqubit, _from_core, to_core, gate_idx) in &sorted_events {
        debug_assert!(gate_idx >= 0, "gate_idx must be non-negative, got {gate_idx}");
        let layer = gate_idx_to_layer(gate_idx as usize, total_ts_gates, num_layers);
        let q = vqubit as usize;
        if q < num_qubits {
            let start = (layer + 1).min(rows);
            for l in start..rows {
                placements[[l, q]] = to_core;
            }
        }
    }

    placements
}

/// Convert swap events to the `[[layer, vq1, vq2], ...]` format accepted by
/// `estimate_hardware_fidelity`.
pub fn build_sparse_swaps(
    swap_events: &[(i32, i32, i32)],            // (vq1, vq2, gate_idx)
    total_ts_gates: usize,
    num_layers: usize,
) -> Vec<[i32; 3]> {
    swap_events
        .iter()
        .map(|&(vq1, vq2, gate_idx)| {
            debug_assert!(gate_idx >= 0, "gate_idx must be non-negative, got {gate_idx}");
            let layer = gate_idx_to_layer(gate_idx as usize, total_ts_gates, num_layers);
            debug_assert!(layer <= i32::MAX as usize, "layer {layer} overflows i32");
            [layer as i32, vq1, vq2]
        })
        .collect()
}

/// Map a TeleSABRE gate index to a DAG layer via linear interpolation,
/// clamped to `[0, num_layers]`.
fn gate_idx_to_layer(gate_idx: usize, total_ts_gates: usize, num_layers: usize) -> usize {
    if total_ts_gates == 0 || num_layers == 0 {
        return 0;
    }
    let layer = ((gate_idx as f64 / total_ts_gates as f64) * num_layers as f64).round() as usize;
    layer.min(num_layers)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 4 virtual qubits, 2 cores (cap=2 each), 6 circuit layers.
    /// Initial: q0,q1 → core 0; q2,q3 → core 1.
    /// One teleport: q0 moves from core 0 → core 1 at gate_idx=3 (halfway through 6 gates).
    /// Expected: placements[layer][0] switches from 0 to 1 AFTER layer 3.
    /// i.e., layers 0..=3 show core 0; layers 4..=6 show core 1.
    #[test]
    fn test_reconstruct_single_teleport() {
        let initial = vec![0_i32, 0, 1, 1]; // q0,q1 in core0; q2,q3 in core1
        let teleports = vec![(0_i32, 0, 1, 3)]; // q0: core0→core1 at gate_idx 3
        let placements = reconstruct_placements(&initial, &teleports, 6, 6);

        // Layers 0..=3: q0 is still in core 0 (teleport takes effect at next layer)
        for l in 0..=3 {
            assert_eq!(placements[[l, 0]], 0, "layer {l}: q0 should be in core 0");
        }
        // Layers 4..=6: q0 is in core 1
        for l in 4..=6 {
            assert_eq!(placements[[l, 0]], 1, "layer {l}: q0 should be in core 1");
        }
        // q1 never moves
        for l in 0..=6 {
            assert_eq!(placements[[l, 1]], 0);
        }
    }

    #[test]
    fn test_reconstruct_no_events() {
        let initial = vec![0_i32, 0, 1, 1];
        let placements = reconstruct_placements(&initial, &[], 6, 4);
        for l in 0..=4 {
            assert_eq!(placements[[l, 0]], 0);
            assert_eq!(placements[[l, 2]], 1);
        }
    }

    #[test]
    fn test_build_sparse_swaps() {
        // SWAP at gate_idx=3, out of 6 total gates, 6 layers → layer 3
        let swap_events = vec![(0_i32, 1, 3)];
        let sparse = build_sparse_swaps(&swap_events, 6, 6);
        assert_eq!(sparse.len(), 1);
        assert_eq!(sparse[0], [3, 0, 1]);
    }
}
