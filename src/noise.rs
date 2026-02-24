use std::collections::HashSet;

use crate::circuit::InteractionTensor;
use crate::routing::RoutingSummary;

/// Hardware noise parameters for a multi-core quantum architecture.
///
/// All error rates are probabilities in [0, 1]. All times are in nanoseconds.
/// Current values are placeholders — replace with physical models.
#[derive(Debug, Clone)]
pub struct ArchitectureParams {
    pub single_gate_error: f64,
    pub two_gate_error: f64,
    pub teleportation_error_per_hop: f64,
    pub single_gate_time: f64,
    pub two_gate_time: f64,
    pub teleportation_time_per_hop: f64,
    pub t1: f64,
    pub t2: f64,
}

impl Default for ArchitectureParams {
    fn default() -> Self {
        Self {
            single_gate_error: 1e-4,
            two_gate_error: 1e-3,
            teleportation_error_per_hop: 1e-2,
            single_gate_time: 20.0,
            two_gate_time: 100.0,
            teleportation_time_per_hop: 1000.0,
            t1: 100_000.0,
            t2: 50_000.0,
        }
    }
}

/// Per-layer fidelity breakdown.
#[derive(Debug, Clone)]
pub struct LayerFidelity {
    pub layer: usize,
    pub num_gates: usize,
    pub num_teleportations: usize,
    pub layer_time: f64,
    pub operational_fidelity: f64,
}

/// Complete fidelity report for a mapped quantum circuit.
#[derive(Debug, Clone)]
pub struct FidelityReport {
    /// Product of all native 1Q/2Q gates present in the original circuit.
    pub algorithmic_fidelity: f64,
    /// Product of all routing overhead (SABRE SWAPs + HQA Teleportations).
    pub routing_fidelity: f64,
    /// Fidelity loss from qubit idle time (T1/T2 decoherence).
    pub coherence_fidelity: f64,
    /// Overall fidelity = algorithmic × routing × coherence.
    pub overall_fidelity: f64,
    /// Total circuit execution time in nanoseconds.
    pub total_circuit_time: f64,
    /// Per-layer breakdown.
    pub layer_details: Vec<LayerFidelity>,

    // Per-qubit, per-timeslice gridded data (flattened for easy Python NumPy conversion)
    // Shape: (num_layers, num_qubits)
    pub algorithmic_fidelity_grid: Vec<f64>,
    pub routing_fidelity_grid: Vec<f64>,
    pub coherence_fidelity_grid: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Placeholder noise equations — replace with physical models
// ---------------------------------------------------------------------------

/// Depolarizing gate fidelity: F = 1 − ε.
#[inline]
fn gate_fidelity(error_rate: f64) -> f64 {
    1.0 - error_rate
}

/// Teleportation fidelity decays exponentially with network distance.
#[inline]
fn teleportation_fidelity(error_per_hop: f64, distance: i32) -> f64 {
    (1.0 - error_per_hop).powi(distance)
}

/// Idle-qubit decoherence from T1 relaxation and T2 dephasing.
#[inline]
fn decoherence_fidelity(idle_time: f64, t1: f64, t2: f64) -> f64 {
    (-idle_time / t1).exp() * (-idle_time / t2).exp()
}

/// Estimates the overall fidelity of a mapped quantum circuit.
pub fn estimate_fidelity(
    tensor: &InteractionTensor,
    routing: &RoutingSummary,
    params: &ArchitectureParams,
    intra_core_swaps_grid: Option<ndarray::ArrayView2<f64>>,
) -> FidelityReport {
    let num_layers = tensor.num_layers();
    let num_qubits = tensor.num_qubits();

    let mut overall_algorithmic = 1.0_f64;
    let mut overall_routing = 1.0_f64;

    let mut total_circuit_time = 0.0_f64;

    // Running sum of time each qubit has been actively occupied
    let mut qubit_busy_time = vec![0.0_f64; num_qubits];

    let mut layer_details = Vec::with_capacity(num_layers);

    let mut algorithmic_fidelity_grid = vec![1.0_f64; num_layers * num_qubits];
    let mut routing_fidelity_grid = vec![1.0_f64; num_layers * num_qubits];
    let mut coherence_fidelity_grid = vec![1.0_f64; num_layers * num_qubits];

    for layer in 0..num_layers {
        if layer > 0 {
            for q in 0..num_qubits {
                algorithmic_fidelity_grid[layer * num_qubits + q] =
                    algorithmic_fidelity_grid[(layer - 1) * num_qubits + q];
                routing_fidelity_grid[layer * num_qubits + q] =
                    routing_fidelity_grid[(layer - 1) * num_qubits + q];
            }
        }

        let gates = tensor.layer_gates(layer);
        let num_gates = gates.len();

        let layer_teleportations: Vec<_> = routing
            .events
            .iter()
            .filter(|e| e.timeslice == layer + 1)
            .collect();
        let num_teleportations = layer_teleportations.len();

        let max_distance = layer_teleportations
            .iter()
            .map(|e| e.network_distance)
            .max()
            .unwrap_or(0);

        let teleportation_time = max_distance as f64 * params.teleportation_time_per_hop;
        let gate_time = if num_gates > 0 {
            params.two_gate_time
        } else {
            0.0
        };
        // Add extra time if SABRE injected SWAPs in this layer
        let mut max_swaps_in_layer = 0.0;
        if let Some(swaps) = intra_core_swaps_grid {
            for q in 0..num_qubits {
                if swaps[[layer, q]] > max_swaps_in_layer {
                    max_swaps_in_layer = swaps[[layer, q]];
                }
            }
        }
        let swap_time = max_swaps_in_layer * params.two_gate_time * 3.0; // SWAP is ~3 CX gates

        let layer_time = teleportation_time + gate_time + swap_time;
        total_circuit_time += layer_time;

        let mut layer_algorithmic_fidelity = 1.0_f64;
        let mut layer_routing_fidelity = 1.0_f64;

        let mut busy: HashSet<usize> = HashSet::new();

        // 1. Process 2-qubit native computational gates (Algorithmic)
        for &(u, v, _) in gates {
            busy.insert(u);
            busy.insert(v);

            let f = gate_fidelity(params.two_gate_error);
            algorithmic_fidelity_grid[layer * num_qubits + u] *= f;
            algorithmic_fidelity_grid[layer * num_qubits + v] *= f;
            layer_algorithmic_fidelity *= f;
        }

        // 2. Process SabreRouting injected SWAP gates (Routing Overhead)
        if let Some(swaps) = intra_core_swaps_grid {
            for q in 0..num_qubits {
                let num_swaps = swaps[[layer, q]];
                if num_swaps > 0.0 {
                    busy.insert(q);
                    // SWAP requires ~3 native 2-qubit operations
                    let f = (1.0 - 3.0 * params.two_gate_error).powf(num_swaps);
                    routing_fidelity_grid[layer * num_qubits + q] *= f;
                    layer_routing_fidelity *= f;
                }
            }
        }

        // 3. Process teleportations (Routing Overhead)
        for event in &layer_teleportations {
            busy.insert(event.qubit);

            let f =
                teleportation_fidelity(params.teleportation_error_per_hop, event.network_distance);
            routing_fidelity_grid[layer * num_qubits + event.qubit] *= f;
            layer_routing_fidelity *= f;
        }

        overall_algorithmic *= layer_algorithmic_fidelity;
        overall_routing *= layer_routing_fidelity;

        // 4. Update busy times and compute coherence based on cumulative idle time
        for q in 0..num_qubits {
            if busy.contains(&q) {
                qubit_busy_time[q] += layer_time;
            }

            // Accumulated idle time up to this layer
            let idle_time = (total_circuit_time - qubit_busy_time[q]).max(0.0);
            let f_coh = decoherence_fidelity(idle_time, params.t1, params.t2);
            coherence_fidelity_grid[layer * num_qubits + q] = f_coh;
        }

        layer_details.push(LayerFidelity {
            layer,
            num_gates,
            num_teleportations,
            layer_time,
            operational_fidelity: layer_algorithmic_fidelity * layer_routing_fidelity,
        });
    }

    // Overall global coherence is the product of final coherence for all qubits
    let mut global_coherence = 1.0_f64;
    for q in 0..num_qubits {
        global_coherence *= coherence_fidelity_grid[(num_layers - 1) * num_qubits + q];
    }

    FidelityReport {
        algorithmic_fidelity: overall_algorithmic,
        routing_fidelity: overall_routing,
        coherence_fidelity: global_coherence,
        overall_fidelity: overall_algorithmic * overall_routing * global_coherence,
        total_circuit_time,
        layer_details,
        algorithmic_fidelity_grid,
        routing_fidelity_grid,
        coherence_fidelity_grid,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hqa::hqa_mapping;
    use crate::routing::extract_inter_core_communications;
    use ndarray::Array2;
    use serde_json::Value;
    use std::fs;
    use std::path::Path;

    fn run_fidelity_test(test_case_name: &str) -> FidelityReport {
        let path = Path::new("dse_pau/test_vectors.json");
        let data = fs::read_to_string(path).expect("Unable to read test_vectors.json");
        let parsed: Value = serde_json::from_str(&data).expect("Unable to parse JSON");

        let tc = &parsed[test_case_name];
        let num_qubits = tc["num_virtual_qubits"].as_u64().unwrap() as usize;
        let num_cores = tc["num_cores"].as_u64().unwrap() as usize;
        let num_layers = tc["num_layers"].as_u64().unwrap() as usize;

        let gs_sparse: Vec<[f64; 4]> = serde_json::from_value(tc["gs_sparse"].clone()).unwrap();
        let tensor = InteractionTensor::from_sparse(&gs_sparse, num_layers, num_qubits);

        let initial_partition: Vec<i32> =
            serde_json::from_value(tc["input_initial_partition"].clone()).unwrap();
        let mut placements = Array2::<i32>::zeros((num_layers + 1, num_qubits));
        for (q, &val) in initial_partition.iter().enumerate() {
            placements[[0, q]] = val;
        }

        let core_caps: Vec<usize> =
            serde_json::from_value(tc["input_core_capacities"].clone()).unwrap();
        let dist_vecs: Vec<Vec<i32>> =
            serde_json::from_value(tc["input_distance_matrix"].clone()).unwrap();
        let dist_flat: Vec<i32> = dist_vecs.into_iter().flatten().collect();
        let dist_array = Array2::from_shape_vec((num_cores, num_cores), dist_flat).unwrap();

        let result = hqa_mapping(
            &tensor,
            placements,
            num_cores,
            &core_caps,
            dist_array.view(),
        );
        let routing = extract_inter_core_communications(&result, dist_array.view());
        let report = estimate_fidelity(&tensor, &routing, &ArchitectureParams::default(), None);

        // Structural invariants
        assert!(
            report.algorithmic_fidelity > 0.0 && report.algorithmic_fidelity <= 1.0,
            "algorithmic fidelity must be in (0, 1]"
        );
        assert!(
            report.coherence_fidelity > 0.0 && report.coherence_fidelity <= 1.0,
            "coherence fidelity must be in (0, 1]"
        );
        assert!(
            (report.overall_fidelity
                - report.algorithmic_fidelity
                    * report.routing_fidelity
                    * report.coherence_fidelity)
                .abs()
                < 1e-12,
            "overall must equal algorithmic × routing × coherence"
        );
        assert!(
            report.total_circuit_time > 0.0,
            "circuit with gates must have positive time"
        );
        assert_eq!(report.layer_details.len(), num_layers);

        report
    }

    #[test]
    fn fidelity_all_to_all() {
        let report = run_fidelity_test("hqa_test_qft_25_all_to_all");
        assert!(
            report.overall_fidelity < 1.0,
            "circuit with teleportations should have fidelity < 1"
        );
    }

    #[test]
    fn fidelity_ring() {
        let report = run_fidelity_test("hqa_test_qft_25_ring");
        assert!(report.overall_fidelity < 1.0);
    }

    #[test]
    fn fidelity_large_cores() {
        run_fidelity_test("hqa_test_qft_25_large_cores");
    }

    #[test]
    fn ring_worse_than_all_to_all() {
        let all_to_all = run_fidelity_test("hqa_test_qft_25_all_to_all");
        let ring = run_fidelity_test("hqa_test_qft_25_ring");
        assert!(
            ring.total_circuit_time >= all_to_all.total_circuit_time,
            "ring should have equal or longer circuit time due to longer hops"
        );
    }

    #[test]
    fn zero_error_gives_perfect_operational_fidelity() {
        let path = Path::new("dse_pau/test_vectors.json");
        let data = fs::read_to_string(path).unwrap();
        let parsed: Value = serde_json::from_str(&data).unwrap();
        let tc = &parsed["hqa_test_qft_25_all_to_all"];

        let num_qubits = tc["num_virtual_qubits"].as_u64().unwrap() as usize;
        let num_cores = tc["num_cores"].as_u64().unwrap() as usize;
        let num_layers = tc["num_layers"].as_u64().unwrap() as usize;

        let gs_sparse: Vec<[f64; 4]> = serde_json::from_value(tc["gs_sparse"].clone()).unwrap();
        let tensor = InteractionTensor::from_sparse(&gs_sparse, num_layers, num_qubits);

        let initial_partition: Vec<i32> =
            serde_json::from_value(tc["input_initial_partition"].clone()).unwrap();
        let mut placements = Array2::<i32>::zeros((num_layers + 1, num_qubits));
        for (q, &val) in initial_partition.iter().enumerate() {
            placements[[0, q]] = val;
        }

        let core_caps: Vec<usize> =
            serde_json::from_value(tc["input_core_capacities"].clone()).unwrap();
        let dist_vecs: Vec<Vec<i32>> =
            serde_json::from_value(tc["input_distance_matrix"].clone()).unwrap();
        let dist_flat: Vec<i32> = dist_vecs.into_iter().flatten().collect();
        let dist_array = Array2::from_shape_vec((num_cores, num_cores), dist_flat).unwrap();

        let result = hqa_mapping(
            &tensor,
            placements,
            num_cores,
            &core_caps,
            dist_array.view(),
        );
        let routing = extract_inter_core_communications(&result, dist_array.view());

        let perfect_params = ArchitectureParams {
            single_gate_error: 0.0,
            two_gate_error: 0.0,
            teleportation_error_per_hop: 0.0,
            t1: f64::INFINITY,
            t2: f64::INFINITY,
            ..Default::default()
        };

        let report = estimate_fidelity(&tensor, &routing, &perfect_params, None);
        assert!(
            (report.algorithmic_fidelity - 1.0).abs() < 1e-12,
            "zero error rates should give perfect algorithmic fidelity"
        );
        assert!(
            (report.coherence_fidelity - 1.0).abs() < 1e-12,
            "infinite T1/T2 should give perfect coherence"
        );
    }
}
