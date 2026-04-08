use std::collections::HashMap;

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
    /// Per-qubit single-gate error rates. Falls back to scalar `single_gate_error` if None.
    pub single_gate_error_per_qubit: Option<Vec<f64>>,
    /// Per-pair two-gate error rates keyed by (q1, q2). Falls back to scalar `two_gate_error` if None or pair missing.
    pub two_gate_error_per_pair: Option<HashMap<(usize, usize), f64>>,
    /// Per-qubit T1 relaxation times in nanoseconds. Falls back to scalar `t1` if None.
    pub t1_per_qubit: Option<Vec<f64>>,
    /// Per-qubit T2 dephasing times in nanoseconds. Falls back to scalar `t2` if None.
    pub t2_per_qubit: Option<Vec<f64>>,
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
            single_gate_error_per_qubit: None,
            two_gate_error_per_pair: None,
            t1_per_qubit: None,
            t2_per_qubit: None,
        }
    }
}

impl ArchitectureParams {
    /// Returns the single-gate error for qubit `q`, using per-qubit value if available.
    #[inline]
    pub fn single_gate_error_for(&self, q: usize) -> f64 {
        self.single_gate_error_per_qubit
            .as_ref()
            .map_or(self.single_gate_error, |v| v[q])
    }

    /// Returns the two-gate error for the pair `(u, v)`, using per-pair value if available.
    #[inline]
    pub fn two_gate_error_for(&self, u: usize, v: usize) -> f64 {
        self.two_gate_error_per_pair
            .as_ref()
            .and_then(|m| m.get(&(u, v)).copied())
            .unwrap_or(self.two_gate_error)
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

/// Convert a gate error rate to the depolarization parameter λ (paper Eq. 39).
/// d = 2 for single-qubit gates, d = 4 for two-qubit gates.
#[inline]
fn depolarization_lambda(gate_error: f64, d: f64) -> f64 {
    d * gate_error / (d - 1.0)
}

/// Teleportation fidelity decays exponentially with network distance.
#[inline]
fn teleportation_fidelity(error_per_hop: f64, distance: i32) -> f64 {
    (1.0 - error_per_hop).powi(distance)
}

/// Idle-qubit decoherence from T1 relaxation and T2 dephasing (paper Eq. 41).
#[inline]
fn decoherence_fidelity(idle_time: f64, t1: f64, t2: f64) -> f64 {
    (-idle_time / t1).exp() * (0.5 * (-idle_time / t2).exp() + 0.5)
}

fn parse_sparse_swaps(
    sparse_swaps: Option<ndarray::ArrayView2<i32>>,
    num_layers: usize,
) -> Vec<Vec<(usize, usize)>> {
    let mut layer_swaps = vec![Vec::new(); num_layers];

    let Some(swaps) = sparse_swaps else {
        return layer_swaps;
    };

    for row in swaps.rows() {
        if row.len() != 3 {
            continue;
        }

        let layer_idx = row[0] as usize;
        if layer_idx >= num_layers {
            continue;
        }

        let q1 = row[1] as usize;
        let q2 = row[2] as usize;
        layer_swaps[layer_idx].push((q1, q2));
    }

    layer_swaps
}

/// Estimates the overall fidelity of a mapped quantum circuit.
pub fn estimate_fidelity(
    tensor: &InteractionTensor,
    routing: &RoutingSummary,
    params: &ArchitectureParams,
    sparse_swaps: Option<ndarray::ArrayView2<i32>>,
) -> FidelityReport {
    let num_layers = tensor.num_layers();
    let num_qubits = tensor.num_qubits();

    let mut overall_routing = 1.0_f64;

    let mut total_circuit_time = 0.0_f64;

    let mut layer_details = Vec::with_capacity(num_layers);

    let layer_swaps = parse_sparse_swaps(sparse_swaps, num_layers);

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
                coherence_fidelity_grid[layer * num_qubits + q] =
                    coherence_fidelity_grid[(layer - 1) * num_qubits + q];
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

        let layer_time = calculate_layer_time(
            &layer_teleportations,
            num_gates,
            &layer_swaps[layer],
            num_qubits,
            params,
        );
        total_circuit_time += layer_time;

        let mut layer_busy_time = vec![0.0_f64; num_qubits];

        // 1. Process computational gates (Algorithmic)
        let layer_algo_grid =
            &mut algorithmic_fidelity_grid[layer * num_qubits..(layer + 1) * num_qubits];
        let layer_algorithmic_fidelity =
            process_computational_gates(gates, params, &mut layer_busy_time, layer_algo_grid);

        // 2. Process SabreRouting injected SWAP gates (Routing Overhead)
        let layer_routing_grid =
            &mut routing_fidelity_grid[layer * num_qubits..(layer + 1) * num_qubits];
        let mut layer_routing_fidelity = process_sabre_swaps(
            &layer_swaps[layer],
            params,
            &mut layer_busy_time,
            layer_routing_grid,
        );

        // 3. Process teleportations (Routing Overhead)
        layer_routing_fidelity *= process_teleportations(
            &layer_teleportations,
            params,
            &mut layer_busy_time,
            layer_routing_grid,
        );

        overall_routing *= layer_routing_fidelity;

        // 4. Update coherence based on per-layer idle time
        let layer_coh_grid =
            &mut coherence_fidelity_grid[layer * num_qubits..(layer + 1) * num_qubits];
        update_busy_and_coherence(
            layer_time,
            params,
            &layer_busy_time,
            layer_coh_grid,
        );

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

    // Overall algorithmic fidelity = product of per-qubit fidelities at final layer (paper Eq. 40)
    let mut overall_algorithmic_from_grid = 1.0_f64;
    if num_layers > 0 {
        for q in 0..num_qubits {
            overall_algorithmic_from_grid *= algorithmic_fidelity_grid[(num_layers - 1) * num_qubits + q];
        }
    }

    FidelityReport {
        algorithmic_fidelity: overall_algorithmic_from_grid,
        routing_fidelity: overall_routing,
        coherence_fidelity: global_coherence,
        overall_fidelity: overall_algorithmic_from_grid * overall_routing * global_coherence,
        total_circuit_time,
        layer_details,
        algorithmic_fidelity_grid,
        routing_fidelity_grid,
        coherence_fidelity_grid,
    }
}

// ---------------------------------------------------------------------------
// Private SRP Helpers for Hardware Fidelity Estimation
// ---------------------------------------------------------------------------

#[inline]
fn calculate_layer_time(
    layer_teleportations: &[&crate::routing::TeleportationEvent],
    num_gates: usize,
    swaps: &[(usize, usize)],
    num_qubits: usize,
    params: &ArchitectureParams,
) -> f64 {
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

    let mut swap_counts = vec![0; num_qubits];
    for &(q1, q2) in swaps {
        if q1 < num_qubits {
            swap_counts[q1] += 1;
        }
        if q2 < num_qubits {
            swap_counts[q2] += 1;
        }
    }
    let max_swaps_in_layer = swap_counts.into_iter().max().unwrap_or(0);
    let swap_time = max_swaps_in_layer as f64 * params.two_gate_time * 3.0; // SWAP is ~3 CX gates
    teleportation_time + gate_time + swap_time
}

#[inline]
fn process_computational_gates(
    gates: &[(usize, usize, f64)],
    params: &ArchitectureParams,
    layer_busy_time: &mut [f64],
    layer_algo_grid: &mut [f64],
) -> f64 {
    let mut layer_algo_fidelity = 1.0;
    for &(u, v, _) in gates {
        if u == v {
            // Single-qubit gate: F_q = (1-λ)·F_q + λ/d, with d=2 (paper Algorithm 1)
            layer_busy_time[u] += params.single_gate_time;
            let error = params.single_gate_error_for(u);
            let lam = depolarization_lambda(error, 2.0);
            let f_before = layer_algo_grid[u];
            layer_algo_grid[u] = (1.0 - lam) * layer_algo_grid[u] + lam / 2.0;
            if f_before > 0.0 {
                layer_algo_fidelity *= layer_algo_grid[u] / f_before;
            }
        } else {
            // Two-qubit gate: η correction (paper Eq. 25, Algorithm 1)
            layer_busy_time[u] += params.two_gate_time;
            layer_busy_time[v] += params.two_gate_time;
            let error = params.two_gate_error_for(u, v);
            let lam = depolarization_lambda(error, 4.0);
            let f1 = layer_algo_grid[u];
            let f2 = layer_algo_grid[v];
            let f1_before = f1;
            let f2_before = f2;

            let sqrt_1_lam = (1.0 - lam).sqrt();
            let eta = 0.5 * (((1.0 - lam) * (f1 + f2).powi(2) + lam).sqrt()
                           - sqrt_1_lam * (f1 + f2));

            layer_algo_grid[u] = sqrt_1_lam * f1 + eta;
            layer_algo_grid[v] = sqrt_1_lam * f2 + eta;

            if f1_before > 0.0 {
                layer_algo_fidelity *= layer_algo_grid[u] / f1_before;
            }
            if f2_before > 0.0 {
                layer_algo_fidelity *= layer_algo_grid[v] / f2_before;
            }
        }
    }
    layer_algo_fidelity
}

#[inline]
fn process_sabre_swaps(
    swaps: &[(usize, usize)],
    params: &ArchitectureParams,
    layer_busy_time: &mut [f64],
    layer_routing_grid: &mut [f64],
) -> f64 {
    if swaps.is_empty() {
        return 1.0;
    }

    let mut layer_routing_fidelity = 1.0;

    for &(q1, q2) in swaps {
        let error = params.two_gate_error_for(q1, q2);
        let f_swap = gate_fidelity(3.0 * error); // SWAP is ~3 CX gates
        if q1 < layer_busy_time.len() {
            layer_busy_time[q1] += params.two_gate_time * 3.0;
            layer_routing_grid[q1] *= f_swap;
        }
        if q2 < layer_busy_time.len() {
            layer_busy_time[q2] += params.two_gate_time * 3.0;
            layer_routing_grid[q2] *= f_swap;
        }
        layer_routing_fidelity *= f_swap;
    }
    layer_routing_fidelity
}

#[inline]
fn process_teleportations(
    layer_teleportations: &[&crate::routing::TeleportationEvent],
    params: &ArchitectureParams,
    layer_busy_time: &mut [f64],
    layer_routing_grid: &mut [f64],
) -> f64 {
    let mut layer_routing_fidelity = 1.0;
    for event in layer_teleportations {
        layer_busy_time[event.qubit] +=
            event.network_distance as f64 * params.teleportation_time_per_hop;

        let f = teleportation_fidelity(params.teleportation_error_per_hop, event.network_distance);
        layer_routing_grid[event.qubit] *= f;
        layer_routing_fidelity *= f;
    }
    layer_routing_fidelity
}

#[inline]
fn update_busy_and_coherence(
    layer_time: f64,
    params: &ArchitectureParams,
    layer_busy_time: &[f64],
    layer_coh_grid: &mut [f64],
) {
    for q in 0..layer_coh_grid.len() {
        // Idle time is how long the qubit was idle during THIS layer only
        let layer_idle = (layer_time - layer_busy_time[q]).max(0.0);
        if layer_idle > 0.0 {
            let q_t1 = params.t1_per_qubit.as_ref().map_or(params.t1, |v| v[q]);
            let q_t2 = params.t2_per_qubit.as_ref().map_or(params.t2, |v| v[q]);
            layer_coh_grid[q] *= decoherence_fidelity(layer_idle, q_t1, q_t2);
        }
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

    // -----------------------------------------------------------------------
    // Helper: set up tensor + routing from a test vector for reuse
    // -----------------------------------------------------------------------
    fn setup_from_test_vector(test_case_name: &str) -> (InteractionTensor, RoutingSummary, usize) {
        let path = Path::new("dse_pau/test_vectors.json");
        let data = fs::read_to_string(path).unwrap();
        let parsed: Value = serde_json::from_str(&data).unwrap();
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

        (tensor, routing, num_qubits)
    }

    // -----------------------------------------------------------------------
    // Regression: uniform per-qubit gate-error arrays == scalar gate errors
    // -----------------------------------------------------------------------
    #[test]
    fn uniform_per_qubit_gate_errors_match_scalar() {
        let (tensor, routing, num_qubits) = setup_from_test_vector("hqa_test_qft_25_all_to_all");

        let scalar_params = ArchitectureParams::default();
        let scalar_report = estimate_fidelity(&tensor, &routing, &scalar_params, None);

        // Build per-qubit arrays that are identical to the scalar defaults
        let per_qubit_params = ArchitectureParams {
            single_gate_error_per_qubit: Some(vec![scalar_params.single_gate_error; num_qubits]),
            two_gate_error_per_pair: None,
            ..Default::default()
        };
        let per_qubit_report = estimate_fidelity(&tensor, &routing, &per_qubit_params, None);

        assert!(
            (scalar_report.algorithmic_fidelity - per_qubit_report.algorithmic_fidelity).abs() < 1e-12,
            "uniform per-qubit 1Q errors must match scalar: {} vs {}",
            scalar_report.algorithmic_fidelity,
            per_qubit_report.algorithmic_fidelity
        );
        assert!(
            (scalar_report.routing_fidelity - per_qubit_report.routing_fidelity).abs() < 1e-12,
            "routing fidelity must be identical when only 1Q errors differ in representation"
        );
        assert!(
            (scalar_report.coherence_fidelity - per_qubit_report.coherence_fidelity).abs() < 1e-12,
            "coherence must be identical"
        );
        assert!(
            (scalar_report.overall_fidelity - per_qubit_report.overall_fidelity).abs() < 1e-12,
            "overall fidelity must match"
        );
    }

    // -----------------------------------------------------------------------
    // Regression: uniform per-pair 2Q error map == scalar two_gate_error
    // -----------------------------------------------------------------------
    #[test]
    fn uniform_per_pair_two_gate_errors_match_scalar() {
        let (tensor, routing, _num_qubits) = setup_from_test_vector("hqa_test_qft_25_all_to_all");

        let scalar_params = ArchitectureParams::default();
        let scalar_report = estimate_fidelity(&tensor, &routing, &scalar_params, None);

        // Build a per-pair map where every pair has the same error as the scalar default.
        // We use an empty map to signal "use scalar fallback" — same as None.
        // But here we explicitly set the map to verify the lookup path works.
        use std::collections::HashMap;
        let mut pair_map = HashMap::new();
        // Populate with all pairs that appear in the tensor
        for layer in 0..tensor.num_layers() {
            for &(u, v, _) in tensor.layer_gates(layer) {
                if u != v {
                    pair_map.insert((u, v), scalar_params.two_gate_error);
                    pair_map.insert((v, u), scalar_params.two_gate_error);
                }
            }
        }

        let per_pair_params = ArchitectureParams {
            two_gate_error_per_pair: Some(pair_map),
            ..Default::default()
        };
        let per_pair_report = estimate_fidelity(&tensor, &routing, &per_pair_params, None);

        assert!(
            (scalar_report.algorithmic_fidelity - per_pair_report.algorithmic_fidelity).abs() < 1e-12,
            "uniform per-pair 2Q errors must match scalar: {} vs {}",
            scalar_report.algorithmic_fidelity,
            per_pair_report.algorithmic_fidelity
        );
        assert!(
            (scalar_report.overall_fidelity - per_pair_report.overall_fidelity).abs() < 1e-12,
            "overall fidelity must match"
        );
    }

    // -----------------------------------------------------------------------
    // New behavior: worse per-qubit 1Q error on one qubit degrades fidelity
    // -----------------------------------------------------------------------
    #[test]
    fn per_qubit_single_gate_error_degrades_fidelity() {
        let (tensor, routing, num_qubits) = setup_from_test_vector("hqa_test_qft_25_all_to_all");

        let uniform_params = ArchitectureParams::default();
        let uniform_report = estimate_fidelity(&tensor, &routing, &uniform_params, None);

        // Give qubit 0 a 100x worse single-gate error
        let mut sq_errors = vec![uniform_params.single_gate_error; num_qubits];
        sq_errors[0] = uniform_params.single_gate_error * 100.0;

        let degraded_params = ArchitectureParams {
            single_gate_error_per_qubit: Some(sq_errors),
            ..Default::default()
        };
        let degraded_report = estimate_fidelity(&tensor, &routing, &degraded_params, None);

        assert!(
            degraded_report.algorithmic_fidelity < uniform_report.algorithmic_fidelity,
            "degraded qubit 0 should lower algorithmic fidelity: {} vs {}",
            degraded_report.algorithmic_fidelity,
            uniform_report.algorithmic_fidelity
        );
        // Coherence should be unchanged (gate errors don't affect T1/T2)
        assert!(
            (degraded_report.coherence_fidelity - uniform_report.coherence_fidelity).abs() < 1e-12,
            "coherence must be unaffected by gate error changes"
        );
    }

    // -----------------------------------------------------------------------
    // New behavior: worse per-pair 2Q error on specific pairs degrades fidelity
    // -----------------------------------------------------------------------
    #[test]
    fn per_pair_two_gate_error_degrades_fidelity() {
        let (tensor, routing, _num_qubits) = setup_from_test_vector("hqa_test_qft_25_all_to_all");

        let uniform_params = ArchitectureParams::default();
        let uniform_report = estimate_fidelity(&tensor, &routing, &uniform_params, None);

        // Make ALL pairs 10x worse
        use std::collections::HashMap;
        let mut pair_map = HashMap::new();
        for layer in 0..tensor.num_layers() {
            for &(u, v, _) in tensor.layer_gates(layer) {
                if u != v {
                    let bad_error = (uniform_params.two_gate_error * 10.0).min(1.0);
                    pair_map.insert((u, v), bad_error);
                    pair_map.insert((v, u), bad_error);
                }
            }
        }

        let degraded_params = ArchitectureParams {
            two_gate_error_per_pair: Some(pair_map),
            ..Default::default()
        };
        let degraded_report = estimate_fidelity(&tensor, &routing, &degraded_params, None);

        assert!(
            degraded_report.algorithmic_fidelity < uniform_report.algorithmic_fidelity,
            "worse per-pair 2Q error should lower algorithmic fidelity: {} vs {}",
            degraded_report.algorithmic_fidelity,
            uniform_report.algorithmic_fidelity
        );
    }

    // -----------------------------------------------------------------------
    // New behavior: zero per-qubit errors give perfect fidelity
    // -----------------------------------------------------------------------
    #[test]
    fn zero_per_qubit_errors_give_perfect_algorithmic_fidelity() {
        let (tensor, routing, num_qubits) = setup_from_test_vector("hqa_test_qft_25_all_to_all");

        let perfect_params = ArchitectureParams {
            single_gate_error: 0.0,
            two_gate_error: 0.0,
            teleportation_error_per_hop: 0.0,
            single_gate_error_per_qubit: Some(vec![0.0; num_qubits]),
            two_gate_error_per_pair: Some(std::collections::HashMap::new()),
            t1: f64::INFINITY,
            t2: f64::INFINITY,
            ..Default::default()
        };
        let report = estimate_fidelity(&tensor, &routing, &perfect_params, None);

        assert!(
            (report.algorithmic_fidelity - 1.0).abs() < 1e-12,
            "zero per-qubit errors should give perfect algorithmic fidelity"
        );
    }

    // -----------------------------------------------------------------------
    // Bug fix: calculate_layer_time should distinguish 1Q-only vs 2Q layers
    // -----------------------------------------------------------------------
    #[test]
    fn layer_with_only_1q_gates_uses_single_gate_time() {
        // Build a minimal tensor: 2 layers, 4 qubits.
        // Layer 0: only 1Q gates (u==v), Layer 1: has a 2Q gate (u!=v).
        let edges = vec![
            [0.0, 0.0, 0.0, 1.0], // layer 0: 1Q gate on q0
            [0.0, 1.0, 1.0, 1.0], // layer 0: 1Q gate on q1
            [1.0, 0.0, 1.0, 1.0], // layer 1: 2Q gate on (q0, q1)
        ];
        let tensor = InteractionTensor::from_sparse(&edges, 2, 4);
        let routing = RoutingSummary {
            total_teleportations: 0,
            total_epr_pairs: 0,
            total_network_distance: 0,
            communications_per_timeslice: vec![0, 0],
            events: vec![],
        };

        let params = ArchitectureParams {
            single_gate_time: 36.0,
            two_gate_time: 68.0,
            t1: f64::INFINITY,
            t2: f64::INFINITY,
            ..Default::default()
        };

        let report = estimate_fidelity(&tensor, &routing, &params, None);

        // Layer 0 (1Q only) should take single_gate_time (36ns), NOT two_gate_time (68ns).
        // Layer 1 (has 2Q) should take two_gate_time (68ns).
        // Total = 36 + 68 = 104ns.
        assert!(
            (report.total_circuit_time - 104.0).abs() < 1e-6,
            "1Q-only layer should use single_gate_time: expected 104ns, got {}ns",
            report.total_circuit_time
        );
    }

    #[test]
    fn per_qubit_t1t2_differs_from_uniform() {
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

        // Uniform T1/T2
        let uniform_params = ArchitectureParams::default();
        let uniform_report = estimate_fidelity(&tensor, &routing, &uniform_params, None);

        // Per-qubit: give qubit 0 much worse T1/T2
        let mut t1_vec = vec![100_000.0; num_qubits];
        let mut t2_vec = vec![50_000.0; num_qubits];
        t1_vec[0] = 10_000.0; // 10x worse
        t2_vec[0] = 5_000.0;

        let per_qubit_params = ArchitectureParams {
            t1_per_qubit: Some(t1_vec),
            t2_per_qubit: Some(t2_vec),
            ..Default::default()
        };
        let per_qubit_report = estimate_fidelity(&tensor, &routing, &per_qubit_params, None);

        // Per-qubit with one bad qubit should have worse coherence
        assert!(
            per_qubit_report.coherence_fidelity < uniform_report.coherence_fidelity,
            "per-qubit with one degraded qubit should have worse coherence: {} vs {}",
            per_qubit_report.coherence_fidelity,
            uniform_report.coherence_fidelity
        );

        // Algorithmic fidelity should be identical (not affected by T1/T2)
        assert!(
            (per_qubit_report.algorithmic_fidelity - uniform_report.algorithmic_fidelity).abs() < 1e-12,
            "algorithmic fidelity should be unaffected by T1/T2 changes"
        );
    }
}
