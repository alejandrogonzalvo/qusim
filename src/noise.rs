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
    /// Product of all gate and teleportation fidelities.
    pub operational_fidelity: f64,
    /// Fidelity loss from qubit idle time (T1/T2 decoherence).
    pub coherence_fidelity: f64,
    /// Overall fidelity = operational × coherence.
    pub overall_fidelity: f64,
    /// Total circuit execution time in nanoseconds.
    pub total_circuit_time: f64,
    /// Per-layer breakdown.
    pub layer_details: Vec<LayerFidelity>,
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
///
/// Combines operational fidelity (gate + teleportation errors) with
/// coherence fidelity (T1/T2 decoherence from idle time) to produce
/// an overall fidelity estimate.
pub fn estimate_fidelity(
    tensor: &InteractionTensor,
    routing: &RoutingSummary,
    params: &ArchitectureParams,
) -> FidelityReport {
    let num_layers = tensor.num_layers();
    let num_qubits = tensor.num_qubits();

    let mut overall_operational = 1.0_f64;
    let mut total_circuit_time = 0.0_f64;
    let mut qubit_busy_time = vec![0.0_f64; num_qubits];
    let mut layer_details = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        let gates = tensor.layer_gates(layer);
        let num_gates = gates.len();

        // Teleportation events for this layer (timeslice = layer + 1)
        let layer_teleportations: Vec<_> = routing
            .events
            .iter()
            .filter(|e| e.timeslice == layer + 1)
            .collect();
        let num_teleportations = layer_teleportations.len();

        // Timing: teleportation phase (limited by longest hop) + gate phase
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
        let layer_time = teleportation_time + gate_time;

        // Operational fidelity for this layer
        let mut layer_fidelity = 1.0_f64;
        for _ in 0..num_gates {
            layer_fidelity *= gate_fidelity(params.two_gate_error);
        }
        for event in &layer_teleportations {
            layer_fidelity *=
                teleportation_fidelity(params.teleportation_error_per_hop, event.network_distance);
        }

        overall_operational *= layer_fidelity;
        total_circuit_time += layer_time;

        // Track which qubits are busy this layer
        let mut busy: HashSet<usize> = HashSet::new();
        for &(u, v, _) in gates {
            busy.insert(u);
            busy.insert(v);
        }
        for event in &layer_teleportations {
            busy.insert(event.qubit);
        }
        for &q in &busy {
            qubit_busy_time[q] += layer_time;
        }

        layer_details.push(LayerFidelity {
            layer,
            num_gates,
            num_teleportations,
            layer_time,
            operational_fidelity: layer_fidelity,
        });
    }

    // Coherence fidelity: product of per-qubit decoherence
    let mut coherence_fidelity = 1.0_f64;
    for q in 0..num_qubits {
        let idle_time = (total_circuit_time - qubit_busy_time[q]).max(0.0);
        coherence_fidelity *= decoherence_fidelity(idle_time, params.t1, params.t2);
    }

    FidelityReport {
        operational_fidelity: overall_operational,
        coherence_fidelity,
        overall_fidelity: overall_operational * coherence_fidelity,
        total_circuit_time,
        layer_details,
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
        let report = estimate_fidelity(&tensor, &routing, &ArchitectureParams::default());

        // Structural invariants
        assert!(
            report.operational_fidelity > 0.0 && report.operational_fidelity <= 1.0,
            "operational fidelity must be in (0, 1]"
        );
        assert!(
            report.coherence_fidelity > 0.0 && report.coherence_fidelity <= 1.0,
            "coherence fidelity must be in (0, 1]"
        );
        assert!(
            (report.overall_fidelity - report.operational_fidelity * report.coherence_fidelity)
                .abs()
                < 1e-12,
            "overall must equal operational × coherence"
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

        let report = estimate_fidelity(&tensor, &routing, &perfect_params);
        assert!(
            (report.operational_fidelity - 1.0).abs() < 1e-12,
            "zero error rates should give perfect operational fidelity"
        );
        assert!(
            (report.coherence_fidelity - 1.0).abs() < 1e-12,
            "infinite T1/T2 should give perfect coherence"
        );
    }
}
