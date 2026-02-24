use ndarray::{Array2, ArrayView2};

/// A single qubit teleportation between two cores.
#[derive(Debug, Clone)]
pub struct TeleportationEvent {
    /// Timeslice at which the qubit arrives (movement from timeslice-1 → timeslice).
    pub timeslice: usize,
    /// Virtual qubit index being teleported.
    pub qubit: usize,
    /// Core the qubit is leaving.
    pub source_core: usize,
    /// Core the qubit is arriving at.
    pub dest_core: usize,
    /// Hop count between source and destination from the distance matrix.
    pub network_distance: i32,
}

/// Aggregated routing result for the full mapped circuit.
#[derive(Debug, Clone)]
pub struct RoutingSummary {
    /// Every individual teleportation that occurs across the circuit.
    pub events: Vec<TeleportationEvent>,
    /// Total number of teleportations (== events.len()).
    pub total_teleportations: usize,
    /// Total EPR pairs consumed (1:1 with teleportations).
    pub total_epr_pairs: usize,
    /// Sum of all network distances across every teleportation.
    pub total_network_distance: i64,
    /// Number of teleportations that occur at each timeslice transition.
    pub communications_per_timeslice: Vec<usize>,
}

/// Extracts all inter-core teleportation events from consecutive placement rows.
///
/// For each timeslice transition `t-1 → t`, any qubit whose core assignment
/// changed requires a quantum teleportation. Each event is annotated with the
/// network distance from the architecture's distance matrix.
pub fn extract_inter_core_communications(
    placements: &Array2<i32>,
    distance_matrix: ArrayView2<'_, i32>,
) -> RoutingSummary {
    let num_rows = placements.nrows();
    let num_qubits = placements.ncols();
    let num_transitions = num_rows.saturating_sub(1);

    let mut events = Vec::new();
    let mut communications_per_timeslice = vec![0usize; num_transitions];

    for t in 1..num_rows {
        for q in 0..num_qubits {
            let prev_core = placements[[t - 1, q]];
            let curr_core = placements[[t, q]];

            if prev_core < 0 || curr_core < 0 || prev_core == curr_core {
                continue;
            }

            let src = prev_core as usize;
            let dst = curr_core as usize;

            events.push(TeleportationEvent {
                timeslice: t,
                qubit: q,
                source_core: src,
                dest_core: dst,
                network_distance: distance_matrix[[src, dst]],
            });

            communications_per_timeslice[t - 1] += 1;
        }
    }

    let total_teleportations = events.len();
    let total_network_distance: i64 = events.iter().map(|e| e.network_distance as i64).sum();

    RoutingSummary {
        events,
        total_teleportations,
        total_epr_pairs: total_teleportations,
        total_network_distance,
        communications_per_timeslice,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::InteractionTensor;
    use crate::hqa::hqa_mapping;
    use serde_json::Value;
    use std::fs;
    use std::path::Path;

    fn run_routing_test(test_case_name: &str) -> RoutingSummary {
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

        let summary = extract_inter_core_communications(&result, dist_array.view());

        // Structural invariants
        assert_eq!(
            summary.total_teleportations,
            summary.events.len(),
            "total_teleportations must equal events.len()"
        );
        assert_eq!(
            summary.total_epr_pairs, summary.total_teleportations,
            "EPR pairs must be 1:1 with teleportations"
        );

        let distance_sum: i64 = summary
            .events
            .iter()
            .map(|e| e.network_distance as i64)
            .sum();
        assert_eq!(
            summary.total_network_distance, distance_sum,
            "total_network_distance must be sum of event distances"
        );

        let event_count_sum: usize = summary.communications_per_timeslice.iter().sum();
        assert_eq!(
            event_count_sum, summary.total_teleportations,
            "per-timeslice counts must sum to total"
        );

        for event in &summary.events {
            assert_ne!(
                event.source_core, event.dest_core,
                "teleportation must be between different cores"
            );
            assert!(
                event.network_distance > 0,
                "network distance must be positive for distinct cores"
            );
        }

        summary
    }

    #[test]
    fn routing_all_to_all() {
        let summary = run_routing_test("hqa_test_qft_25_all_to_all");
        assert!(
            summary.total_teleportations > 0,
            "QFT on multiple cores should require teleportations"
        );
    }

    #[test]
    fn routing_ring() {
        let summary = run_routing_test("hqa_test_qft_25_ring");
        assert!(
            summary.total_teleportations > 0,
            "QFT on ring topology should require teleportations"
        );
    }

    #[test]
    fn routing_large_cores() {
        let summary = run_routing_test("hqa_test_qft_25_large_cores");
        assert!(
            !summary.communications_per_timeslice.is_empty(),
            "should have entries for timeslice transitions"
        );
    }

    #[test]
    fn no_teleportation_for_static_placement() {
        let placements = Array2::from_shape_vec((1, 4), vec![0, 0, 1, 1]).unwrap();
        let dist = Array2::from_shape_vec((2, 2), vec![0, 1, 1, 0]).unwrap();

        let summary = extract_inter_core_communications(&placements, dist.view());

        assert_eq!(summary.total_teleportations, 0);
        assert!(summary.events.is_empty());
        assert!(summary.communications_per_timeslice.is_empty());
    }

    #[test]
    fn detects_single_qubit_move() {
        let placements = Array2::from_shape_vec(
            (2, 3),
            vec![
                0, 1, 1, // t=0
                1, 1, 1, // t=1: qubit 0 moved
            ],
        )
        .unwrap();
        let dist = Array2::from_shape_vec((2, 2), vec![0, 3, 3, 0]).unwrap();

        let summary = extract_inter_core_communications(&placements, dist.view());

        assert_eq!(summary.total_teleportations, 1);
        assert_eq!(summary.events[0].qubit, 0);
        assert_eq!(summary.events[0].source_core, 0);
        assert_eq!(summary.events[0].dest_core, 1);
        assert_eq!(summary.events[0].network_distance, 3);
        assert_eq!(summary.total_network_distance, 3);
    }
}
