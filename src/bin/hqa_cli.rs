use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::io::{self, Read};
use std::time::Instant;

use qusim::circuit::InteractionTensor;
use qusim::hqa::hqa_mapping;
use qusim::noise::{estimate_fidelity, ArchitectureParams};
use qusim::routing::extract_inter_core_communications;

#[derive(Deserialize)]
struct HqaInput {
    num_virtual_qubits: usize,
    num_cores: usize,
    num_layers: usize,

    gs_sparse: Vec<[f64; 4]>,

    input_initial_partition: Vec<i32>,
    input_core_capacities: Vec<usize>,
    input_distance_matrix: Vec<Vec<i32>>,
}

#[derive(Serialize)]
struct HqaOutput {
    execution_time_ms: f64,
    final_partition: Vec<i32>,
    total_teleportations: usize,
    total_epr_pairs: usize,
    total_network_distance: i64,
    communications_per_timeslice: Vec<usize>,
    algorithmic_fidelity: f64,
    routing_fidelity: f64,
    coherence_fidelity: f64,
    overall_fidelity: f64,
    total_circuit_time_ns: f64,
}

fn main() -> io::Result<()> {
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer)?;

    let input: HqaInput = serde_json::from_str(&buffer).expect("Failed to parse JSON input");

    let num_layers = input.num_layers;
    let num_qubits = input.num_virtual_qubits;
    let num_cores = input.num_cores;

    let tensor = InteractionTensor::from_sparse(&input.gs_sparse, num_layers, num_qubits);

    let dist_flat: Vec<i32> = input.input_distance_matrix.into_iter().flatten().collect();
    let dist_array = Array2::from_shape_vec((num_cores, num_cores), dist_flat)
        .expect("Failed to reshape distance_matrix into Array2");

    let mut ps = Array2::<i32>::zeros((num_layers + 1, num_qubits));
    for (q, &val) in input.input_initial_partition.iter().enumerate() {
        ps[[0, q]] = val;
    }

    let start = Instant::now();

    let result = hqa_mapping(
        &tensor,
        ps,
        num_cores,
        &input.input_core_capacities,
        dist_array.view(),
    );

    let duration = start.elapsed();

    let routing = extract_inter_core_communications(&result, dist_array.view());
    let fidelity = estimate_fidelity(&tensor, &routing, &ArchitectureParams::default(), None);

    let output = HqaOutput {
        execution_time_ms: duration.as_secs_f64() * 1000.0,
        final_partition: result.row(result.nrows() - 1).to_vec(),
        total_teleportations: routing.total_teleportations,
        total_epr_pairs: routing.total_epr_pairs,
        total_network_distance: routing.total_network_distance,
        communications_per_timeslice: routing.communications_per_timeslice,
        algorithmic_fidelity: fidelity.algorithmic_fidelity,
        routing_fidelity: fidelity.routing_fidelity,
        coherence_fidelity: fidelity.coherence_fidelity,
        overall_fidelity: fidelity.overall_fidelity,
        total_circuit_time_ns: fidelity.total_circuit_time,
    };

    println!("{}", serde_json::to_string(&output).unwrap());

    Ok(())
}
