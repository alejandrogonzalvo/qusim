use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};
use std::io::{self, Read};
use std::time::Instant;

use qusim::hqa::hqa_mapping;
use qusim::interactionTensor::InteractionTensor;

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
}

fn main() -> io::Result<()> {
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer)?;

    let input: HqaInput = serde_json::from_str(&buffer).expect("Failed to parse JSON input");

    let num_layers = input.num_layers;
    let num_qubits = input.num_virtual_qubits;
    let num_cores = input.num_cores;

    let mut gs_array = Array3::<f64>::zeros((num_layers, num_qubits, num_qubits));
    for edge in input.gs_sparse {
        let layer = edge[0] as usize;
        let u = edge[1] as usize;
        let v = edge[2] as usize;
        let w = edge[3];
        if layer < num_layers && u < num_qubits && v < num_qubits {
            gs_array[[layer, u, v]] = w;
        }
    }

    let dist_flat: Vec<i32> = input.input_distance_matrix.into_iter().flatten().collect();
    let dist_array = Array2::from_shape_vec((num_cores, num_cores), dist_flat)
        .expect("Failed to reshape distance_matrix into Array2");

    // Build ps as Array2: shape (num_layers + 1, num_qubits)
    let mut ps = Array2::<i32>::zeros((num_layers + 1, num_qubits));
    for (q, &val) in input.input_initial_partition.iter().enumerate() {
        ps[[0, q]] = val;
    }

    let interaction_tensor = InteractionTensor::new(gs_array.view());

    let start = Instant::now();

    let result = hqa_mapping(
        interaction_tensor,
        ps,
        num_cores,
        &input.input_core_capacities,
        dist_array.view(),
    );

    let duration = start.elapsed();

    let output = HqaOutput {
        execution_time_ms: duration.as_secs_f64() * 1000.0,
        final_partition: result.row(result.nrows() - 1).to_vec(),
    };

    println!("{}", serde_json::to_string(&output).unwrap());

    Ok(())
}
