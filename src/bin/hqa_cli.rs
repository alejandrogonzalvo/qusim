use serde::{Deserialize, Serialize};
use std::io::{self, Read};
use qusim::hqa::hqa_mapping;
use std::time::Instant;

#[derive(Deserialize)]
struct HqaInput {
    gs: Vec<Vec<[f64; 3]>>, // List of slices, each slice is [u, v, w]
    ps: Vec<Vec<i32>>,
    num_cores: usize,
    num_qubits: usize,
    core_capacities: Vec<usize>,
    distance_matrix: Vec<Vec<i32>>,
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

    // Reconstruct dense matrices
    let mut gs_dense = Vec::with_capacity(input.gs.len());
    for slice_edges in input.gs {
        let mut g = vec![vec![0.0; input.num_qubits]; input.num_qubits];
        for edge in slice_edges {
            let u = edge[0] as usize;
            let v = edge[1] as usize;
            let w = edge[2];
            if u < input.num_qubits && v < input.num_qubits {
                g[u][v] = w;
            }
        }
        gs_dense.push(g);
    }

    let start = Instant::now();
    let result = hqa_mapping(
        &gs_dense,
        input.ps,
        input.num_cores,
        input.num_qubits,
        &input.core_capacities,
        &input.distance_matrix,
    );
    let duration = start.elapsed();

    let output = HqaOutput {
        execution_time_ms: duration.as_secs_f64() * 1000.0,
        final_partition: result.last().cloned().unwrap_or_default(),
    };

    println!("{}", serde_json::to_string(&output).unwrap());

    Ok(())
}
