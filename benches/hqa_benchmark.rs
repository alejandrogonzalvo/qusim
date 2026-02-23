use criterion::{black_box, criterion_group, criterion_main, Criterion};
use qusim::hqa::hqa_mapping;
use serde_json::Value;
use std::fs;
use std::path::Path;

fn bench_hqa_all_to_all(c: &mut Criterion) {
    let path = Path::new("dse_pau/test_vectors.json");
    let data = fs::read_to_string(path).expect("Unable to read test_vectors.json");
    let parsed: Value = serde_json::from_str(&data).expect("Unable to parse test_vectors.json");
    
    let test_case = &parsed["hqa_test_qft_25_all_to_all"];
    
    let num_virtual_qubits = test_case["num_virtual_qubits"].as_u64().unwrap() as usize;
    let num_cores = test_case["num_cores"].as_u64().unwrap() as usize;

    let input_circuit: Vec<Vec<[f64; 3]>> = serde_json::from_value(test_case["input_circuit"].clone()).unwrap();
    let mut gs: Vec<Vec<Vec<f64>>> = Vec::new();
    for slice_edges in input_circuit {
        let mut g = vec![vec![0.0; num_virtual_qubits]; num_virtual_qubits];
        for edge in slice_edges {
            let u = edge[0] as usize;
            let v = edge[1] as usize;
            let w = edge[2];
            g[u][v] = w;
        }
        gs.push(g);
    }

    let input_initial_partition: Vec<i32> = serde_json::from_value(test_case["input_initial_partition"].clone()).unwrap();
    let ps = vec![input_initial_partition];
    
    let core_capacities: Vec<usize> = serde_json::from_value(test_case["input_core_capacities"].clone()).unwrap();
    let distance_matrix: Vec<Vec<i32>> = serde_json::from_value(test_case["input_distance_matrix"].clone()).unwrap();

    c.bench_function("hqa_all_to_all_25q", |b| {
        b.iter(|| {
            hqa_mapping(
                black_box(&gs),
                black_box(ps.clone()),
                black_box(num_cores),
                black_box(num_virtual_qubits),
                black_box(&core_capacities),
                black_box(&distance_matrix),
            )
        })
    });
}

fn bench_hqa_ring(c: &mut Criterion) {
    let path = Path::new("dse_pau/test_vectors.json");
    let data = fs::read_to_string(path).expect("Unable to read test_vectors.json");
    let parsed: Value = serde_json::from_str(&data).expect("Unable to parse test_vectors.json");
    
    let test_case = &parsed["hqa_test_qft_25_ring"];
    
    let num_virtual_qubits = test_case["num_virtual_qubits"].as_u64().unwrap() as usize;
    let num_cores = test_case["num_cores"].as_u64().unwrap() as usize;

    let input_circuit: Vec<Vec<[f64; 3]>> = serde_json::from_value(test_case["input_circuit"].clone()).unwrap();
    let mut gs: Vec<Vec<Vec<f64>>> = Vec::new();
    for slice_edges in input_circuit {
        let mut g = vec![vec![0.0; num_virtual_qubits]; num_virtual_qubits];
        for edge in slice_edges {
            let u = edge[0] as usize;
            let v = edge[1] as usize;
            let w = edge[2];
            g[u][v] = w;
        }
        gs.push(g);
    }

    let input_initial_partition: Vec<i32> = serde_json::from_value(test_case["input_initial_partition"].clone()).unwrap();
    let ps = vec![input_initial_partition];
    
    let core_capacities: Vec<usize> = serde_json::from_value(test_case["input_core_capacities"].clone()).unwrap();
    let distance_matrix: Vec<Vec<i32>> = serde_json::from_value(test_case["input_distance_matrix"].clone()).unwrap();

    c.bench_function("hqa_ring_25q", |b| {
        b.iter(|| {
            hqa_mapping(
                black_box(&gs),
                black_box(ps.clone()),
                black_box(num_cores),
                black_box(num_virtual_qubits),
                black_box(&core_capacities),
                black_box(&distance_matrix),
            )
        })
    });
}

fn bench_hqa_large_cores(c: &mut Criterion) {
    let path = Path::new("dse_pau/test_vectors.json");
    let data = fs::read_to_string(path).expect("Unable to read test_vectors.json");
    let parsed: Value = serde_json::from_str(&data).expect("Unable to parse test_vectors.json");
    
    let test_case = &parsed["hqa_test_qft_25_large_cores"];
    
    let num_virtual_qubits = test_case["num_virtual_qubits"].as_u64().unwrap() as usize;
    let num_cores = test_case["num_cores"].as_u64().unwrap() as usize;

    let input_circuit: Vec<Vec<[f64; 3]>> = serde_json::from_value(test_case["input_circuit"].clone()).unwrap();
    let mut gs: Vec<Vec<Vec<f64>>> = Vec::new();
    for slice_edges in input_circuit {
        let mut g = vec![vec![0.0; num_virtual_qubits]; num_virtual_qubits];
        for edge in slice_edges {
            let u = edge[0] as usize;
            let v = edge[1] as usize;
            let w = edge[2];
            g[u][v] = w;
        }
        gs.push(g);
    }

    let input_initial_partition: Vec<i32> = serde_json::from_value(test_case["input_initial_partition"].clone()).unwrap();
    let ps = vec![input_initial_partition];
    
    let core_capacities: Vec<usize> = serde_json::from_value(test_case["input_core_capacities"].clone()).unwrap();
    let distance_matrix: Vec<Vec<i32>> = serde_json::from_value(test_case["input_distance_matrix"].clone()).unwrap();

    c.bench_function("hqa_large_cores_25q", |b| {
        b.iter(|| {
            hqa_mapping(
                black_box(&gs),
                black_box(ps.clone()),
                black_box(num_cores),
                black_box(num_virtual_qubits),
                black_box(&core_capacities),
                black_box(&distance_matrix),
            )
        })
    });
}

criterion_group!(benches, bench_hqa_all_to_all, bench_hqa_ring, bench_hqa_large_cores);
criterion_main!(benches);
