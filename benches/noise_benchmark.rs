use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array2;
use qusim::circuit::InteractionTensor;
use qusim::hqa::hqa_mapping;
use qusim::noise::{estimate_fidelity, ArchitectureParams};
use qusim::routing::extract_inter_core_communications;
use qusim::routing::RoutingSummary;
use serde_json::Value;
use std::fs;
use std::path::Path;

/// Helper: load a test case from the flat-sparse JSON format,
/// build an InteractionTensor directly, and return all inputs.
fn load_test_case(
    parsed: &Value,
    case_name: &str,
) -> (
    InteractionTensor,
    Array2<i32>,
    usize,
    Vec<usize>,
    Array2<i32>,
) {
    let test_case = &parsed[case_name];

    let num_qubits = test_case["num_virtual_qubits"].as_u64().unwrap() as usize;
    let num_cores = test_case["num_cores"].as_u64().unwrap() as usize;
    let num_layers = test_case["num_layers"].as_u64().unwrap() as usize;

    let gs_sparse: Vec<[f64; 4]> = serde_json::from_value(test_case["gs_sparse"].clone()).unwrap();
    let tensor = InteractionTensor::from_sparse(&gs_sparse, num_layers, num_qubits);

    let initial_partition: Vec<i32> =
        serde_json::from_value(test_case["input_initial_partition"].clone()).unwrap();
    let mut ps = Array2::<i32>::zeros((num_layers + 1, num_qubits));
    for (q, &val) in initial_partition.iter().enumerate() {
        ps[[0, q]] = val;
    }

    let core_caps: Vec<usize> =
        serde_json::from_value(test_case["input_core_capacities"].clone()).unwrap();

    let dist_vecs: Vec<Vec<i32>> =
        serde_json::from_value(test_case["input_distance_matrix"].clone()).unwrap();
    let dist_flat: Vec<i32> = dist_vecs.into_iter().flatten().collect();
    let dist_array = Array2::from_shape_vec((num_cores, num_cores), dist_flat).unwrap();

    (tensor, ps, num_cores, core_caps, dist_array)
}

fn prepare_fidelity_inputs(
    case_name: &str,
) -> (InteractionTensor, RoutingSummary, ArchitectureParams) {
    let path = Path::new("dse_pau/test_vectors.json");
    let data = fs::read_to_string(path).expect("Unable to read test_vectors.json");
    let parsed: Value = serde_json::from_str(&data).expect("Unable to parse test_vectors.json");

    let (tensor, ps, num_cores, core_caps, dist_array) = load_test_case(&parsed, case_name);

    // Run mapping and routing ONCE to generate the necessary data for the fidelity bencher
    let mapped_ps = hqa_mapping(&tensor, ps, num_cores, &core_caps, dist_array.view());
    let routing = extract_inter_core_communications(&mapped_ps, dist_array.view());

    let params = ArchitectureParams {
        single_gate_error: 1e-4,
        two_gate_error: 1e-3,
        teleportation_error_per_hop: 1e-2,
        single_gate_time: 20.0,
        two_gate_time: 100.0,
        teleportation_time_per_hop: 1000.0,
        t1: 100_000.0,
        t2: 50_000.0,
    };

    (tensor, routing, params)
}

fn bench_noise_estimation_25q(c: &mut Criterion) {
    let (tensor, routing, params) = prepare_fidelity_inputs("hqa_test_qft_25_all_to_all");

    c.bench_function("noise_estimation_25q", |b| {
        b.iter(|| {
            estimate_fidelity(
                black_box(&tensor),
                black_box(&routing),
                black_box(&params),
                None,
            )
        })
    });
}

fn bench_noise_scalability(c: &mut Criterion) {
    let path = Path::new("dse_pau/scalability_vectors.json");
    if !path.exists() {
        return;
    }
    let data = fs::read_to_string(path).expect("Unable to read scalability_vectors.json");
    let parsed: Value =
        serde_json::from_str(&data).expect("Unable to parse scalability_vectors.json");

    let mut group = c.benchmark_group("noise_scalability");
    group.sample_size(10);

    let core_counts = [5, 10, 15, 20, 25, 30];

    for &num_cores in &core_counts {
        let test_case = &parsed[num_cores.to_string()];
        let num_qubits = test_case["num_qubits"].as_u64().unwrap() as usize;

        // Convert old per-slice edge-list format to flat sparse
        let input_circuit: Vec<Vec<[f64; 3]>> =
            serde_json::from_value(test_case["gs"].clone()).unwrap();
        let num_layers = input_circuit.len();

        let mut gs_sparse_flat: Vec<[f64; 4]> = Vec::new();
        for (layer, slice_edges) in input_circuit.iter().enumerate() {
            for edge in slice_edges {
                gs_sparse_flat.push([layer as f64, edge[0], edge[1], edge[2]]);
            }
        }

        let tensor = InteractionTensor::from_sparse(&gs_sparse_flat, num_layers, num_qubits);

        let input_partitions: Vec<Vec<i32>> =
            serde_json::from_value(test_case["ps"].clone()).unwrap();
        let mut ps = Array2::<i32>::zeros((num_layers + 1, num_qubits));
        for (q, &val) in input_partitions[0].iter().enumerate() {
            ps[[0, q]] = val;
        }

        let core_capacities: Vec<usize> =
            serde_json::from_value(test_case["core_capacities"].clone()).unwrap();

        let dist_vecs: Vec<Vec<i32>> =
            serde_json::from_value(test_case["distance_matrix"].clone()).unwrap();
        let dist_flat: Vec<i32> = dist_vecs.into_iter().flatten().collect();
        let dist_array = Array2::from_shape_vec((num_cores, num_cores), dist_flat).unwrap();

        // Prepare the mapping and routing data just once
        let mapped_ps = hqa_mapping(&tensor, ps, num_cores, &core_capacities, dist_array.view());
        let routing = extract_inter_core_communications(&mapped_ps, dist_array.view());

        let params = ArchitectureParams {
            single_gate_error: 1e-4,
            two_gate_error: 1e-3,
            teleportation_error_per_hop: 1e-2,
            single_gate_time: 20.0,
            two_gate_time: 100.0,
            teleportation_time_per_hop: 1000.0,
            t1: 100_000.0,
            t2: 50_000.0,
        };

        group.bench_with_input(
            criterion::BenchmarkId::from_parameter(num_cores),
            &num_cores,
            |b, _| {
                b.iter(|| {
                    estimate_fidelity(
                        black_box(&tensor),
                        black_box(&routing),
                        black_box(&params),
                        None,
                    )
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_noise_estimation_25q, bench_noise_scalability);
criterion_main!(benches);
