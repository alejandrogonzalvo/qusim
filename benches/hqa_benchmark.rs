use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{Array2, Array3};
use qusim::hqa::hqa_mapping;
use qusim::interactionTensor::InteractionTensor;
use serde_json::Value;
use std::fs;
use std::path::Path;

/// Helper: load a test case from the new flat-sparse JSON format,
/// reconstruct ndarray structures, and return all benchmark inputs.
fn load_test_case(
    parsed: &Value,
    case_name: &str,
) -> (Array3<f64>, Array2<i32>, usize, Vec<usize>, Array2<i32>) {
    let test_case = &parsed[case_name];

    let num_qubits = test_case["num_virtual_qubits"].as_u64().unwrap() as usize;
    let num_cores = test_case["num_cores"].as_u64().unwrap() as usize;
    let num_layers = test_case["num_layers"].as_u64().unwrap() as usize;

    let gs_sparse: Vec<[f64; 4]> = serde_json::from_value(test_case["gs_sparse"].clone()).unwrap();

    let mut gs_array = Array3::<f64>::zeros((num_layers, num_qubits, num_qubits));
    for edge in &gs_sparse {
        let layer = edge[0] as usize;
        let u = edge[1] as usize;
        let v = edge[2] as usize;
        let w = edge[3];
        if layer < num_layers && u < num_qubits && v < num_qubits {
            gs_array[[layer, u, v]] = w;
        }
    }

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

    (gs_array, ps, num_cores, core_caps, dist_array)
}

fn bench_hqa_all_to_all(c: &mut Criterion) {
    let path = Path::new("dse_pau/test_vectors.json");
    let data = fs::read_to_string(path).expect("Unable to read test_vectors.json");
    let parsed: Value = serde_json::from_str(&data).expect("Unable to parse test_vectors.json");

    let (gs_array, ps, num_cores, core_caps, dist_array) =
        load_test_case(&parsed, "hqa_test_qft_25_all_to_all");

    c.bench_function("hqa_all_to_all_25q", |b| {
        b.iter(|| {
            let tensor = InteractionTensor::new(black_box(gs_array.view()));
            hqa_mapping(
                tensor,
                black_box(ps.clone()),
                black_box(num_cores),
                black_box(&core_caps),
                black_box(dist_array.view()),
            )
        })
    });
}

fn bench_hqa_ring(c: &mut Criterion) {
    let path = Path::new("dse_pau/test_vectors.json");
    let data = fs::read_to_string(path).expect("Unable to read test_vectors.json");
    let parsed: Value = serde_json::from_str(&data).expect("Unable to parse test_vectors.json");

    let (gs_array, ps, num_cores, core_caps, dist_array) =
        load_test_case(&parsed, "hqa_test_qft_25_ring");

    c.bench_function("hqa_ring_25q", |b| {
        b.iter(|| {
            let tensor = InteractionTensor::new(black_box(gs_array.view()));
            hqa_mapping(
                tensor,
                black_box(ps.clone()),
                black_box(num_cores),
                black_box(&core_caps),
                black_box(dist_array.view()),
            )
        })
    });
}

fn bench_hqa_scalability(c: &mut Criterion) {
    let path = Path::new("dse_pau/scalability_vectors.json");
    if !path.exists() {
        return;
    }
    let data = fs::read_to_string(path).expect("Unable to read scalability_vectors.json");
    let parsed: Value =
        serde_json::from_str(&data).expect("Unable to parse scalability_vectors.json");

    let mut group = c.benchmark_group("hqa_scalability");
    group.sample_size(10);

    let core_counts = [5, 10, 15, 20, 25, 30];

    for &num_cores in &core_counts {
        let test_case = &parsed[num_cores.to_string()];
        let num_qubits = test_case["num_qubits"].as_u64().unwrap() as usize;

        // Old per-slice edge-list format from scalability_vectors.json
        let input_circuit: Vec<Vec<[f64; 3]>> =
            serde_json::from_value(test_case["gs"].clone()).unwrap();
        let num_layers = input_circuit.len();
        let mut gs_array = Array3::<f64>::zeros((num_layers, num_qubits, num_qubits));
        for (layer, slice_edges) in input_circuit.iter().enumerate() {
            for edge in slice_edges {
                let u = edge[0] as usize;
                let v = edge[1] as usize;
                let w = edge[2];
                gs_array[[layer, u, v]] = w;
            }
        }

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

        group.bench_with_input(
            criterion::BenchmarkId::from_parameter(num_cores),
            &num_cores,
            |b, _| {
                b.iter(|| {
                    let tensor = InteractionTensor::new(black_box(gs_array.view()));
                    hqa_mapping(
                        tensor,
                        black_box(ps.clone()),
                        black_box(num_cores),
                        black_box(&core_capacities),
                        black_box(dist_array.view()),
                    )
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_hqa_all_to_all,
    bench_hqa_ring,
    bench_hqa_scalability
);
criterion_main!(benches);
