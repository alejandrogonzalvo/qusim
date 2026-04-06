use std::collections::HashMap;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::circuit::InteractionTensor;
use crate::hqa::hqa_mapping;
use crate::noise::{estimate_fidelity, ArchitectureParams};
use crate::routing::extract_inter_core_communications;

/// Convert a Python dict of {(int,int): float} to a Rust HashMap<(usize,usize), f64>.
fn pydict_to_pair_map(dict: &Bound<'_, PyDict>) -> PyResult<HashMap<(usize, usize), f64>> {
    let mut map = HashMap::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let tuple = key.downcast::<PyTuple>()?;
        let q1: usize = tuple.get_item(0)?.extract()?;
        let q2: usize = tuple.get_item(1)?.extract()?;
        let err: f64 = value.extract()?;
        map.insert((q1, q2), err);
    }
    Ok(map)
}

/// Build ArchitectureParams from the common Python function arguments.
#[allow(clippy::too_many_arguments)]
fn build_params<'py>(
    single_gate_error: f64,
    two_gate_error: f64,
    teleportation_error_per_hop: f64,
    single_gate_time: f64,
    two_gate_time: f64,
    teleportation_time_per_hop: f64,
    t1: f64,
    t2: f64,
    single_gate_error_per_qubit: Option<&Bound<'py, PyArray1<f64>>>,
    two_gate_error_per_pair: Option<&Bound<'py, PyDict>>,
    t1_per_qubit: Option<&Bound<'py, PyArray1<f64>>>,
    t2_per_qubit: Option<&Bound<'py, PyArray1<f64>>>,
) -> PyResult<ArchitectureParams> {
    Ok(ArchitectureParams {
        single_gate_error,
        two_gate_error,
        teleportation_error_per_hop,
        single_gate_time,
        two_gate_time,
        teleportation_time_per_hop,
        t1,
        t2,
        single_gate_error_per_qubit: single_gate_error_per_qubit
            .map(|a| a.readonly().as_array().to_vec()),
        two_gate_error_per_pair: two_gate_error_per_pair
            .map(|d| pydict_to_pair_map(d))
            .transpose()?,
        t1_per_qubit: t1_per_qubit.map(|a| a.readonly().as_array().to_vec()),
        t2_per_qubit: t2_per_qubit.map(|a| a.readonly().as_array().to_vec()),
    })
}

/// Parse gs_sparse into (num_layers, num_qubits, edge_list).
fn parse_sparse_tensor(gs_sparse: &Bound<'_, PyArray2<f64>>, extra_qubits: usize) -> (usize, usize, Vec<[f64; 4]>) {
    let gs_sparse_rust = gs_sparse.readonly().as_array().to_owned();
    let mut num_layers = 0;
    let mut num_qubits = 0;

    for row in gs_sparse_rust.rows() {
        let layer_val = row[0] as usize;
        let u = row[1] as usize;
        let v = row[2] as usize;
        if layer_val + 1 > num_layers {
            num_layers = layer_val + 1;
        }
        if u + 1 > num_qubits {
            num_qubits = u + 1;
        }
        if v + 1 > num_qubits {
            num_qubits = v + 1;
        }
    }

    if extra_qubits > num_qubits {
        num_qubits = extra_qubits;
    }

    let mut edge_list: Vec<[f64; 4]> = Vec::with_capacity(gs_sparse_rust.shape()[0]);
    for row in gs_sparse_rust.rows() {
        edge_list.push([row[0], row[1], row[2], row[3]]);
    }

    (num_layers, num_qubits, edge_list)
}

#[pyfunction]
#[pyo3(signature = (
    gs_sparse,
    initial_partition,
    num_cores,
    core_capacities,
    distance_matrix,
    single_gate_error = 1e-4,
    two_gate_error = 1e-3,
    teleportation_error_per_hop = 1e-2,
    single_gate_time = 20.0,
    two_gate_time = 100.0,
    teleportation_time_per_hop = 1000.0,
    t1 = 100_000.0,
    t2 = 50_000.0,
    single_gate_error_per_qubit = None,
    two_gate_error_per_pair = None,
    t1_per_qubit = None,
    t2_per_qubit = None
))]
#[allow(clippy::too_many_arguments)]
pub fn map_and_estimate<'py>(
    py: Python<'py>,
    gs_sparse: &Bound<'py, PyArray2<f64>>,
    initial_partition: &Bound<'py, PyArray1<i32>>,
    num_cores: usize,
    core_capacities: &Bound<'py, PyArray1<usize>>,
    distance_matrix: &Bound<'py, PyArray2<i32>>,
    single_gate_error: f64,
    two_gate_error: f64,
    teleportation_error_per_hop: f64,
    single_gate_time: f64,
    two_gate_time: f64,
    teleportation_time_per_hop: f64,
    t1: f64,
    t2: f64,
    single_gate_error_per_qubit: Option<&Bound<'py, PyArray1<f64>>>,
    two_gate_error_per_pair: Option<&Bound<'py, PyDict>>,
    t1_per_qubit: Option<&Bound<'py, PyArray1<f64>>>,
    t2_per_qubit: Option<&Bound<'py, PyArray1<f64>>>,
) -> PyResult<Bound<'py, PyDict>> {
    let part_len = initial_partition.len().unwrap_or(0);
    let (num_layers, num_qubits, edge_list) = parse_sparse_tensor(gs_sparse, part_len);

    let tensor = InteractionTensor::from_sparse(&edge_list, num_layers, num_qubits);

    let mut placements = Array2::<i32>::zeros((num_layers + 1, num_qubits));
    let initial_part = initial_partition.readonly().as_array().to_owned();
    for (q, &val) in initial_part.iter().enumerate() {
        if q < num_qubits {
            placements[[0, q]] = val;
        }
    }

    let core_caps = core_capacities.readonly().as_array().to_vec();
    let dist_array = distance_matrix.readonly().as_array().to_owned();

    // 1. Map
    let result = hqa_mapping(
        &tensor,
        placements,
        num_cores,
        &core_caps,
        dist_array.view(),
    );

    // 2. Route
    let routing = extract_inter_core_communications(&result, dist_array.view());

    // 3. Noise Model
    let params = build_params(
        single_gate_error, two_gate_error, teleportation_error_per_hop,
        single_gate_time, two_gate_time, teleportation_time_per_hop,
        t1, t2,
        single_gate_error_per_qubit, two_gate_error_per_pair,
        t1_per_qubit, t2_per_qubit,
    )?;
    let fidelity = estimate_fidelity(&tensor, &routing, &params, None);

    // Build the output Python dict
    let dict = PyDict::new_bound(py);

    dict.set_item("execution_success", true)?;
    dict.set_item("placements", result.into_pyarray_bound(py))?;

    dict.set_item("total_teleportations", routing.total_teleportations)?;
    dict.set_item("total_epr_pairs", routing.total_epr_pairs)?;
    dict.set_item("total_network_distance", routing.total_network_distance)?;

    let tps_py = PyList::new_bound(py, routing.communications_per_timeslice);
    dict.set_item("teleportations_per_slice", tps_py)?;

    dict.set_item("algorithmic_fidelity", fidelity.algorithmic_fidelity)?;
    dict.set_item("routing_fidelity", fidelity.routing_fidelity)?;
    dict.set_item("coherence_fidelity", fidelity.coherence_fidelity)?;
    dict.set_item("overall_fidelity", fidelity.overall_fidelity)?;
    dict.set_item("total_circuit_time_ns", fidelity.total_circuit_time)?;

    let algo_grid =
        Array2::from_shape_vec((num_layers, num_qubits), fidelity.algorithmic_fidelity_grid)
            .expect("Shape mismatch for algorithmic_fidelity_grid");
    dict.set_item("algorithmic_fidelity_grid", algo_grid.into_pyarray_bound(py))?;

    let route_grid =
        Array2::from_shape_vec((num_layers, num_qubits), fidelity.routing_fidelity_grid)
            .expect("Shape mismatch for routing_fidelity_grid");
    dict.set_item("routing_fidelity_grid", route_grid.into_pyarray_bound(py))?;

    let coh_grid =
        Array2::from_shape_vec((num_layers, num_qubits), fidelity.coherence_fidelity_grid)
            .expect("Shape mismatch for coherence_fidelity_grid");
    dict.set_item("coherence_fidelity_grid", coh_grid.into_pyarray_bound(py))?;

    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (
    gs_sparse,
    placements,
    distance_matrix,
    sparse_swaps,
    single_gate_error = 1e-4,
    two_gate_error = 1e-3,
    teleportation_error_per_hop = 1e-2,
    single_gate_time = 20.0,
    two_gate_time = 100.0,
    teleportation_time_per_hop = 1000.0,
    t1 = 100_000.0,
    t2 = 50_000.0,
    single_gate_error_per_qubit = None,
    two_gate_error_per_pair = None,
    t1_per_qubit = None,
    t2_per_qubit = None
))]
#[allow(clippy::too_many_arguments)]
pub fn estimate_hardware_fidelity<'py>(
    py: Python<'py>,
    gs_sparse: &Bound<'py, PyArray2<f64>>,
    placements: &Bound<'py, PyArray2<i32>>,
    distance_matrix: &Bound<'py, PyArray2<i32>>,
    sparse_swaps: &Bound<'py, PyArray2<i32>>,
    single_gate_error: f64,
    two_gate_error: f64,
    teleportation_error_per_hop: f64,
    single_gate_time: f64,
    two_gate_time: f64,
    teleportation_time_per_hop: f64,
    t1: f64,
    t2: f64,
    single_gate_error_per_qubit: Option<&Bound<'py, PyArray1<f64>>>,
    two_gate_error_per_pair: Option<&Bound<'py, PyDict>>,
    t1_per_qubit: Option<&Bound<'py, PyArray1<f64>>>,
    t2_per_qubit: Option<&Bound<'py, PyArray1<f64>>>,
) -> PyResult<Bound<'py, PyDict>> {
    let (num_layers, num_qubits, edge_list) = parse_sparse_tensor(gs_sparse, 0);

    let tensor = InteractionTensor::from_sparse(&edge_list, num_layers, num_qubits);

    let placements_arr = placements.readonly().as_array().to_owned();
    let dist_array = distance_matrix.readonly().as_array().to_owned();
    let swaps_readonly = sparse_swaps.readonly();
    let sparse_swaps_arr = swaps_readonly.as_array();

    let routing = extract_inter_core_communications(&placements_arr, dist_array.view());

    let params = build_params(
        single_gate_error, two_gate_error, teleportation_error_per_hop,
        single_gate_time, two_gate_time, teleportation_time_per_hop,
        t1, t2,
        single_gate_error_per_qubit, two_gate_error_per_pair,
        t1_per_qubit, t2_per_qubit,
    )?;
    let fidelity = estimate_fidelity(&tensor, &routing, &params, Some(sparse_swaps_arr));

    let dict = PyDict::new_bound(py);

    dict.set_item("algorithmic_fidelity", fidelity.algorithmic_fidelity)?;
    dict.set_item("routing_fidelity", fidelity.routing_fidelity)?;
    dict.set_item("coherence_fidelity", fidelity.coherence_fidelity)?;
    dict.set_item("overall_fidelity", fidelity.overall_fidelity)?;
    dict.set_item("total_circuit_time_ns", fidelity.total_circuit_time)?;

    let algo_grid =
        Array2::from_shape_vec((num_layers, num_qubits), fidelity.algorithmic_fidelity_grid)
            .unwrap();
    dict.set_item("algorithmic_fidelity_grid", algo_grid.into_pyarray_bound(py))?;

    let route_grid =
        Array2::from_shape_vec((num_layers, num_qubits), fidelity.routing_fidelity_grid).unwrap();
    dict.set_item("routing_fidelity_grid", route_grid.into_pyarray_bound(py))?;

    let coh_grid =
        Array2::from_shape_vec((num_layers, num_qubits), fidelity.coherence_fidelity_grid).unwrap();
    dict.set_item("coherence_fidelity_grid", coh_grid.into_pyarray_bound(py))?;

    Ok(dict)
}

/// The core native module compiled by Maturin
#[pymodule]
fn rust_core<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(map_and_estimate, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_hardware_fidelity, m)?)?;
    Ok(())
}
