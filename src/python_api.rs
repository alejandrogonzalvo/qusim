use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::circuit::InteractionTensor;
use crate::hqa::hqa_mapping;
use crate::noise::{estimate_fidelity, ArchitectureParams};
use crate::routing::extract_inter_core_communications;

/// Python dataclass equivalent for the fidelity report.
/// Returned as a dict to keep PyO3 boundary simple.
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
    t2 = 50_000.0
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
) -> PyResult<Bound<'py, PyDict>> {
    // Convert PyArray to Rust ArrayView (zero-copy)
    let gs_sparse_rust = gs_sparse.readonly().as_array().to_owned();

    // Find num_layers and num_qubits from the sparse tensor array
    // gs_sparse_rust is shape (E, 4) where cols are [layer, q1, q2, weight]
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

    // Safety check in case the sparse tensor is empty but we know sizes from partition
    let part_len = initial_partition.len().unwrap_or(0);
    if part_len > num_qubits {
        num_qubits = part_len;
    }

    // Convert edge list to &[[f64; 4]] for InteractionTensor directly
    let mut edge_list: Vec<[f64; 4]> = Vec::with_capacity(gs_sparse_rust.shape()[0]);
    for row in gs_sparse_rust.rows() {
        edge_list.push([row[0], row[1], row[2], row[3]]);
    }

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
    let params = ArchitectureParams {
        single_gate_error,
        two_gate_error,
        teleportation_error_per_hop,
        single_gate_time,
        two_gate_time,
        teleportation_time_per_hop,
        t1,
        t2,
    };
    let fidelity = estimate_fidelity(&tensor, &routing, &params, None);

    // Build the output Python dict
    let dict = PyDict::new_bound(py);

    dict.set_item("execution_success", true)?;

    // Placements matrix into NumPy array
    dict.set_item("placements", result.into_pyarray_bound(py))?;

    // Routing simple stats
    dict.set_item("total_teleportations", routing.total_teleportations)?;
    dict.set_item("total_epr_pairs", routing.total_epr_pairs)?;
    dict.set_item("total_network_distance", routing.total_network_distance)?;

    // Vector of teleportations per timeslice
    let tps_py = PyList::new_bound(py, routing.communications_per_timeslice);
    dict.set_item("teleportations_per_slice", tps_py)?;

    // Top level fidelity metrics
    dict.set_item("algorithmic_fidelity", fidelity.algorithmic_fidelity)?;
    dict.set_item("routing_fidelity", fidelity.routing_fidelity)?;
    dict.set_item("coherence_fidelity", fidelity.coherence_fidelity)?;
    dict.set_item("overall_fidelity", fidelity.overall_fidelity)?;
    dict.set_item("total_circuit_time_ns", fidelity.total_circuit_time)?;

    // Gridded data: convert flat Vec<f64> into 2D Array2<f64> then into PyArray
    let algo_grid =
        Array2::from_shape_vec((num_layers, num_qubits), fidelity.algorithmic_fidelity_grid)
            .expect("Shape mismatch for algorithmic_fidelity_grid");
    dict.set_item(
        "algorithmic_fidelity_grid",
        algo_grid.into_pyarray_bound(py),
    )?;

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
    intra_core_swaps_grid,
    single_gate_error = 1e-4,
    two_gate_error = 1e-3,
    teleportation_error_per_hop = 1e-2,
    single_gate_time = 20.0,
    two_gate_time = 100.0,
    teleportation_time_per_hop = 1000.0,
    t1 = 100_000.0,
    t2 = 50_000.0
))]
#[allow(clippy::too_many_arguments)]
pub fn estimate_hardware_fidelity<'py>(
    py: Python<'py>,
    gs_sparse: &Bound<'py, PyArray2<f64>>,
    placements: &Bound<'py, PyArray2<i32>>,
    distance_matrix: &Bound<'py, PyArray2<i32>>,
    intra_core_swaps_grid: &Bound<'py, PyArray2<f64>>,
    single_gate_error: f64,
    two_gate_error: f64,
    teleportation_error_per_hop: f64,
    single_gate_time: f64,
    two_gate_time: f64,
    teleportation_time_per_hop: f64,
    t1: f64,
    t2: f64,
) -> PyResult<Bound<'py, PyDict>> {
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

    let mut edge_list: Vec<[f64; 4]> = Vec::with_capacity(gs_sparse_rust.shape()[0]);
    for row in gs_sparse_rust.rows() {
        edge_list.push([row[0], row[1], row[2], row[3]]);
    }

    let tensor = InteractionTensor::from_sparse(&edge_list, num_layers, num_qubits);

    let placements_arr = placements.readonly().as_array().to_owned();
    let dist_array = distance_matrix.readonly().as_array().to_owned();
    let swaps_readonly = intra_core_swaps_grid.readonly();
    let swaps_arr = swaps_readonly.as_array();

    let routing = extract_inter_core_communications(&placements_arr, dist_array.view());

    let params = ArchitectureParams {
        single_gate_error,
        two_gate_error,
        teleportation_error_per_hop,
        single_gate_time,
        two_gate_time,
        teleportation_time_per_hop,
        t1,
        t2,
    };
    let fidelity = estimate_fidelity(&tensor, &routing, &params, Some(swaps_arr));

    let dict = PyDict::new_bound(py);

    dict.set_item("algorithmic_fidelity", fidelity.algorithmic_fidelity)?;
    dict.set_item("routing_fidelity", fidelity.routing_fidelity)?;
    dict.set_item("coherence_fidelity", fidelity.coherence_fidelity)?;
    dict.set_item("overall_fidelity", fidelity.overall_fidelity)?;
    dict.set_item("total_circuit_time_ns", fidelity.total_circuit_time)?;

    let algo_grid =
        Array2::from_shape_vec((num_layers, num_qubits), fidelity.algorithmic_fidelity_grid)
            .unwrap();
    dict.set_item(
        "algorithmic_fidelity_grid",
        algo_grid.into_pyarray_bound(py),
    )?;

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
