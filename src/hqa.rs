use ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use pathfinding::matrix::Matrix;
use pathfinding::prelude::kuhn_munkres;
use std::collections::HashSet;

use crate::interactionTensor::InteractionTensor;

/// Maximum number of future slices the lookahead considers.
/// With exponential decay (sigma=1), contributions beyond ~20 slices are negligible.
const LOOKAHEAD_HORIZON: usize = 20;

/// Computes the lookahead matrix (L + W) from a 3D interaction view.
/// Only examines up to `LOOKAHEAD_HORIZON` layers to avoid wasted computation.
pub fn lookahead(gs_view: ArrayView3<f64>, sigma: f64, inf: f64) -> Array2<f64> {
    let num_layers = gs_view.dim().0.min(LOOKAHEAD_HORIZON);
    let num_qubits = gs_view.dim().1;

    let mut result = Array2::<f64>::zeros((num_qubits, num_qubits));

    if num_layers == 0 {
        return result;
    }

    // W: Layer 0
    for i in 0..num_qubits {
        for j in 0..num_qubits {
            if gs_view[[0, i, j]] > 0.0 {
                result[[i, j]] = inf;
            }
        }
    }

    // L_sum: Layers 1..horizon
    for l in 1..num_layers {
        let decay_weight = 2.0_f64.powf(-(l as f64) / sigma);
        for i in 0..num_qubits {
            for j in 0..num_qubits {
                let interaction = gs_view[[l, i, j]];
                if interaction > 0.0 {
                    result[[i, j]] += interaction * decay_weight;
                }
            }
        }
    }

    result
}

pub fn validate_partition(g: ArrayView2<f64>, p: &[i32]) -> bool {
    g.indexed_iter()
        .all(|((i, j), &weight)| weight <= 0.0 || p[i] == p[j])
}

pub fn hqa_mapping(
    gs: InteractionTensor,
    mut ps: Array2<i32>,
    num_cores: usize,
    core_capacities: &[usize],
    distance_matrix: ArrayView2<'_, i32>,
) -> Array2<i32> {
    let num_layers = gs.num_layers();
    let num_qubits = gs.num_qubits();

    for i in 0..num_layers {
        // Work on a mutable Vec copy; write back into ps at the end
        let mut next_ps: Vec<i32> = ps.row(i).to_vec();

        let future_view = gs.future_view(i);
        let l_matrix = lookahead(future_view, 1.0, 65536.0);
        let g = gs.current_layer(i);

        let mut free_spaces = core_capacities.to_vec();
        for q in 0..num_qubits {
            let assignment = ps[[i, q]];
            if assignment >= 0 {
                free_spaces[assignment as usize] -= 1;
            }
        }

        let mut well_placed_qubits: HashSet<usize> = HashSet::new();
        let mut unplaced_qubits: Vec<[usize; 2]> = Vec::new();
        let mut movable_qubits: Vec<Vec<usize>> = vec![Vec::new(); num_cores];
        let mut core_likelihood = vec![vec![0.0; num_cores]; num_qubits];

        for q1 in 0..num_qubits {
            let mut is_movable = true;
            for j in 0..num_qubits {
                if g[[q1, j]] > 0.0 {
                    is_movable = false;
                    break;
                }
            }
            if is_movable && ps[[i, q1]] >= 0 {
                movable_qubits[ps[[i, q1]] as usize].push(q1);
            }

            for q2 in 0..q1 {
                if g[[q1, q2]] > 0.0 {
                    if ps[[i, q1]] != ps[[i, q2]] {
                        if next_ps[q1] != -1 {
                            free_spaces[ps[[i, q1]] as usize] += 1;
                            next_ps[q1] = -1;
                        }
                        if next_ps[q2] != -1 {
                            free_spaces[ps[[i, q2]] as usize] += 1;
                            next_ps[q2] = -1;
                        }

                        unplaced_qubits.push([q1, q2]);

                        for qaux in 0..num_qubits {
                            if ps[[i, qaux]] >= 0 {
                                let core_idx = ps[[i, qaux]] as usize;
                                core_likelihood[q1][core_idx] += l_matrix[[q1, qaux]];
                                core_likelihood[q2][core_idx] += l_matrix[[q2, qaux]];
                            }
                        }

                        let sum_q1: f64 = core_likelihood[q1].iter().sum();
                        if sum_q1 > 0.0 {
                            for val in core_likelihood[q1].iter_mut() {
                                *val /= sum_q1;
                            }
                        }
                        let sum_q2: f64 = core_likelihood[q2].iter().sum();
                        if sum_q2 > 0.0 {
                            for val in core_likelihood[q2].iter_mut() {
                                *val /= sum_q2;
                            }
                        }
                    } else {
                        well_placed_qubits.insert(q1);
                        well_placed_qubits.insert(q2);
                    }
                }
            }
        }

        let mut troubling_cores = Vec::new();
        for core_idx in 0..num_cores {
            if free_spaces[core_idx] % 2 != 0 {
                troubling_cores.push(core_idx);
            }
        }

        for j in (0..troubling_cores.len().saturating_sub(1)).step_by(2) {
            let core_1 = troubling_cores[j];
            let core_2 = troubling_cores[j + 1];

            let mut interaction = 0.0;
            let mut to_move_q1_opt: Option<usize> = None;
            let mut to_move_q2_opt: Option<usize> = None;

            if movable_qubits[core_1].is_empty() {
                movable_qubits[core_1] = (0..num_qubits)
                    .filter(|&q| next_ps[q] == core_1 as i32)
                    .collect();
            }
            if movable_qubits[core_2].is_empty() {
                movable_qubits[core_2] = (0..num_qubits)
                    .filter(|&q| next_ps[q] == core_2 as i32)
                    .collect();
            }

            if movable_qubits[core_1].is_empty() && !movable_qubits[core_2].is_empty() {
                let q_to_move = movable_qubits[core_2].pop().unwrap();
                next_ps[q_to_move] = core_1 as i32;
                movable_qubits[core_1].push(q_to_move);
                free_spaces[core_1] -= 1;
                free_spaces[core_2] += 1;
                continue;
            }

            if !movable_qubits[core_1].is_empty() && movable_qubits[core_2].is_empty() {
                let q_to_move = movable_qubits[core_1].pop().unwrap();
                next_ps[q_to_move] = core_2 as i32;
                movable_qubits[core_2].push(q_to_move);
                free_spaces[core_1] += 1;
                free_spaces[core_2] -= 1;
                continue;
            }

            for &q1 in &movable_qubits[core_1] {
                for &q2 in &movable_qubits[core_2] {
                    if interaction <= l_matrix[[q1, q2]] {
                        interaction = l_matrix[[q1, q2]];
                        to_move_q1_opt = Some(q1);
                        to_move_q2_opt = Some(q2);
                    }
                }
            }

            if let (Some(to_move_q1), Some(to_move_q2)) = (to_move_q1_opt, to_move_q2_opt) {
                if next_ps[to_move_q1] != -1 {
                    free_spaces[core_1] += 1;
                    next_ps[to_move_q1] = -1;
                }
                if next_ps[to_move_q2] != -1 {
                    free_spaces[core_2] += 1;
                    next_ps[to_move_q2] = -1;
                }

                unplaced_qubits.push([to_move_q1, to_move_q2]);

                for qaux in 0..num_qubits {
                    if ps[[i, qaux]] >= 0 {
                        let core_idx = ps[[i, qaux]] as usize;
                        core_likelihood[to_move_q1][core_idx] += l_matrix[[to_move_q1, qaux]];
                        core_likelihood[to_move_q2][core_idx] += l_matrix[[to_move_q2, qaux]];
                    }
                }

                let sum_q1: f64 = core_likelihood[to_move_q1].iter().sum();
                if sum_q1 > 0.0 {
                    for val in core_likelihood[to_move_q1].iter_mut() {
                        *val /= sum_q1;
                    }
                }

                let sum_q2: f64 = core_likelihood[to_move_q2].iter().sum();
                if sum_q2 > 0.0 {
                    for val in core_likelihood[to_move_q2].iter_mut() {
                        *val /= sum_q2;
                    }
                }
            }
        }

        // Assignation of qubits to cores
        while !unplaced_qubits.is_empty() {
            let scale_factor = 10_000_000.0;
            let inf_cost = 100_000_000_000_i64;
            let max_dim = std::cmp::max(unplaced_qubits.len(), num_cores);

            let mut flat = Vec::with_capacity(max_dim * max_dim);

            for r in 0..max_dim {
                for c in 0..max_dim {
                    if r < unplaced_qubits.len() && c < num_cores {
                        let [q1, q2] = unplaced_qubits[r];
                        let core_1 = ps[[i, q1]] as usize;
                        let core_2 = ps[[i, q2]] as usize;

                        let cost;

                        if free_spaces[c] < 2 {
                            cost = inf_cost;
                        } else if c == core_1 {
                            cost = (distance_matrix[[core_2, core_1]] as f64 * scale_factor) as i64;
                        } else if c == core_2 {
                            cost = (distance_matrix[[core_1, core_2]] as f64 * scale_factor) as i64;
                        } else {
                            cost = ((distance_matrix[[core_1, c]] + distance_matrix[[core_2, c]])
                                as f64
                                * scale_factor) as i64;
                        }

                        let likelihood_deduction =
                            (core_likelihood[q1][c] + core_likelihood[q2][c]) / 2.0;
                        let final_cost = cost - (likelihood_deduction * scale_factor) as i64;

                        flat.push(-final_cost);
                    } else if r >= unplaced_qubits.len() {
                        flat.push(0_i64);
                    } else {
                        flat.push(-inf_cost);
                    }
                }
            }

            let sq_matrix = Matrix::from_vec(max_dim, max_dim, flat).unwrap();
            let (_, assignments): (i64, Vec<usize>) = kuhn_munkres(&sq_matrix);

            let mut pairs_to_remove = Vec::new();
            for (row_idx, col_idx) in assignments.into_iter().enumerate() {
                if row_idx < unplaced_qubits.len() && col_idx < num_cores {
                    if free_spaces[col_idx] >= 2 {
                        let [qubit_1, qubit_2] = unplaced_qubits[row_idx];
                        next_ps[qubit_1] = col_idx as i32;
                        next_ps[qubit_2] = col_idx as i32;
                        free_spaces[col_idx] -= 2;

                        well_placed_qubits.insert(qubit_1);
                        well_placed_qubits.insert(qubit_2);

                        pairs_to_remove.push(row_idx);
                    }
                }
            }

            pairs_to_remove.sort_unstable_by(|a: &usize, b: &usize| b.cmp(a));
            for idx in pairs_to_remove {
                unplaced_qubits.remove(idx);
            }
        }

        if !validate_partition(g, &next_ps) {
            // Diagnostic: Heuristic failed to co-locate all gates in this slice
        }

        // Write next_ps into ps row i+1
        let target_row = i + 1;
        for q in 0..num_qubits {
            ps[[target_row, q]] = next_ps[q];
        }
    }

    ps
}

/// Helper to reconstruct an Array3 from the new flat sparse JSON format:
/// Each entry is [layer_idx, q1, q2, weight]
pub fn array3_from_sparse(
    gs_sparse: &[[f64; 4]],
    num_layers: usize,
    num_qubits: usize,
) -> Array3<f64> {
    let mut gs = Array3::<f64>::zeros((num_layers, num_qubits, num_qubits));
    for edge in gs_sparse {
        let layer = edge[0] as usize;
        let u = edge[1] as usize;
        let v = edge[2] as usize;
        let w = edge[3];
        if layer < num_layers && u < num_qubits && v < num_qubits {
            gs[[layer, u, v]] = w;
        }
    }
    gs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interactionTensor::InteractionTensor;
    use serde_json::Value;
    use std::fs;
    use std::path::Path;

    fn run_test_case(test_case_name: &str) {
        let path = Path::new("dse_pau/test_vectors.json");
        let data = fs::read_to_string(path).expect("Unable to read test_vectors.json");
        let parsed: Value = serde_json::from_str(&data).expect("Unable to parse test_vectors.json");

        let test_case = &parsed[test_case_name];

        let num_virtual_qubits = test_case["num_virtual_qubits"].as_u64().unwrap() as usize;
        let num_cores = test_case["num_cores"].as_u64().unwrap() as usize;
        let num_layers = test_case["num_layers"].as_u64().unwrap() as usize;

        let gs_sparse: Vec<[f64; 4]> =
            serde_json::from_value(test_case["gs_sparse"].clone()).unwrap();
        let gs_array = array3_from_sparse(&gs_sparse, num_layers, num_virtual_qubits);

        let input_initial_partition: Vec<i32> =
            serde_json::from_value(test_case["input_initial_partition"].clone()).unwrap();

        // Build ps as Array2: shape (num_layers + 1, num_qubits), row 0 = initial partition
        let mut ps = Array2::<i32>::zeros((num_layers + 1, num_virtual_qubits));
        for (q, &val) in input_initial_partition.iter().enumerate() {
            ps[[0, q]] = val;
        }

        let core_capacities: Vec<usize> =
            serde_json::from_value(test_case["input_core_capacities"].clone()).unwrap();
        let dist_matrix_vecs: Vec<Vec<i32>> =
            serde_json::from_value(test_case["input_distance_matrix"].clone()).unwrap();
        let dist_flat: Vec<i32> = dist_matrix_vecs.into_iter().flatten().collect();
        let dist_array = Array2::from_shape_vec((num_cores, num_cores), dist_flat)
            .expect("Failed to reshape distance_matrix");

        let interaction_tensor = InteractionTensor::new(gs_array.view());

        let rust_output = hqa_mapping(
            interaction_tensor,
            ps,
            num_cores,
            &core_capacities,
            dist_array.view(),
        );

        // Convert Array2 result to Vec<Vec<i32>> for comparison with expected output
        let rust_output_vecs: Vec<Vec<i32>> = rust_output
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect();

        let expected_output: Vec<Vec<i32>> =
            serde_json::from_value(test_case["expected_output"].clone()).unwrap();

        assert_eq!(
            rust_output_vecs, expected_output,
            "Rust port diverges from Python on {} test!",
            test_case_name
        );
    }

    #[test]
    fn validate_hqa_against_python_all_to_all() {
        run_test_case("hqa_test_qft_25_all_to_all");
    }

    #[test]
    fn validate_hqa_against_python_ring() {
        run_test_case("hqa_test_qft_25_ring");
    }

    #[test]
    fn validate_hqa_against_python_large_cores() {
        run_test_case("hqa_test_qft_25_large_cores");
    }
}
