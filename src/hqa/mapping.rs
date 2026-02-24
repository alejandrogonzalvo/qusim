use ndarray::{Array2, ArrayView2};
use pathfinding::matrix::Matrix;
use pathfinding::prelude::kuhn_munkres;
use std::collections::HashSet;

use crate::circuit::{ActiveGates, InteractionTensor};

/// Maximum number of future slices the lookahead considers.
const LOOKAHEAD_HORIZON: usize = 20;

/// Computes the lookahead matrix using sparse edge traversal and a truncated horizon.
fn lookahead(
    active_gates: &ActiveGates,
    current_layer: usize,
    num_qubits: usize,
    sigma: f64,
    inf: f64,
) -> Array2<f64> {
    let mut result = Array2::<f64>::zeros((num_qubits, num_qubits));
    let num_layers = active_gates.len();

    if current_layer >= num_layers {
        return result;
    }

    // W: current layer — only iterate over actual gates
    for &(i, j, _) in &active_gates[current_layer] {
        result[[i, j]] = inf;
    }

    // L: future layers with exponential decay, only over actual gates
    let end_layer = (current_layer + 1 + LOOKAHEAD_HORIZON).min(num_layers);
    for l in (current_layer + 1)..end_layer {
        let distance = l - current_layer;
        let decay = 2.0_f64.powf(-(distance as f64) / sigma);
        for &(i, j, weight) in &active_gates[l] {
            result[[i, j]] += weight * decay;
        }
    }

    result
}

/// Validates that all interacting qubit pairs are co-located on the same core.
fn validate_partition(layer_gates: &[(usize, usize, f64)], p: &[i32]) -> bool {
    layer_gates
        .iter()
        .all(|&(i, j, w)| w <= 0.0 || p[i] == p[j])
}

/// Collects the set of qubits that participate in at least one gate this layer.
#[inline]
fn collect_active_qubits(layer_gates: &[(usize, usize, f64)]) -> HashSet<usize> {
    let mut active = HashSet::new();
    for &(u, v, _) in layer_gates {
        active.insert(u);
        active.insert(v);
    }
    active
}

/// Deduplicates and sorts gate pairs with positive interaction weight.
#[inline]
fn collect_edge_pairs(layer_gates: &[(usize, usize, f64)]) -> Vec<(usize, usize)> {
    let mut edge_pairs = Vec::new();
    let mut seen = HashSet::new();
    for &(u, v, w) in layer_gates {
        if w <= 0.0 {
            continue;
        }
        let (hi, lo) = if u > v { (u, v) } else { (v, u) };
        if seen.insert((hi, lo)) {
            edge_pairs.push((hi, lo));
        }
    }
    edge_pairs.sort_unstable();
    edge_pairs
}

/// Normalizes a distribution vector to sum to 1.0, leaving it unchanged if the sum is zero.
#[inline]
fn normalize_distribution(distribution: &mut [f64]) {
    let sum: f64 = distribution.iter().sum();
    if sum > 0.0 {
        for val in distribution.iter_mut() {
            *val /= sum;
        }
    }
}

/// Identifies cores with an odd number of free slots.
#[inline]
fn find_troubling_cores(free_spaces: &[usize]) -> Vec<usize> {
    free_spaces
        .iter()
        .enumerate()
        .filter(|(_, &s)| s % 2 != 0)
        .map(|(i, _)| i)
        .collect()
}

/// Mutable state for processing a single timeslice of the HQA algorithm.
struct TimesliceState {
    placement_row: Vec<i32>,
    num_qubits: usize,
    num_cores: usize,
    next_placement: Vec<i32>,
    free_spaces: Vec<usize>,
    well_placed_qubits: HashSet<usize>,
    unplaced_qubits: Vec<[usize; 2]>,
    movable_qubits: Vec<Vec<usize>>,
    core_likelihood: Vec<Vec<f64>>,
    lookahead_matrix: Array2<f64>,
}

impl TimesliceState {
    /// Initializes state from current placements, computing free spaces and movable qubits.
    #[inline]
    fn new(
        placements: &Array2<i32>,
        timeslice: usize,
        num_qubits: usize,
        num_cores: usize,
        core_capacities: &[usize],
        layer_gates: &[(usize, usize, f64)],
        lookahead_matrix: Array2<f64>,
    ) -> Self {
        let placement_row = placements.row(timeslice).to_vec();
        let next_placement = placement_row.clone();

        let mut free_spaces = core_capacities.to_vec();
        for &assignment in &placement_row {
            if assignment >= 0 {
                free_spaces[assignment as usize] -= 1;
            }
        }

        let active_qubits = collect_active_qubits(layer_gates);

        let mut movable_qubits = vec![Vec::new(); num_cores];
        for (q, &assignment) in placement_row.iter().enumerate() {
            if !active_qubits.contains(&q) && assignment >= 0 {
                movable_qubits[assignment as usize].push(q);
            }
        }

        Self {
            placement_row,
            num_qubits,
            num_cores,
            next_placement,
            free_spaces,
            well_placed_qubits: HashSet::new(),
            unplaced_qubits: Vec::new(),
            movable_qubits,
            core_likelihood: vec![vec![0.0; num_cores]; num_qubits],
            lookahead_matrix,
        }
    }

    /// Removes a qubit from its current core assignment, freeing the slot.
    #[inline]
    fn evict_qubit(&mut self, qubit: usize) {
        if self.next_placement[qubit] != -1 {
            self.free_spaces[self.placement_row[qubit] as usize] += 1;
            self.next_placement[qubit] = -1;
        }
    }

    /// Accumulates future interaction weight toward each core for a qubit.
    #[inline]
    fn accumulate_likelihood(&mut self, qubit: usize) {
        for qaux in 0..self.num_qubits {
            if self.placement_row[qaux] >= 0 {
                let core_idx = self.placement_row[qaux] as usize;
                self.core_likelihood[qubit][core_idx] += self.lookahead_matrix[[qubit, qaux]];
            }
        }
    }

    /// Classifies each gate pair as co-located or conflicting, evicting and enqueuing conflicts.
    #[inline]
    fn detect_conflicts(&mut self, edge_pairs: &[(usize, usize)]) {
        for &(q1, q2) in edge_pairs {
            if self.placement_row[q1] == self.placement_row[q2] {
                self.well_placed_qubits.insert(q1);
                self.well_placed_qubits.insert(q2);
                continue;
            }

            self.evict_qubit(q1);
            self.evict_qubit(q2);
            self.unplaced_qubits.push([q1, q2]);

            self.accumulate_likelihood(q1);
            self.accumulate_likelihood(q2);

            normalize_distribution(&mut self.core_likelihood[q1]);
            normalize_distribution(&mut self.core_likelihood[q2]);
        }
    }

    /// Balances cores with odd free-slot counts by swapping idle qubits between paired cores.
    #[inline]
    fn balance_odd_capacity_cores(&mut self) {
        let troubling_cores = find_troubling_cores(&self.free_spaces);

        for j in (0..troubling_cores.len().saturating_sub(1)).step_by(2) {
            let core_1 = troubling_cores[j];
            let core_2 = troubling_cores[j + 1];

            if self.movable_qubits[core_1].is_empty() {
                self.movable_qubits[core_1] = (0..self.num_qubits)
                    .filter(|&q| self.next_placement[q] == core_1 as i32)
                    .collect();
            }
            if self.movable_qubits[core_2].is_empty() {
                self.movable_qubits[core_2] = (0..self.num_qubits)
                    .filter(|&q| self.next_placement[q] == core_2 as i32)
                    .collect();
            }

            if self.movable_qubits[core_1].is_empty() && !self.movable_qubits[core_2].is_empty() {
                let q_to_move = self.movable_qubits[core_2].pop().unwrap();
                self.next_placement[q_to_move] = core_1 as i32;
                self.movable_qubits[core_1].push(q_to_move);
                self.free_spaces[core_1] -= 1;
                self.free_spaces[core_2] += 1;
                continue;
            }

            if !self.movable_qubits[core_1].is_empty() && self.movable_qubits[core_2].is_empty() {
                let q_to_move = self.movable_qubits[core_1].pop().unwrap();
                self.next_placement[q_to_move] = core_2 as i32;
                self.movable_qubits[core_2].push(q_to_move);
                self.free_spaces[core_1] += 1;
                self.free_spaces[core_2] -= 1;
                continue;
            }

            let mut interaction = 0.0;
            let mut to_move_q1_opt: Option<usize> = None;
            let mut to_move_q2_opt: Option<usize> = None;

            for &q1 in &self.movable_qubits[core_1] {
                for &q2 in &self.movable_qubits[core_2] {
                    if interaction <= self.lookahead_matrix[[q1, q2]] {
                        interaction = self.lookahead_matrix[[q1, q2]];
                        to_move_q1_opt = Some(q1);
                        to_move_q2_opt = Some(q2);
                    }
                }
            }

            if let (Some(to_move_q1), Some(to_move_q2)) = (to_move_q1_opt, to_move_q2_opt) {
                self.evict_qubit(to_move_q1);
                self.evict_qubit(to_move_q2);

                self.unplaced_qubits.push([to_move_q1, to_move_q2]);

                self.accumulate_likelihood(to_move_q1);
                self.accumulate_likelihood(to_move_q2);

                normalize_distribution(&mut self.core_likelihood[to_move_q1]);
                normalize_distribution(&mut self.core_likelihood[to_move_q2]);
            }
        }
    }

    /// Assigns unplaced qubit pairs to cores using the Hungarian algorithm.
    #[inline]
    fn assign_pairs_to_cores(&mut self, distance_matrix: ArrayView2<'_, i32>) {
        let scale_factor = 10_000_000.0;
        let inf_cost = 100_000_000_000_i64;

        while !self.unplaced_qubits.is_empty() {
            let max_dim = std::cmp::max(self.unplaced_qubits.len(), self.num_cores);
            let mut flat = Vec::with_capacity(max_dim * max_dim);

            for r in 0..max_dim {
                for c in 0..max_dim {
                    if r < self.unplaced_qubits.len() && c < self.num_cores {
                        let [q1, q2] = self.unplaced_qubits[r];
                        let core_1 = self.placement_row[q1] as usize;
                        let core_2 = self.placement_row[q2] as usize;

                        let cost = if self.free_spaces[c] < 2 {
                            inf_cost
                        } else if c == core_1 {
                            (distance_matrix[[core_2, core_1]] as f64 * scale_factor) as i64
                        } else if c == core_2 {
                            (distance_matrix[[core_1, core_2]] as f64 * scale_factor) as i64
                        } else {
                            ((distance_matrix[[core_1, c]] + distance_matrix[[core_2, c]]) as f64
                                * scale_factor) as i64
                        };

                        let likelihood_deduction =
                            (self.core_likelihood[q1][c] + self.core_likelihood[q2][c]) / 2.0;
                        let final_cost = cost - (likelihood_deduction * scale_factor) as i64;

                        flat.push(-final_cost);
                    } else if r >= self.unplaced_qubits.len() {
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
                if row_idx >= self.unplaced_qubits.len() || col_idx >= self.num_cores {
                    continue;
                }

                if self.free_spaces[col_idx] < 2 {
                    continue;
                }

                let [qubit_1, qubit_2] = self.unplaced_qubits[row_idx];
                self.next_placement[qubit_1] = col_idx as i32;
                self.next_placement[qubit_2] = col_idx as i32;
                self.free_spaces[col_idx] -= 2;

                self.well_placed_qubits.insert(qubit_1);
                self.well_placed_qubits.insert(qubit_2);

                pairs_to_remove.push(row_idx);
            }

            pairs_to_remove.sort_unstable_by(|a: &usize, b: &usize| b.cmp(a));
            for idx in pairs_to_remove {
                self.unplaced_qubits.remove(idx);
            }
        }
    }

    /// Writes the computed placement into the target row of the placements matrix.
    #[inline]
    fn commit(self, placements: &mut Array2<i32>, target_row: usize) {
        for (q, &val) in self.next_placement.iter().enumerate() {
            placements[[target_row, q]] = val;
        }
    }
}

pub fn hqa_mapping(
    gate_interactions: &InteractionTensor,
    mut placements: Array2<i32>,
    num_cores: usize,
    core_capacities: &[usize],
    distance_matrix: ArrayView2<'_, i32>,
) -> Array2<i32> {
    let num_layers = gate_interactions.num_layers();
    let num_qubits = gate_interactions.num_qubits();
    let active_gates = gate_interactions.active_gates();

    for timeslice in 0..num_layers {
        let layer_gates = gate_interactions.layer_gates(timeslice);
        let lookahead_matrix = lookahead(&active_gates, timeslice, num_qubits, 1.0, 65536.0);
        let edge_pairs = collect_edge_pairs(layer_gates);

        let mut state = TimesliceState::new(
            &placements,
            timeslice,
            num_qubits,
            num_cores,
            core_capacities,
            layer_gates,
            lookahead_matrix,
        );

        state.detect_conflicts(&edge_pairs);
        state.balance_odd_capacity_cores();
        state.assign_pairs_to_cores(distance_matrix);

        if !validate_partition(layer_gates, &state.next_placement) {
            // Diagnostic: Heuristic failed to co-locate all gates in this slice
        }

        state.commit(&mut placements, timeslice + 1);
    }

    placements
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::InteractionTensor;
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

        let tensor = InteractionTensor::from_sparse(&gs_sparse, num_layers, num_virtual_qubits);

        let input_initial_partition: Vec<i32> =
            serde_json::from_value(test_case["input_initial_partition"].clone()).unwrap();

        let mut placements = Array2::<i32>::zeros((num_layers + 1, num_virtual_qubits));
        for (q, &val) in input_initial_partition.iter().enumerate() {
            placements[[0, q]] = val;
        }

        let core_capacities: Vec<usize> =
            serde_json::from_value(test_case["input_core_capacities"].clone()).unwrap();
        let dist_matrix_vecs: Vec<Vec<i32>> =
            serde_json::from_value(test_case["input_distance_matrix"].clone()).unwrap();
        let dist_flat: Vec<i32> = dist_matrix_vecs.into_iter().flatten().collect();
        let dist_array = Array2::from_shape_vec((num_cores, num_cores), dist_flat)
            .expect("Failed to reshape distance_matrix");

        let rust_output = hqa_mapping(
            &tensor,
            placements,
            num_cores,
            &core_capacities,
            dist_array.view(),
        );

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
