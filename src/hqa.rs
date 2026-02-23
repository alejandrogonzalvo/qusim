use pathfinding::prelude::kuhn_munkres;
use pathfinding::matrix::Matrix;
use std::collections::HashSet;

pub fn lookahead(gs: &[Vec<Vec<f64>>], sigma: f64, inf: f64) -> Vec<Vec<f64>> {
    let num_qubits = gs[0].len();
    let mut w = vec![vec![0.0; num_qubits]; num_qubits];
    
    // W = inf * Gs[0]
    for i in 0..num_qubits {
        for j in 0..num_qubits {
            if gs[0][i][j] > 0.0 {
                w[i][j] = inf;
            }
        }
    }
    
    // L = sum(Gs[l] * 2^(-l/sigma)) for l in 1..Gs.len()
    let mut l_sum = vec![vec![0.0; num_qubits]; num_qubits];
    
    for l in 1..gs.len() {
        let weight = 2.0_f64.powf(-(l as f64) / sigma);
        for i in 0..num_qubits {
            for j in 0..num_qubits {
                if gs[l][i][j] > 0.0 {
                    l_sum[i][j] += gs[l][i][j] * weight;
                }
            }
        }
    }
    
    // Return L + W
    for i in 0..num_qubits {
        for j in 0..num_qubits {
            l_sum[i][j] += w[i][j];
        }
    }
    
    l_sum
}

fn validate_partition(g: &[Vec<f64>], p: &[i32]) -> bool {
    let n = g.len();
    for i in 0..n {
        for j in 0..n {
            if g[i][j] > 0.0 {
                if p[i] != p[j] {
                    return false;
                }
            }
        }
    }
    true
}

pub fn hqa_mapping(
    gs: &[Vec<Vec<f64>>],
    mut ps: Vec<Vec<i32>>,
    num_cores: usize,
    num_qubits: usize,
    core_capacities: &[usize],
    distance_matrix: &[Vec<i32>]
) -> Vec<Vec<i32>> {
    
    for i in 0..gs.len() {
        let mut next_ps = ps[i].clone();
        
        let l_matrix = lookahead(&gs[i..], 1.0, 65536.0); // 2^16
        let g = &gs[i];
        
        let mut free_spaces = core_capacities.to_vec();
        for &assignment in &ps[i] {
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
                if g[q1][j] > 0.0 {
                    is_movable = false;
                    break;
                }
            }
            if is_movable && ps[i][q1] >= 0 {
                movable_qubits[ps[i][q1] as usize].push(q1);
            }
            
            for q2 in 0..q1 {
                if g[q1][q2] > 0.0 { // Qubits involved in a two-qubit gate
                    if ps[i][q1] != ps[i][q2] { // Qubits in different partition
                        if next_ps[q1] != -1 {
                            free_spaces[ps[i][q1] as usize] += 1;
                            next_ps[q1] = -1;
                        }
                        if next_ps[q2] != -1 {
                            free_spaces[ps[i][q2] as usize] += 1;
                            next_ps[q2] = -1;
                        }
                        
                        unplaced_qubits.push([q1, q2]);
                        
                        // Compute core_attraction
                        for qaux in 0..num_qubits {
                            if ps[i][qaux] >= 0 {
                                let core_idx = ps[i][qaux] as usize;
                                core_likelihood[q1][core_idx] += l_matrix[q1][qaux];
                                core_likelihood[q2][core_idx] += l_matrix[q2][qaux];
                            }
                        }
                        
                        // Vector normalization q1
                        let sum_q1: f64 = core_likelihood[q1].iter().sum();
                        if sum_q1 > 0.0 {
                            for val in core_likelihood[q1].iter_mut() {
                                *val /= sum_q1;
                            }
                        }
                        // Vector normalization q2
                        let sum_q2: f64 = core_likelihood[q2].iter().sum();
                        if sum_q2 > 0.0 {
                            for val in core_likelihood[q2].iter_mut() {
                                *val /= sum_q2;
                            }
                        }
                    } else { // Qubits in the same partition
                        well_placed_qubits.insert(q1);
                        well_placed_qubits.insert(q2);
                    }
                }
            }
        }
        
        // Check which cores have odd number of interacting qubits
        let mut troubling_cores = Vec::new();
        for core_idx in 0..num_cores {
            if free_spaces[core_idx] % 2 != 0 {
                troubling_cores.push(core_idx);
            }
        }
        
        // pair qubits from both troubling cores
        for j in (0..troubling_cores.len().saturating_sub(1)).step_by(2) {
            let core_1 = troubling_cores[j];
            let core_2 = troubling_cores[j+1];
            
            let mut interaction = 0.0;
            let mut to_move_q1_opt: Option<usize> = None;
            let mut to_move_q2_opt: Option<usize> = None;
            
            if movable_qubits[core_1].is_empty() {
                movable_qubits[core_1] = (0..num_qubits).filter(|&q| next_ps[q] == core_1 as i32).collect();
            }
            if movable_qubits[core_2].is_empty() {
                movable_qubits[core_2] = (0..num_qubits).filter(|&q| next_ps[q] == core_2 as i32).collect();
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
                    if interaction <= l_matrix[q1][q2] {
                        interaction = l_matrix[q1][q2];
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
                
                // Compute core attraction
                for qaux in 0..num_qubits {
                    if ps[i][qaux] >= 0 {
                        let core_idx = ps[i][qaux] as usize;
                        core_likelihood[to_move_q1][core_idx] += l_matrix[to_move_q1][qaux];
                        core_likelihood[to_move_q2][core_idx] += l_matrix[to_move_q2][qaux];
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
            let mut cost_matrix = vec![vec![0_i64; num_cores]; unplaced_qubits.len()];
            
            for pair_idx in 0..unplaced_qubits.len() {
                let [q1, q2] = unplaced_qubits[pair_idx];
                let core_1 = ps[i][q1] as usize;
                let core_2 = ps[i][q2] as usize;
                
                for core_idx in 0..num_cores {
                    if free_spaces[core_idx] < 2 {
                        cost_matrix[pair_idx][core_idx] = inf_cost;
                    } else if core_idx == core_1 {
                        cost_matrix[pair_idx][core_idx] = (distance_matrix[core_2][core_1] as f64 * scale_factor) as i64;
                    } else if core_idx == core_2 {
                        cost_matrix[pair_idx][core_idx] = (distance_matrix[core_1][core_2] as f64 * scale_factor) as i64;
                    } else {
                        cost_matrix[pair_idx][core_idx] = ((distance_matrix[core_1][core_idx] + distance_matrix[core_2][core_idx]) as f64 * scale_factor) as i64;
                    }
                    
                    let likelihood_deduction = (core_likelihood[q1][core_idx] + core_likelihood[q2][core_idx]) / 2.0;
                    cost_matrix[pair_idx][core_idx] -= (likelihood_deduction * scale_factor) as i64;
                }
            }
            
            // Build a square Matrix for kuhn_munkres from pathfinding
            let max_dim = std::cmp::max(unplaced_qubits.len(), num_cores);
            let mut flat = Vec::with_capacity(max_dim * max_dim);
            for r in 0..max_dim {
                for c in 0..max_dim {
                    if r < cost_matrix.len() && c < num_cores {
                        // Negate costs because kuhn_munkres is a max-weight solver
                        flat.push(-cost_matrix[r][c]);
                    } else if r >= cost_matrix.len() {
                        // Dummy row: cost 0 to match anything that isn't a high priority real pair
                        flat.push(0_i64);
                    } else {
                        // Dummy col: use a very negative cost to avoid selecting these
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
            
            pairs_to_remove.sort_unstable_by(|a: &usize, b: &usize| b.cmp(a)); // Reverse sort
            for idx in pairs_to_remove {
                unplaced_qubits.remove(idx);
            }
        }
        
        if !validate_partition(g, &next_ps) {
            // Diagnostic: Heuristic failed to co-locate all gates in this slice
        }
        
        // Push next partition to results
        if ps.len() > i + 1 {
             ps[i+1] = next_ps;
        } else {
             ps.push(next_ps);
        }
    }
    
    ps
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::fs;
    use std::path::Path;

    #[test]
    fn validate_hqa_against_python_all_to_all() {
        let path = Path::new("dse_pau/test_vectors.json");
        let data = fs::read_to_string(path).expect("Unable to read test_vectors.json");
        let parsed: Value = serde_json::from_str(&data).expect("Unable to parse test_vectors.json");
        
        let test_case = &parsed["hqa_test_qft_25_all_to_all"];
        
        let num_virtual_qubits = test_case["num_virtual_qubits"].as_u64().unwrap() as usize;
        let num_cores = test_case["num_cores"].as_u64().unwrap() as usize;

        let input_circuit: Vec<Vec<[f64; 3]>> = serde_json::from_value(test_case["input_circuit"].clone()).unwrap();
        // Reconstruct Gs sparse representation to dense
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
        
        // Run Rust implementation
        let rust_output = hqa_mapping(
            &gs,
            ps.clone(),
            num_cores,
            num_virtual_qubits,
            &core_capacities,
            &distance_matrix
        );
        
        // Deserialize expected Python output
        let expected_output: Vec<Vec<i32>> = serde_json::from_value(test_case["expected_output"].clone()).unwrap();
        
        // The ultimate validation
        assert_eq!(rust_output, expected_output, "Rust port diverges from Python on All-to-All test!");
    }

    #[test]
    fn validate_hqa_against_python_ring() {
        let path = Path::new("dse_pau/test_vectors.json");
        let data = fs::read_to_string(path).expect("Unable to read test_vectors.json");
        let parsed: Value = serde_json::from_str(&data).expect("Unable to parse test_vectors.json");
        
        let test_case = &parsed["hqa_test_qft_25_ring"];
        
        let num_virtual_qubits = test_case["num_virtual_qubits"].as_u64().unwrap() as usize;
        let num_cores = test_case["num_cores"].as_u64().unwrap() as usize;

        let input_circuit: Vec<Vec<[f64; 3]>> = serde_json::from_value(test_case["input_circuit"].clone()).unwrap();
        // Reconstruct Gs sparse representation to dense
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
        
        // Run Rust implementation
        let rust_output = hqa_mapping(
            &gs,
            ps.clone(),
            num_cores,
            num_virtual_qubits,
            &core_capacities,
            &distance_matrix
        );
        
        // Deserialize expected Python output
        let expected_output: Vec<Vec<i32>> = serde_json::from_value(test_case["expected_output"].clone()).unwrap();
        
        // The ultimate validation
        assert_eq!(rust_output, expected_output, "Rust port diverges from Python on Ring test!");
    }

    #[test]
    fn validate_hqa_against_python_large_cores() {
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
        
        // Run Rust implementation
        let rust_output = hqa_mapping(
            &gs,
            ps.clone(),
            num_cores,
            num_virtual_qubits,
            &core_capacities,
            &distance_matrix
        );
        
        // Deserialize expected Python output
        let expected_output: Vec<Vec<i32>> = serde_json::from_value(test_case["expected_output"].clone()).unwrap();
        
        // The ultimate validation
        assert_eq!(rust_output, expected_output, "Rust port diverges from Python on Large Cores test!");
    }
}
