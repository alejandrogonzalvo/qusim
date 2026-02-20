import networkx as nx
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

def lookahead(Gs, func='exp', sigma=1, inf=2**16):
    W = inf * Gs[0]
    W.fill_value = np.float64(0.0)
    if func == 'exp':
        pass
    else:
        raise NotImplementedError('For the time being only exp is implemented')
    
    L = np.arange(1, Gs.shape[0])[:,np.newaxis, np.newaxis]
    L = Gs[1:] * 2**(-L/sigma)
    L.fill_value = np.single(0.0)
    return np.sum(L, axis=0) + W

def validate_partition(G, P, inf=2**16):
    if isinstance(P, list):
        P = np.array(P)

    if isinstance(G, nx.Graph):
        for (u, v) in G.edges:
            if P[u] != P[v]:
                # print(f'Nodes {u},{v} are not in the same partition')
                return False
    else:
        edges = np.where(G != 0)
        if any(P[edges[0]] != P[edges[1]]):
            # print(P[edges[0]], P[edges[1]])
            return False
    
    return True

def HQA(Gs, Ps, N, qubits, physical_qubits, distance_matrix=None):
    if distance_matrix == None:
        distance_matrix = [[1 if j != i else 0 for j in range(N)] for i in range(N)]

    for i in tqdm(range(len(Gs))):
        Ps[i+1] = Ps[i]
        L = lookahead(Gs[i:])
        G = Gs[i].todense()

        # Count how many virtual qubits per core
        free_spaces = [physical_qubits/N for _ in range(N)]
        for assignment in Ps[i]:
            free_spaces[assignment] -= 1
        
        well_placed_qubits = set()
        unplaced_qubits = []
        movable_qubits = [[] for _ in range(N)]

        core_likelihood = [[0 for _ in range(N)] for _ in range(qubits)]

        for q1 in range(qubits):
            if sum(G[q1]) == 0:
                movable_qubits[Ps[i][q1]].append(q1)

            for q2 in range(0, q1):
                if G[q1][q2] == 1: # Qubits involved in a two-qubit gate
                    if Ps[i][q1] != Ps[i][q2]: # Qubits in diferent partition
                        unplaced_qubits.append([q1, q2])
                        free_spaces[Ps[i][q1]] += 1
                        free_spaces[Ps[i][q2]] += 1

                        Ps[i+1][q1] = -1
                        Ps[i+1][q2] = -1

                        # Computer core_attraction for each unplaced qubit
                        for qaux in range(qubits):
                            core_likelihood[q1][Ps[i][qaux]] += L[q1][qaux]
                            core_likelihood[q2][Ps[i][qaux]] += L[q2][qaux]

                        # Vector normalization
                        core_likelihood[q1] = [val/sum(core_likelihood[q1]) if sum(core_likelihood[q1])!=0 else 0 for val in core_likelihood[q1]]
                        core_likelihood[q2] = [val/sum(core_likelihood[q2]) if sum(core_likelihood[q2])!=0 else 0 for val in core_likelihood[q2]]

                    else: # Qubits in the same partition
                        well_placed_qubits.add(q1)
                        well_placed_qubits.add(q2)

        # unplaced_qubits, well_placed_qubits, movable_qubits

        # Check which cores have odd number of interacting qubits (trouble)
        troubling_cores = []
        for core_idx in range(N):
            if free_spaces[core_idx] %2 != 0:
                troubling_cores.append(core_idx)
        
        # need to pair qubits from both troubling cores
        for j in range(0, len(troubling_cores)-1, 2):
            core_1, core_2 = troubling_cores[j], troubling_cores[j+1]

            interaction = 0
            to_move_q1 = None
            to_move_q2 = None

            if len(movable_qubits[core_1]) == 0 and len(movable_qubits[core_2]) != 0:
                Ps[i+1][movable_qubits[core_2][-1]] = core_1

                movable_qubits[core_1].append(movable_qubits[core_2][-1])
                movable_qubits[core_2].pop()

                free_spaces[core_1] -= 1
                free_spaces[core_2] += 1

                continue
            
            if len(movable_qubits[core_1]) != 0 and len(movable_qubits[core_2]) == 0:
                Ps[i+1][movable_qubits[core_1][-1]] = core_2

                movable_qubits[core_2].append(movable_qubits[core_1][-1])
                movable_qubits[core_1].pop()

                free_spaces[core_1] += 1
                free_spaces[core_2] -= 1

                continue

            for q1 in movable_qubits[core_1]:
                for q2 in movable_qubits[core_2]:
                    if interaction <= L[q1][q2]:
                        interaction = L[q1][q2]
                        to_move_q1 = q1
                        to_move_q2 = q2
            
            free_spaces[core_1] += 1
            free_spaces[core_2] += 1

            Ps[i+1][to_move_q1] = -1
            Ps[i+1][to_move_q2] = -1

            unplaced_qubits.append([to_move_q1, to_move_q2])

            # Computer core_attraction for each unplaced qubit
            for qaux in range(qubits):
                core_likelihood[to_move_q1][Ps[i][qaux]] += L[to_move_q1][qaux]
                core_likelihood[to_move_q2][Ps[i][qaux]] += L[to_move_q2][qaux]
            
            # Vector normalization
            core_likelihood[to_move_q1] = [val/sum(core_likelihood[to_move_q1]) if sum(core_likelihood[to_move_q1])!=0 else 0 for val in core_likelihood[to_move_q1]]
            core_likelihood[to_move_q2] = [val/sum(core_likelihood[to_move_q2]) if sum(core_likelihood[to_move_q2])!=0 else 0 for val in core_likelihood[to_move_q2]]

        # Assignation of qubits to cores
        while len(unplaced_qubits) > 0:
            cost_matrix = [[0 for _ in range(N)] for _ in range(len(unplaced_qubits))]

            for pair_idx in range(len(unplaced_qubits)):
                q1, q2 = unplaced_qubits[pair_idx]
                core_1, core_2 = Ps[i][q1], Ps[i][q2]

                for core_idx in range(N):
                    if free_spaces[core_idx] < 2:
                        cost_matrix[pair_idx][core_idx] = 10000
                    elif core_idx == core_1: # moving q2 to core_1==core_idx
                        cost_matrix[pair_idx][core_idx] = distance_matrix[core_2][core_1]
                    elif core_idx == core_2: # moving q1 to core_2==core_idx
                        cost_matrix[pair_idx][core_idx] = distance_matrix[core_1][core_2]
                    else: # moving q1 and q2 to core_idx
                        cost_matrix[pair_idx][core_idx] = distance_matrix[core_1][core_idx] + distance_matrix[core_2][core_idx]
                    
                    # if lookahead_weights:
                    cost_matrix[pair_idx][core_idx] -= (core_likelihood[q1][core_idx] + core_likelihood[q2][core_idx])/2
            
            row, col = linear_sum_assignment(cost_matrix)

            pairs_to_remove = []
            for idx in range(len(row)):
                if free_spaces[col[idx]] < 2:
                    continue

                qubit_1, qubit_2 = unplaced_qubits[row[idx]]
                Ps[i+1][qubit_1] = col[idx]
                Ps[i+1][qubit_2] = col[idx]
                free_spaces[col[idx]] -= 2

                well_placed_qubits.add(qubit_1)
                well_placed_qubits.add(qubit_2)

                pairs_to_remove.append(row[idx])
            
            for idx in pairs_to_remove[::-1]:
                del(unplaced_qubits[idx])
                
        if not (validate_partition(Gs[i], Ps[i+1])):
            print('Error!')
        # else:
            # print('OK!')

    return Ps

def HQA_variation(Gs, Ps, N, qubits, qubits_per_core, distance_matrix=None):
    if distance_matrix == None:
        distance_matrix = [[1 if j != i else 0 for j in range(N)] for i in range(N)]

    for i in tqdm(range(len(Gs))):
    # for i in range(len(Gs)):
        Ps[i+1] = Ps[i]
        L = lookahead(Gs[i:])
        G = Gs[i].todense()

        # Count how many virtual qubits per core
        free_spaces = qubits_per_core.copy() # in this version of HQA, qubits_per_core is a list with how many physical qubits each core has
        # print(1, free_spaces)
        for assignment in Ps[i]:
            free_spaces[assignment] -= 1

        # print(2, free_spaces)

        well_placed_qubits = set()
        unplaced_qubits = []
        movable_qubits = [[] for _ in range(N)]

        core_likelihood = [[0 for _ in range(N)] for _ in range(qubits)]

        for q1 in range(qubits):
            if sum(G[q1]) == 0:
                movable_qubits[Ps[i][q1]].append(q1)

            for q2 in range(0, q1):
                if G[q1][q2] == 1: # Qubits involved in a two-qubit gate
                    if Ps[i][q1] != Ps[i][q2]: # Qubits in diferent partition
                        unplaced_qubits.append([q1, q2])
                        free_spaces[Ps[i][q1]] += 1
                        free_spaces[Ps[i][q2]] += 1

                        Ps[i+1][q1] = -1
                        Ps[i+1][q2] = -1

                        # Computer core_attraction for each unplaced qubit
                        for qaux in range(qubits):
                            core_likelihood[q1][Ps[i][qaux]] += L[q1][qaux]
                            core_likelihood[q2][Ps[i][qaux]] += L[q2][qaux]

                        # Vector normalization
                        core_likelihood[q1] = [val/sum(core_likelihood[q1]) if sum(core_likelihood[q1])!=0 else 0 for val in core_likelihood[q1]]
                        core_likelihood[q2] = [val/sum(core_likelihood[q2]) if sum(core_likelihood[q2])!=0 else 0 for val in core_likelihood[q2]]

                    else: # Qubits in the same partition
                        well_placed_qubits.add(q1)
                        well_placed_qubits.add(q2)
        # print(3, free_spaces, unplaced_qubits)
        # unplaced_qubits, well_placed_qubits, movable_qubits

        # Check which cores have odd number of interacting qubits (trouble)
        troubling_cores = []
        for core_idx in range(N):
            if free_spaces[core_idx] %2 != 0:
                troubling_cores.append(core_idx)
        
        # print(4, free_spaces, troubling_cores)
        # need to pair qubits from both troubling cores
        for j in range(0, len(troubling_cores)-1, 2):
            core_1, core_2 = troubling_cores[j], troubling_cores[j+1]

            interaction = 0
            to_move_q1 = None
            to_move_q2 = None

            if len(movable_qubits[core_1]) == 0 and len(movable_qubits[core_2]) != 0:
                Ps[i+1][movable_qubits[core_2][-1]] = core_1

                movable_qubits[core_1].append(movable_qubits[core_2][-1])
                movable_qubits[core_2].pop()

                free_spaces[core_1] -= 1
                free_spaces[core_2] += 1

                continue
            
            if len(movable_qubits[core_1]) != 0 and len(movable_qubits[core_2]) == 0:
                Ps[i+1][movable_qubits[core_1][-1]] = core_2

                movable_qubits[core_2].append(movable_qubits[core_1][-1])
                movable_qubits[core_1].pop()

                free_spaces[core_1] += 1
                free_spaces[core_2] -= 1

                continue

            for q1 in movable_qubits[core_1]:
                for q2 in movable_qubits[core_2]:
                    if interaction <= L[q1][q2]:
                        interaction = L[q1][q2]
                        to_move_q1 = q1
                        to_move_q2 = q2
            
            free_spaces[core_1] += 1
            free_spaces[core_2] += 1

            Ps[i+1][to_move_q1] = -1
            Ps[i+1][to_move_q2] = -1

            unplaced_qubits.append([to_move_q1, to_move_q2])

            # Computer core_attraction for each unplaced qubit
            for qaux in range(qubits):
                core_likelihood[to_move_q1][Ps[i][qaux]] += L[to_move_q1][qaux]
                core_likelihood[to_move_q2][Ps[i][qaux]] += L[to_move_q2][qaux]
            
            # Vector normalization
            core_likelihood[to_move_q1] = [val/sum(core_likelihood[to_move_q1]) if sum(core_likelihood[to_move_q1])!=0 else 0 for val in core_likelihood[to_move_q1]]
            core_likelihood[to_move_q2] = [val/sum(core_likelihood[to_move_q2]) if sum(core_likelihood[to_move_q2])!=0 else 0 for val in core_likelihood[to_move_q2]]

        # print(5, free_spaces)

        # Assignation of qubits to cores
        while len(unplaced_qubits) > 0:
            cost_matrix = [[0 for _ in range(N)] for _ in range(len(unplaced_qubits))]

            for pair_idx in range(len(unplaced_qubits)):
                q1, q2 = unplaced_qubits[pair_idx]
                core_1, core_2 = Ps[i][q1], Ps[i][q2]

                for core_idx in range(N):
                    if free_spaces[core_idx] < 2:
                        cost_matrix[pair_idx][core_idx] = 10000
                    elif core_idx == core_1: # moving q2 to core_1==core_idx
                        cost_matrix[pair_idx][core_idx] = distance_matrix[core_2][core_1]
                    elif core_idx == core_2: # moving q1 to core_2==core_idx
                        cost_matrix[pair_idx][core_idx] = distance_matrix[core_1][core_2]
                    else: # moving q1 and q2 to core_idx
                        cost_matrix[pair_idx][core_idx] = distance_matrix[core_1][core_idx] + distance_matrix[core_2][core_idx]
                    
                    # if lookahead_weights:
                    cost_matrix[pair_idx][core_idx] -= (core_likelihood[q1][core_idx] + core_likelihood[q2][core_idx])/2
            
            # print(6, free_spaces)
            # print(cost_matrix)
            # print()
            # input()
            
            row, col = linear_sum_assignment(cost_matrix)

            pairs_to_remove = []
            for idx in range(len(row)):
                if free_spaces[col[idx]] < 2:
                    continue

                qubit_1, qubit_2 = unplaced_qubits[row[idx]]
                Ps[i+1][qubit_1] = col[idx]
                Ps[i+1][qubit_2] = col[idx]
                free_spaces[col[idx]] -= 2

                well_placed_qubits.add(qubit_1)
                well_placed_qubits.add(qubit_2)

                pairs_to_remove.append(row[idx])
            
            for idx in pairs_to_remove[::-1]:
                del(unplaced_qubits[idx])
                
        if not (validate_partition(Gs[i], Ps[i+1])):
            print('Error!')
        # else:
            # print('OK!')

    return Ps