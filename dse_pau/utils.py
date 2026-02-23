import numpy as np
import qiskit
from qiskit.converters import circuit_to_dag, dag_to_circuit
import sparse
import random
from HQA import HQA_variation
from joblib import Parallel, delayed
from collections import defaultdict

#### TOPOLOGIES
def line_topology(N):
    return [[abs(i-j) for j in range(N)] for i in range(N)]

def ring_topology(N):
    return [[min(abs(i - j), N - abs(i - j)) for j in range(N)] for i in range(N)]

def star_topology(N):
    return [[0 if i==j else (1 if (i == 0 or j == 0) else 2) for j in range(N)] for i in range(N)]

def grid_topology(N):
    return [[abs((i // int(np.sqrt(N))) - (j // int(np.sqrt(N)))) + abs((i % int(np.sqrt(N))) - (j % int(np.sqrt(N)))) for j in range(N)] for i in range(N)]

def all_to_all_topology(N):
    return [[0 if i==j else 1 for j in range(N)] for i in range(N)]

def get_line_cores(comp_qubits, comm_qubits, qubits_per_core):
    buffer_qubits = comm_qubits
    edge_capacity = qubits_per_core - (comm_qubits + buffer_qubits)
    centre_capacity = qubits_per_core - 2*(comm_qubits + buffer_qubits)

    # add buffer qubits for the odd number of qubits per core
    if edge_capacity % 2 == 1:
        edge_capacity -= 1
    if centre_capacity % 2 == 1:
        centre_capacity -= 1

    if edge_capacity <= 0 or centre_capacity <= 0:
        # raise ValueError("Too many communication qubits for the number of qubits per core")
        return -1
    
    if comp_qubits <= qubits_per_core:
        return [qubits_per_core]

    number_of_central_cores = np.ceil((comp_qubits-2*edge_capacity)/centre_capacity)

    return [edge_capacity] + [centre_capacity]*int(number_of_central_cores) + [edge_capacity]

def get_ring_cores(comp_qubits, comm_qubits, qubits_per_core):
    buffer_qubits = comm_qubits
    centre_capacity = qubits_per_core - 2*(comm_qubits+buffer_qubits) # one buffer qubits per link

    # add buffer qubits for the odd number of qubits per core
    if centre_capacity % 2 == 1:
        centre_capacity -= 1

    if centre_capacity <= 0:
        # raise ValueError("Too many communication qubits for the number of qubits per core")
        return -1

    number_of_central_cores = np.ceil(comp_qubits/centre_capacity)

    return [centre_capacity]*int(number_of_central_cores)

def get_star_cores(comp_qubits, comm_qubits, qubits_per_core):
    buffer_qubits = comm_qubits
    edge_capacity = qubits_per_core - (comm_qubits+buffer_qubits) # one buffer qubit per link
    if edge_capacity % 2 == 1:
        edge_capacity -= 1

    edge_cores = 0

    central_capacity = qubits_per_core

    while True:
        if central_capacity + edge_capacity*edge_cores >= comp_qubits:
            return [central_capacity] + [edge_capacity]*edge_cores
        
        edge_cores += 1
        central_capacity -= (comm_qubits+buffer_qubits)

        if central_capacity % 2 == 1:
            central_capacity -= 1
        
        if central_capacity < 0:
            return -1


def get_all_to_all_cores(comp_qubits, comm_qubits, qubits_per_core):
    buffer_qubits = comm_qubits
    cores = 0

    while True:
        cores += 1
        links = cores - 1

        core_capacity = qubits_per_core - links*(comm_qubits+buffer_qubits)
        if core_capacity <= 0:
            # raise ValueError("Too many communication qubits for the number of qubits per core")
            return -1

        if core_capacity*cores >= comp_qubits:
            if core_capacity%2 == 0:
                return [core_capacity]*cores
            elif (core_capacity-1)*cores >= comp_qubits:
                return [core_capacity-1]*cores


def get_grid_cores(comp_qubits, comm_qubits, qubits_per_core):
    buffer_qubits = comm_qubits

    # Start with a single core, and keep adding them till we have enough qubits
    corner_capacity = qubits_per_core - 2*(comm_qubits+buffer_qubits)
    edge_capacity = qubits_per_core - 3*(comm_qubits+buffer_qubits)
    centre_capacity = qubits_per_core - 4*(comm_qubits+buffer_qubits)

    if corner_capacity <= 0 or edge_capacity <= 0 or centre_capacity <= 0:
        # raise ValueError("Too many communication qubits for the number of qubits per core")
        return -1
    
    rows, cols = 1, 1
    while True:
        total_capacity = 4*corner_capacity + 2*(rows-2)*edge_capacity + 2*(cols-2)*edge_capacity + (rows-2)*(cols-2)*centre_capacity
        if total_capacity >= comp_qubits:
            capacities = []

            for i in range(rows):
                if i == 0 or i == rows-1:
                    capacities.append(corner_capacity)
                    for _ in range(cols-2):
                        capacities.append(edge_capacity)
                    capacities.append(corner_capacity)
                else:
                    capacities.append(edge_capacity)
                    for _ in range(cols-2):
                        capacities.append(centre_capacity)
                    capacities.append(edge_capacity)
            
            return capacities
        else:
            if rows == cols:
                cols += 1
            else:
                rows += 1


#### INITAL PLACEMENT
def oee(A, G, N, part=None):
    '''
    :param A: Weight matrix (Chong's weights)
    :param G: Timeslice-Interaction graph (binary)
    :param N: Num of partitions
    :param part: Initial partition, generates random if None

    :return: OEE solution
    '''
    if isinstance(A, sparse.COO):
        A = A.todense()
    
    n_nodes = A.shape[0]
    n_per_part = int(n_nodes / N)

    if part is None:
        part = [i for i in range(N) for _ in range(n_per_part)]
        random.shuffle(part)

    g_max = 1
    swaps = 0

    swapped = []

    # Step 7
    while g_max > 0:
        # Step 1
        C = [i for i in range(n_nodes)]
        index = 0

        W = np.zeros([n_nodes, N])
        D = np.empty([n_nodes, N])

        # Precompute partitions
        P = np.stack([A[np.where(np.array(part) == i)[0]] for i in range(N)])
        
        for i in range(n_nodes):
            for l in range(N):
                W[i, l] = np.sum(P[l, :, i])
        
        for i in range(n_nodes):
            for l in range(N):
                D[i, l] = W[i, l] - W[i, part[i]]

        g = []
        # Step 4
        while len(C) > 1:

            # Step 2
            g.append([-np.inf, None, None])
            for i in C:
                for j in C:
                    g_aux = D[i, part[j]] + D[j, part[i]] - 2*A[i, j]
                    if g_aux > g[index][0]:
                        g[index][0] = g_aux
                        g[index][1] = i
                        g[index][2] = j
            
            a = g[index][1]
            b = g[index][2]

            C.remove(g[index][1])
            if g[index][1] != g[index][2]:
                C.remove(g[index][2])

            # Step 3
            for i in C:
                for l in range(N):
                    if l == part[a]:
                        if part[i] != part[a] and part[i] != part[b]:
                            D[i, l] = D[i, l] + A[i, b] - A[i, a]
                        if part[i] == part[b]:
                            D[i, l] = D[i, l] + 2*A[i, b] - 2*A[i, a]
                    elif l == part[b]:
                        if part[i] != part[a] and part[i] != part[b]:
                            D[i, l] = D[i, l] + A[i, a] - A[i, b]
                        if part[i] == part[a]:
                            D[i, l] = D[i, l] +2*A[i, a] - 2*A[i, b]
                    else:
                        if part[i] == part[a]:
                            D[i, l] = D[i, l] + A[i, a] - A[i, b]
                        elif part[i] == part[b]:
                            D[i, l] = D[i, l] + A[i, b] - A[i, a]
                    
            index += 1
        g_max = np.cumsum([i[0] for i in g])
        m = np.argmax(g_max)
        g_max = g_max[m]

        for i in g[:m+1]:
            # print(f'Swapping nodes {i[1]} and {i[2]}')
            swaps += 1
            part[i[1]], part[i[2]] = part[i[2]], part[i[1]] # Swap
            swapped.append([i[1], i[2]])
            # print(swapped[i[1]][i[2]])
    
    return part, swaps, swapped

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




#### ROUTING CLASSICAL PACKETS
def get_links_used(C1, C2, topology, N):
    # print(C1, C2)
    if topology == 'line':
        if C1 < C2:
            return [(j, j+1) for j in range(C1, C2)]
        else:
            return [(j, j-1) for j in range(C1, C2, -1)]
        
    elif topology == 'ring':
        dist = min(abs(C1-C2), N-abs(C1-C2))
        if C1 < C2:
            direction = abs(C1-C2) < N-abs(C1-C2)
        else:
            direction = abs(C1-C2) > N-abs(C1-C2)

        if direction:
            return [(j%N, (j+1)%N) for j in range(C1, C1+dist)]
        elif not direction:
            return [(j%N, (j-1)%N) for j in range(C1, C1-dist, -1)]
    
    elif topology == 'star':
        # assuming central node is core 0
        if C1 == 0:
            return [(0, C2)]
        elif C2 == 0:
            return [(C1, 0)]
        else:
            return [(C1, 0), (0, C2)]
    
    elif topology == 'grid':
        x, y = np.sqrt(N), np.sqrt(N)

        C1_x, C1_y = int(C1 / y), int(C1 % x)
        C2_x, C2_y = int(C2 / y), int(C2 % x)

        # print(C1_x, C1_y, C2_x, C2_y)

        links = []
        # horizontal links
        if C1_y < C2_y:
            links += [((C1_x, i), (C1_x, i+1)) for i in range(C1_y, C2_y)]
        elif C1_y > C2_y:
            links += [((C1_x, i), (C1_x, i-1)) for i in range(C1_y, C2_y, -1)]

        # vertical links
        if C1_x < C2_x:
            links += [((i, C2_y), (i+1, C2_y)) for i in range(C1_x, C2_x)]
        elif C1_x > C2_x:
            links += [((i, C2_y), (i-1, C2_y)) for i in range(C1_x, C2_x, -1)]

        # print(links)
        cores_per_row = int(np.sqrt(N))
        return [(x1*cores_per_row+y1, x2*cores_per_row+y2) for (x1, y1), (x2, y2) in links]

    elif topology == 'all-to-all':
        return [(C1, C2)]
    

def cycles_for_routing(Ps, qubits, N, link_width, distance_matrix=None, topology='line'):
    if distance_matrix == None:
        distance_matrix = [[1 if j != i else 0 for j in range(N)] for i in range(N)]

    pre_processing_cycles = 2
    packet_size = 2*np.ceil(np.log2(qubits))+2
    hop_cycles = int(np.ceil(packet_size/link_width))
    # print(qubits, link_width, hop_cycles)

    cycles_per_slice = []
    for j in range(len(Ps)-1):
        P1 = Ps[j]
        P2 = Ps[j+1]

        paths = []

        for q in range(qubits):
            if P1[q] != P2[q]:
                movement = get_links_used(P1[q], P2[q], topology, N)
                paths.append((len(movement), movement))
        
        # sort paths by decreasing length
        paths.sort(key=lambda x: x[0], reverse=True)

        routing_cycles = 0
        while True:
            if len(paths) == 0:
                break

            used_links = set()
            finished_paths = []
            for i,path in enumerate(paths):
                # print(path)
                if path[1][0] not in used_links:
                    used_links.add(path[1][0])
                    path[1].pop(0)
                    if len(path[1]) == 0:
                        finished_paths.append(i)

            # remove paths that are done
            for i in finished_paths[::-1]:
                del paths[i]
            
            routing_cycles += (pre_processing_cycles + hop_cycles)
        
        if routing_cycles != 0:
            routing_cycles += pre_processing_cycles # processing cycles in destination node

        cycles_per_slice.append(routing_cycles)
    return cycles_per_slice

def tlp_slices(Ps, qubits, N, comm_qubits, topology):
    """
    Returns the number of sequential teleportations needed to move from one slice to the next
    (i.e., parallel teleportations are limited by comm_qubits).
    """

    teleportation_slices = [0] * (len(Ps) - 1)
    tlp_schedule = []

    for j in range(len(Ps) - 1):
        P1, P2 = Ps[j], Ps[j + 1]
        movements = [
            get_links_used(P1[q], P2[q], topology, N)
            for q in range(qubits) if P1[q] != P2[q]
        ]

        # Sort by path length (descending) to prioritize longer paths
        movements.sort(key=len, reverse=True)

        sequential_steps = 0
        tlp_schedule.append([])

        while movements:
            sequential_steps += 1
            step_links = []
            used_links = defaultdict(int)
            remaining_movements = []

            for path in movements:
                if not path:
                    continue

                link = tuple(sorted(path[0]))  # Normalize link order
                if used_links[link] < comm_qubits:
                    used_links[link] += 1
                    step_links.append(link)
                    path.pop(0)  # Remove used step
                if path:  # Keep path if it still has remaining steps
                    remaining_movements.append(path)

            tlp_schedule[-1].append(step_links)
            movements = remaining_movements  # Update paths for the next iteration

        teleportation_slices[j] = sequential_steps

    return teleportation_slices, tlp_schedule

def tlp_slices_old(Ps, qubits, N, comm_qubits=1, distance_matrix=None, topology='line'):
    # Returns the number of sequential teleporations needed to move from slice to slice (i.e., parallel teleportations <= comm_qubits) 
    if distance_matrix == None:
        distance_matrix = [[1 if j != i else 0 for j in range(N)] for i in range(N)]

    teleportation_slices = [0 for _ in range(len(Ps)-1)]
    tlp = []
    for j in range(len(Ps)-1):
        P1 = Ps[j]
        P2 = Ps[j+1]

        paths = []

        for q in range(qubits):
            if P1[q] != P2[q]:
                movement = get_links_used(P1[q], P2[q], topology, N)
                paths.append((len(movement), movement))
        
        # sort paths by decreasing length
        paths.sort(key=lambda x: x[0], reverse=True)

        sequential_tlp = 0
        tlp.append([])
        while True:
            if len(paths) == 0:
                break

            sequential_tlp += 1
            tlp[-1].append([])

            used_links = dict()
            finished_paths = []
            for i,path in enumerate(paths):
                s_path = (min(path[1][0]), max(path[1][0]))
                if s_path not in used_links:
                    used_links[s_path] = 1
                    tlp[-1][-1].append(s_path)

                    path[1].pop(0)
                    if len(path[1]) == 0:
                        finished_paths.append(i)
                elif used_links[s_path] < comm_qubits:
                    used_links[s_path] += 1
                    # print('\t', s_path, used_links[s_path])
                    tlp[-1][-1].append(s_path)

                    path[1].pop(0)
                    if len(path[1]) == 0:
                        finished_paths.append(i)
                elif used_links[s_path] >= comm_qubits:
                    # print('\tMax link capacity', s_path)
                    continue
                    
            # remove paths that are done
            for i in finished_paths[::-1]:
                del paths[i]

        teleportation_slices[j] = sequential_tlp
    
    return teleportation_slices, tlp    




#### OTHER
def qiskit_circ_to_slices(circ):
    slices_all = []
    slices_two_qubit_gates = []

    dag = circuit_to_dag(circ)
    layers = [dag_to_circuit(layer['graph']) for layer in dag.layers()]
    
    for i, layer_as_circuit in enumerate(layers):

        adj_list_all = []
        adj_list_two_qubit_gates = []
        for instruction in layer_as_circuit:
            if instruction.operation.num_qubits == 2:
                q0 = circ.find_bit(instruction.qubits[0]).index
                q1 = circ.find_bit(instruction.qubits[1]).index
                adj_list_all.extend([((q0, q1), 1), ((q1, q0), 1)])
                adj_list_two_qubit_gates.extend([((q0, q1), 1), ((q1, q0), 1)])
            elif instruction.operation.num_qubits == 1:
                q0 = circ.find_bit(instruction.qubits[0]).index
                adj_list_all.append(((q0, q0), 1))
    
        slices_all.append(sparse.COO(adj_list_all, shape=(circ.num_qubits, circ.num_qubits)))
        slices_two_qubit_gates.append(sparse.COO(adj_list_two_qubit_gates, shape=(circ.num_qubits, circ.num_qubits)))

    return sparse.stack(slices_two_qubit_gates), sparse.stack(slices_all)

def count_non_local_comms(Ps, N, distance_matrix=None):
    if distance_matrix == None:
        distance_matrix = [[1 if j != i else 0 for j in range(N)] for i in range(N)]

    comms = []
    for i in range(1, len(Ps)):
        slice_comms = 0
        for q in range(len(Ps[i])):
            slice_comms += distance_matrix[Ps[i-1][q]][Ps[i][q]]
        comms.append(slice_comms)

    return comms

def get_q_communications_time(
        link_usage, 
        single_qubit_gate_time, 
        two_qubit_gate_time, 
        measurement_time, 
        epr_time, 
        packet_size, 
        link_width, 
        processing_cycles,
        clock_frequency, 
        wireless_bandwidth):
    
    wired_tlp_time = [[] for _ in range(len(link_usage))]
    wireless_tlp_time = [[] for _ in range(len(link_usage))]

    for i, link_usage_slice in enumerate(link_usage):
        if len(link_usage_slice) == 0:
            wired_tlp_time[i].append(0)
            wireless_tlp_time[i].append(0)
            continue

        for parallel_tlp_links in link_usage_slice:
            # (EPR Generation, Pre-Processing, Transmission, Post-Processing)
            wired_tlp_time[i].append((epr_time, two_qubit_gate_time + single_qubit_gate_time + measurement_time, (processing_cycles+np.ceil(packet_size/link_width))/clock_frequency, 2*single_qubit_gate_time + 3*two_qubit_gate_time))
            wireless_tlp_time[i].append((epr_time, two_qubit_gate_time + single_qubit_gate_time + measurement_time, (len(parallel_tlp_links)*packet_size)/wireless_bandwidth, 2*single_qubit_gate_time + 3*two_qubit_gate_time))

    return wired_tlp_time, wireless_tlp_time

def get_computation_time(
        Gs,
        single_qubit_gate_time,
        two_qubit_gate_time):
    comp_time = 0

    for slice in Gs:
        two_qubit_found = False
        for i,row in enumerate(slice):
            if row.nnz > 0:
                if row[i] == 0: # 2-qubit gate
                    comp_time += two_qubit_gate_time
                    two_qubit_found = True
                    continue
        if not two_qubit_found:
            comp_time += single_qubit_gate_time     

    return comp_time 

def get_cores(virtual_qubits, comm_qubits_per_link, qubits_per_core, topology):
    if topology == 'line':
        core_capacities = get_line_cores(virtual_qubits, comm_qubits_per_link, qubits_per_core)
        distance_matrix = line_topology(len(core_capacities)) if core_capacities != -1 else -1 
    elif topology == 'ring':
        core_capacities = get_ring_cores(virtual_qubits, comm_qubits_per_link, qubits_per_core)
        distance_matrix = ring_topology(len(core_capacities)) if core_capacities != -1 else -1
    elif topology == 'star':
        core_capacities = get_star_cores(virtual_qubits, comm_qubits_per_link, qubits_per_core)
        distance_matrix = star_topology(len(core_capacities)) if core_capacities != -1 else -1
    elif topology == 'grid':
        core_capacities = get_grid_cores(virtual_qubits, comm_qubits_per_link, qubits_per_core)
        distance_matrix = grid_topology(len(core_capacities)) if core_capacities != -1 else -1
    elif topology == 'all-to-all':
        core_capacities = get_all_to_all_cores(virtual_qubits, comm_qubits_per_link, qubits_per_core)
        distance_matrix = all_to_all_topology(len(core_capacities)) if core_capacities != -1 else -1
    
    return core_capacities, distance_matrix

def get_operational_fidelity_esp(
        Gs,
        Ps,
        distance_matrix,
        single_qubit_gate_error = 1e-4,
        two_qubit_gate_error = 5e-4,
        meas_error = 0,
        EPR_error = 0): # Very simple error model based on ESP NOTE: Not used in the paper
    
    comms = 0
    for i in range(1, len(Ps)):
        for q in range(len(Ps[i])):
            comms += distance_matrix[Ps[i-1][q]][Ps[i][q]]
    
    tlp_fid = (1-EPR_error) * (1-two_qubit_gate_error)**4 * (1-single_qubit_gate_error)**3 * (1-meas_error)**2

    circ_fid = 1
    for slice in Gs:
        for i,row in enumerate(slice):
            if row[i] == 1:
                circ_fid *= (1-single_qubit_gate_error)
            for j in range(i+1, len(row)):
                if row[j] == 1:
                    circ_fid *= (1-two_qubit_gate_error)
        
    return circ_fid * tlp_fid**comms


def get_operational_fidelity_depol(
        Gs,
        Ps,
        distance_matrix,
        single_qubit_gate_error = 1e-4,
        two_qubit_gate_error = 5e-4,
        meas_error = 0,
        EPR_error = 0):
    
    p1 = 2*(1-single_qubit_gate_error-1)/(1-2)
    p2 = 4*(1-two_qubit_gate_error-1)/(1-4)

    qubits = len(Ps[0])
    qubit_fidelities = [1 for _ in range(qubits)]

    for s,slice in enumerate(Gs):
        for i,row in enumerate(slice):
            if row[i] == 1:
                qubit_fidelities[i] = (1-p1) * qubit_fidelities[i] + p1/2
            for j in range(i+1, len(row)):
                if row[j] == 1:
                    fid_1 = qubit_fidelities[i]
                    fid_2 = qubit_fidelities[j]

                    eta = 1/2 *(np.sqrt((1-p2) * (fid_1 + fid_2)**2 + p2) - np.sqrt(1-p2) * (fid_1 + fid_2))

                    qubit_fidelities[i] = np.sqrt(1-p2) * fid_1 + eta
                    qubit_fidelities[j] = np.sqrt(1-p2) * fid_2 + eta

        # Non-local Communications
        for q in range(qubits):
            if Ps[s][q] != Ps[s+1][q]:
                for _ in range(distance_matrix[Ps[s][q]][Ps[s+1][q]]):
                    # Generate entanglement with fidelity sqrt(1-EPR_error)
                    epr_A_fid = np.sqrt(1-EPR_error) # Generate entanglement with fidelity sqrt(1-EPR_error)

                    # Preprocessing
                    eta = 1/2 *(np.sqrt((1-p2) * (epr_A_fid + qubit_fidelities[q])**2 + p2) - np.sqrt(1-p2) * (epr_A_fid + qubit_fidelities[q]))
                    qubit_fidelities[q] = np.sqrt(1-p2) * qubit_fidelities[q] + eta

                    qubit_fidelities[q] = (1-p1) * qubit_fidelities[q] + p1/2

                    qubit_fidelities[q] *= (1 - meas_error)

                    # Classical Communications

                    # Postprocessing
                    buffer_B_fid = 1

                    qubit_fidelities[q] = (1-p1) * qubit_fidelities[q] + p1/2 # X or Z gate (on average, only one gate will be executed)

                    eta = 1/2 *(np.sqrt((1-p2) * (buffer_B_fid + qubit_fidelities[q])**2 + p2) - np.sqrt(1-p2) * (buffer_B_fid + qubit_fidelities[q])) # First CNOT
                    qubit_fidelities[q] = np.sqrt(1-p2) * qubit_fidelities[q] + eta
                    buffer_B_fid = np.sqrt(1-p2) * buffer_B_fid + eta

                    eta = 1/2 *(np.sqrt((1-p2) * (buffer_B_fid + qubit_fidelities[q])**2 + p2) - np.sqrt(1-p2) * (buffer_B_fid + qubit_fidelities[q])) # Second CNOT
                    qubit_fidelities[q] = np.sqrt(1-p2) * qubit_fidelities[q] + eta
                    buffer_B_fid = np.sqrt(1-p2) * buffer_B_fid + eta

                    eta = 1/2 *(np.sqrt((1-p2) * (buffer_B_fid + qubit_fidelities[q])**2 + p2) - np.sqrt(1-p2) * (buffer_B_fid + qubit_fidelities[q])) # Third CNOT
                    qubit_fidelities[q] = np.sqrt(1-p2) * qubit_fidelities[q] + eta
                    buffer_B_fid = np.sqrt(1-p2) * buffer_B_fid + eta

    return np.prod(qubit_fidelities)


def full_process(
        Gs,
        # virtual_qubits = 64,
        qubits_per_core = 20,
        comm_qubits_per_link = [1],
        # QCores = 4,
        single_qubit_gate_time = 20e-9,
        two_qubit_gate_time = 100e-9,
        meas_time = 100e-9,
        EPR_time = 200e-9,
        single_qubit_gate_error = 1e-5,
        two_qubit_gate_error = 1e-4,
        meas_error = 0,
        EPR_error = 0,
        T1 = 1e-3,
        T2 = 1e-3,
        # packet_size = 2* np.ceil(np.log2(qubits_per_core * QCores)) + 2,
        link_width = 1,
        proc_cycles = 2,
        clk_freq = 100e6,
        topology='line',
        trials=10):
    
    virtual_qubits = Gs[0].shape[0]

    comm_times_list = [[] for _ in comm_qubits_per_link]
    comp_times_list = [[] for _ in comm_qubits_per_link]

    operational_fidelity_list = [[] for _ in comm_qubits_per_link]
    
    time_decoherence_list = [[] for _ in comm_qubits_per_link]

    core_capacities_list = []

    for c,comm_qubits in enumerate(comm_qubits_per_link):
        print(f'\nComm qubits: {comm_qubits}')
        core_capacities, distance_matrix = get_cores(virtual_qubits, comm_qubits, qubits_per_core, topology)
        print('\tCore capacities:', core_capacities)

        if core_capacities == -1:
            print('Skipping')
            continue

        QCores = len(core_capacities)
        packet_size = 2* np.ceil(np.log2(qubits_per_core * QCores)) + 2

        core_capacities_list.append(core_capacities)

        for _ in range(trials):
            part = [i for i in range(len(core_capacities)) for _ in range(core_capacities[i])][:virtual_qubits]

            while True:
                try:
                    random.shuffle(part) # Random initial partition

                    Ps = np.zeros((len(Gs)+1, virtual_qubits), dtype=int)
                    Ps[0] = part
                    Ps_HQA = HQA_variation(Gs, Ps.copy(), QCores, virtual_qubits, core_capacities.copy(), distance_matrix=distance_matrix)

                    break
                except:
                    print('Error in HQA, retrying')

            num_tlp_slices, link_usage = tlp_slices(Ps_HQA, virtual_qubits, QCores, comm_qubits=comm_qubits, distance_matrix=distance_matrix, topology=topology)
            operational_fidelity = get_operational_fidelity_depol(Gs, Ps_HQA, distance_matrix, single_qubit_gate_error, two_qubit_gate_error, meas_error, EPR_error)
            comm_times, _ = get_q_communications_time(link_usage, single_qubit_gate_time, two_qubit_gate_time, meas_time, EPR_time, packet_size, link_width, proc_cycles, clk_freq, 1)
            comp_time = get_computation_time(Gs, single_qubit_gate_time, two_qubit_gate_time)

            comm_time = sum([sum([sum(tlp_tuple) for tlp_tuple in tlp_slice]) for tlp_slice in comm_times if not np.array_equal([0], tlp_slice)])
            comm_times_list[c].append(comm_time)
            comp_times_list[c].append(comp_time)

            operational_fidelity_list[c].append(operational_fidelity)

            total_time = comm_time + comp_time
            decoherence_over_time = np.exp(-total_time/T1) * (1/2 * np.exp(-total_time/T2) + 1/2)
            time_decoherence_list[c].append(decoherence_over_time)
        
    return comp_times_list, comm_times_list, operational_fidelity_list, time_decoherence_list, core_capacities_list

def get_circuit_fidelity( # Similar to full_process but not executing all steps (e.g. compilation). Used in delta_exploration() func
        Gs,
        Ps,
        distance_matrix,
        link_usage,
        single_qubit_gate_time = 20e-9,
        two_qubit_gate_time = 100e-9,
        meas_time = 100e-9,
        EPR_time = 200e-9,
        single_qubit_gate_error = 1e-4,
        two_qubit_gate_error = 5e-4,
        meas_error = 0,
        EPR_error = 0,
        T1 = 1e-3,
        T2 = 1e-3,
        link_width = 15,
        proc_cycles = 2,
        clk_freq = 500e6,
        topology='line'):
    
    qubits = len(Ps[0])
    packet_size = 2*np.ceil(np.log2(qubits))+2

    comms_times = 0

    for i, link_usage_slice in enumerate(link_usage):
        for parallel_tlp_links in link_usage_slice:
            comms_times += (EPR_time + two_qubit_gate_time + single_qubit_gate_time + meas_time + (proc_cycles+np.ceil(packet_size/link_width))/clk_freq + 1*single_qubit_gate_time + 3*two_qubit_gate_time)


    p1 = 2*(1-single_qubit_gate_error-1)/(1-2)
    p2 = 4*(1-two_qubit_gate_error-1)/(1-4)

    qubit_fidelities = [1 for _ in range(qubits)]
    comp_time = 0

    for s,slice in enumerate(Gs):
        two_qubit_gate_found = False
        for i,row in enumerate(slice):
            if row[i] == 1:
                qubit_fidelities[i] = (1-p1) * qubit_fidelities[i] + p1/2
            for j in range(i+1, len(row)):
                if row[j] == 1:
                    fid_1 = qubit_fidelities[i]
                    fid_2 = qubit_fidelities[j]

                    eta = 1/2 *(np.sqrt((1-p2) * (fid_1 + fid_2)**2 + p2) - np.sqrt(1-p2) * (fid_1 + fid_2))

                    qubit_fidelities[i] = np.sqrt(1-p2) * fid_1 + eta
                    qubit_fidelities[j] = np.sqrt(1-p2) * fid_2 + eta

                    two_qubit_found = True
        if not two_qubit_gate_found:
            comp_time += single_qubit_gate_time
        else:
            comp_time += two_qubit_gate_time

        # Non-local Communications
        for q in range(qubits):
            if Ps[s][q] != Ps[s+1][q]:
                for _ in range(distance_matrix[Ps[s][q]][Ps[s+1][q]]):
                    # Generate entanglement with fidelity sqrt(1-EPR_error)
                    epr_A_fid = np.sqrt(1-EPR_error) # Generate entanglement with fidelity sqrt(1-EPR_error)

                    # Preprocessing
                    eta = 1/2 *(np.sqrt((1-p2) * (epr_A_fid + qubit_fidelities[q])**2 + p2) - np.sqrt(1-p2) * (epr_A_fid + qubit_fidelities[q]))
                    qubit_fidelities[q] = np.sqrt(1-p2) * qubit_fidelities[q] + eta

                    qubit_fidelities[q] = (1-p1) * qubit_fidelities[q] + p1/2

                    qubit_fidelities[q] *= (1 - meas_error)

                    # Classical Communications

                    # Postprocessing
                    buffer_B_fid = 1

                    qubit_fidelities[q] = (1-p1) * qubit_fidelities[q] + p1/2 # X or Z gate (on average, only one gate will be executed)

                    eta = 1/2 *(np.sqrt((1-p2) * (buffer_B_fid + qubit_fidelities[q])**2 + p2) - np.sqrt(1-p2) * (buffer_B_fid + qubit_fidelities[q])) # First CNOT
                    qubit_fidelities[q] = np.sqrt(1-p2) * qubit_fidelities[q] + eta
                    buffer_B_fid = np.sqrt(1-p2) * buffer_B_fid + eta

                    eta = 1/2 *(np.sqrt((1-p2) * (buffer_B_fid + qubit_fidelities[q])**2 + p2) - np.sqrt(1-p2) * (buffer_B_fid + qubit_fidelities[q])) # Second CNOT
                    qubit_fidelities[q] = np.sqrt(1-p2) * qubit_fidelities[q] + eta
                    buffer_B_fid = np.sqrt(1-p2) * buffer_B_fid + eta

                    eta = 1/2 *(np.sqrt((1-p2) * (buffer_B_fid + qubit_fidelities[q])**2 + p2) - np.sqrt(1-p2) * (buffer_B_fid + qubit_fidelities[q])) # Third CNOT
                    qubit_fidelities[q] = np.sqrt(1-p2) * qubit_fidelities[q] + eta
                    buffer_B_fid = np.sqrt(1-p2) * buffer_B_fid + eta


    total_time = comms_times + comp_time
    decoherence_over_time = np.exp(-total_time/T1) * (1/2 * np.exp(-total_time/T2) + 1/2)
    return np.prod(qubit_fidelities) * decoherence_over_time

def delta_exploration(
        Gs,
        qubits_per_core = 20,
        comm_qubits_per_link = 1,
        single_qubit_gate_time = 20e-9,
        two_qubit_gate_time = 100e-9,
        meas_time = 100e-9,
        EPR_time = 200e-9,
        single_qubit_gate_error = 1e-5,
        two_qubit_gate_error = 1e-4,
        meas_error = 0,
        EPR_error = 0,
        T1 = 1e-3,
        T2 = 1e-3,
        link_width = 1,
        proc_cycles = 2,
        clk_freq = 100e6,
        topology='line',
        delta_time = [1],
        delta_error = [1]):
    
    virtual_qubits =  Gs[0].shape[0]

    core_capacities, distance_matrix = get_cores(virtual_qubits, comm_qubits_per_link, qubits_per_core, topology)
    QCores = len(core_capacities)

    part = [i for i in range(len(core_capacities)) for _ in range(core_capacities[i])][:virtual_qubits]
    Ps = np.zeros((len(Gs)+1, virtual_qubits), dtype=int)
    Ps[0] = part
    Ps_HQA = HQA_variation(Gs, Ps.copy(), QCores, virtual_qubits, core_capacities.copy(), distance_matrix=distance_matrix)

    num_tlp_slices, link_usage = tlp_slices(Ps_HQA, virtual_qubits, QCores, comm_qubits_per_link, topology)
    
    delta_fidelities = [[] for _ in delta_time]

    for d,dt in enumerate(delta_time):
        for de in delta_error:
            print(f'\nDelta time: {dt} Delta error: {de}')

            delta_single_qubit_gate_time = single_qubit_gate_time/dt
            delta_two_qubit_gate_time = two_qubit_gate_time/dt
            delta_meas_time = meas_time/dt
            delta_EPR_time = EPR_time/dt
            delta_single_qubit_gate_error = single_qubit_gate_error/de
            delta_two_qubit_gate_error = two_qubit_gate_error/de
            delta_meas_error = meas_error/de
            delta_EPR_error = EPR_error/de
            delta_T1 = T1*dt
            delta_T2 = T2*dt

            circ_fid = get_circuit_fidelity(
                Gs,
                Ps_HQA,
                distance_matrix,
                link_usage,
                delta_single_qubit_gate_time,
                delta_two_qubit_gate_time,
                delta_meas_time,
                delta_EPR_time,
                delta_single_qubit_gate_error,
                delta_two_qubit_gate_error,
                delta_meas_error,
                delta_EPR_error,
                delta_T1,
                delta_T2,
                link_width,
                proc_cycles,
                clk_freq,
                topology)
            
            print(f'\tCircuit Fidelity: {circ_fid}')

            delta_fidelities[d].append(circ_fid)

    return delta_fidelities


def compilation_only(
        Gs,
        qubits_per_core = 20,
        comm_qubits_per_link = [1],
        topology='line',
        trials=10):
    
    non_local_comms = [[] for _ in comm_qubits_per_link]
    non_local_slices = [[] for _ in comm_qubits_per_link]

    virtual_qubits = Gs[0].shape[0]

    for c,comm_qubits in enumerate(comm_qubits_per_link):
        print(f'\nComm qubits: {comm_qubits}')
        core_capacities, distance_matrix = get_cores(virtual_qubits, comm_qubits, qubits_per_core, topology)
        print('\tCore capacities:', core_capacities)

        if core_capacities == -1:
            print('Skipping')
            continue

        QCores = len(core_capacities)

        for _ in range(trials):
            part = [i for i in range(len(core_capacities)) for _ in range(core_capacities[i])][:virtual_qubits]

            while True:
                try:
                    random.shuffle(part) # Random initial partition

                    Ps = np.zeros((len(Gs)+1, virtual_qubits), dtype=int)
                    Ps[0] = part
                    Ps_HQA = HQA_variation(Gs, Ps.copy(), QCores, virtual_qubits, core_capacities.copy(), distance_matrix=distance_matrix)

                    break
                except:
                    print('Error in HQA, retrying')

            num_tlp_slices, link_usage = tlp_slices(Ps_HQA, virtual_qubits, QCores, comm_qubits=comm_qubits, distance_matrix=distance_matrix, topology=topology)
            
            non_local_comms[c].append(sum(count_non_local_comms(Ps_HQA, QCores, distance_matrix)))
            non_local_slices[c].append(sum(num_tlp_slices))
    
    return non_local_comms, non_local_slices

def comm_time_decomposition(
        physical_qubits,
        single_qubit_gate_time = 20e-9,
        two_qubit_gate_time = 100e-9,
        meas_time = 100e-9,
        EPR_time = 200e-9,
        link_width = 1,
        proc_cycles = 2,
        clk_freq = 100e6):
    
    # print(f'\nPhysical qubits: {physical_qubits}', type(physical_qubits))
    packet_size = 2* np.ceil(np.log2(physical_qubits)) + 2
    tlp_time = (EPR_time, two_qubit_gate_time + single_qubit_gate_time + meas_time, (proc_cycles+np.ceil(packet_size/link_width))/clk_freq, 2*single_qubit_gate_time + 3*two_qubit_gate_time)

    return tlp_time

def run_trial(Gs, virtual_qubits, core_capacities, distance_matrix, comm_qubits, QCores, comp_time, single_qubit_gate_time, 
              two_qubit_gate_time, meas_time, EPR_time, single_qubit_gate_error, two_qubit_gate_error, meas_error, EPR_error, 
              T1, T2, link_width, proc_cycles, clk_freq, topology, packet_size):
    while True:
        try:
            random_partition = [i for i in range(len(core_capacities)) for _ in range(core_capacities[i])][:virtual_qubits]
            random.shuffle(random_partition)
            Ps = np.zeros((len(Gs) + 1, virtual_qubits), dtype=int)
            Ps[0] = random_partition
            
            Ps_HQA = HQA_variation(Gs, Ps.copy(), QCores, virtual_qubits, core_capacities.copy(), distance_matrix=distance_matrix)
            break
        except:
            print('Error in HQA, retrying')

    num_tlp_slices, link_usage = tlp_slices(Ps_HQA, virtual_qubits, QCores, comm_qubits=comm_qubits, 
                                            distance_matrix=distance_matrix, topology=topology)
    non_local_comm = sum(count_non_local_comms(Ps_HQA, QCores, distance_matrix))
    non_local_slice = sum(num_tlp_slices)

    operational_fidelity = get_operational_fidelity_depol(Gs, Ps_HQA, distance_matrix, single_qubit_gate_error, 
                                                          two_qubit_gate_error, meas_error, EPR_error)

    comm_times, _ = get_q_communications_time(link_usage, single_qubit_gate_time, two_qubit_gate_time, meas_time, 
                                              EPR_time, packet_size, link_width, proc_cycles, clk_freq, 1)
    comm_time = sum([sum([sum(tlp_tuple) for tlp_tuple in tlp_slice]) for tlp_slice in comm_times if not np.array_equal([0], tlp_slice)])

    total_time = comm_time + comp_time
    decoherence_over_time = np.exp(-total_time / T1) * (1 / 2 * np.exp(-total_time / T2) + 1 / 2)
    
    return comm_time, operational_fidelity, decoherence_over_time, non_local_comm, non_local_slice

def qlink_exploration(
        Gs, qubits_per_core, comm_qubits_per_link, single_qubit_gate_time, two_qubit_gate_time, meas_time, EPR_time,
        single_qubit_gate_error, two_qubit_gate_error, meas_error, EPR_error, T1, T2, link_width, proc_cycles, clk_freq,
        topology, trials):
    
    virtual_qubits = Gs[0].shape[0]
    comp_time = get_computation_time(Gs, single_qubit_gate_time, two_qubit_gate_time)

    comm_times_list = [[] for _ in comm_qubits_per_link]
    comp_times_list = [[] for _ in comm_qubits_per_link]
    operational_fidelity_list = [[] for _ in comm_qubits_per_link]
    time_decoherence_list = [[] for _ in comm_qubits_per_link]
    core_capacities_list = []
    non_local_comms = [[] for _ in comm_qubits_per_link]
    non_local_slices = [[] for _ in comm_qubits_per_link]

    for c, comm_qubits in enumerate(comm_qubits_per_link):
        print(f'\tComm qubits: {comm_qubits}')
        core_capacities, distance_matrix = get_cores(virtual_qubits, comm_qubits, qubits_per_core, topology)
        
        if core_capacities == -1:
            print('Skipping')
            continue

        QCores = len(core_capacities)
        packet_size = 2 * np.ceil(np.log2(qubits_per_core * QCores)) + 2
        
        results = Parallel(n_jobs=-1)(delayed(run_trial)(Gs, virtual_qubits, core_capacities, distance_matrix, comm_qubits, QCores, comp_time,
                                                          single_qubit_gate_time, two_qubit_gate_time, meas_time, EPR_time, single_qubit_gate_error,
                                                          two_qubit_gate_error, meas_error, EPR_error, T1, T2, link_width, proc_cycles, clk_freq, topology, packet_size)
                                     for _ in range(trials))
        
        for comm_time, operational_fidelity, decoherence_over_time, non_local_comm, non_local_slice in results:
            comm_times_list[c].append(comm_time)
            comp_times_list[c].append(comp_time)
            operational_fidelity_list[c].append(operational_fidelity)
            time_decoherence_list[c].append(decoherence_over_time)
            non_local_comms[c].append(non_local_comm)
            non_local_slices[c].append(non_local_slice)

    return comp_times_list, comm_times_list, operational_fidelity_list, time_decoherence_list, core_capacities_list, non_local_comms, non_local_slices