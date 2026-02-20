import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT, CDKMRippleCarryAdder, QuantumVolume

from utils import *
import csv
import sys

def GHZ(num_qubits):
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    for i in range(1, num_qubits):
        qc.cx(0, i)

    return qc

########## TO DEFINE ##########
virtual_qubits = 1024
qubits_per_core = 128
comm_qubits_per_link = np.arange(1, 2)

# obtain circuit name and error and time delta from arguments
circ_name = sys.argv[1]
time_delta = int(sys.argv[2])
error_delta = int(sys.argv[2])

trials = 5
###############################

single_qubit_gate_time = 7.9e-9
two_qubit_gate_time = 30e-9
meas_time = 40e-9
EPR_time = 130e-9
single_qubit_gate_error = 7.42e-5
two_qubit_gate_error = 7e-4
meas_error = 1.67e-4
EPR_error = 9e-3
T1 = 1.2e-3
T2 = 1.16e-3
link_width = 10
proc_cycles = 2
clk_freq = 200e6

single_qubit_gate_time = single_qubit_gate_time/time_delta
two_qubit_gate_time = two_qubit_gate_time/time_delta
meas_time = meas_time/time_delta
EPR_time = EPR_time/time_delta
single_qubit_gate_error = single_qubit_gate_error/error_delta
two_qubit_gate_error = two_qubit_gate_error/error_delta
meas_error = meas_error/error_delta
EPR_error = EPR_error/error_delta
T1 = T1*time_delta
T2 = T2*time_delta

if circ_name == 'qft':
    circ = QFT(virtual_qubits)
    transp_circ = transpile(circ, basis_gates=['x', 'cx', 'cp', 'rz', 'h', 's', 'sdg', 't', 'tdg', 'measure'])
elif circ_name == 'cuccaro':
    circ = CDKMRippleCarryAdder(int(virtual_qubits/2)-1)
    transp_circ = transpile(circ, basis_gates=['x', 'cx', 'cp', 'rz', 'h', 's', 'sdg', 't', 'tdg', 'measure'])
elif circ_name == 'qvol':
    transp_circ = QuantumVolume(virtual_qubits)
elif circ_name == 'ghz':
    circ = GHZ(virtual_qubits)
    transp_circ = transpile(circ, basis_gates=['x', 'cx', 'cp', 'rz', 'h', 's', 'sdg', 't', 'tdg', 'measure'])
else:
    raise ValueError("Invalid circuit name")

Gs, Gs_all = qiskit_circ_to_slices(transp_circ)

line_capacities = [get_line_cores(virtual_qubits, comm_qubits, qubits_per_core) for comm_qubits in comm_qubits_per_link if get_line_cores(virtual_qubits, comm_qubits, qubits_per_core) != -1]
ring_capacities = [get_ring_cores(virtual_qubits, comm_qubits, qubits_per_core) for comm_qubits in comm_qubits_per_link if get_ring_cores(virtual_qubits, comm_qubits, qubits_per_core) != -1]
star_capacities = [get_star_cores(virtual_qubits, comm_qubits, qubits_per_core) for comm_qubits in comm_qubits_per_link if get_star_cores(virtual_qubits, comm_qubits, qubits_per_core) != -1]
grid_capacities = [get_grid_cores(virtual_qubits, comm_qubits, qubits_per_core) for comm_qubits in comm_qubits_per_link if get_grid_cores(virtual_qubits, comm_qubits, qubits_per_core) != -1]
all_to_all_capacities = [get_all_to_all_cores(virtual_qubits, comm_qubits, qubits_per_core) for comm_qubits in comm_qubits_per_link if get_all_to_all_cores(virtual_qubits, comm_qubits, qubits_per_core) != -1]

print('Line topology')
line_comp_times_list, line_comm_times_list, line_operational_fid_list, list_time_decoherence, line_core_capacities_list, line_non_local_comms, line_non_local_slices = qlink_exploration(
    Gs=Gs_all,
    qubits_per_core=qubits_per_core,
    comm_qubits_per_link=comm_qubits_per_link,
    single_qubit_gate_time=single_qubit_gate_time,
    two_qubit_gate_time=two_qubit_gate_time,
    meas_time=meas_time,
    EPR_time=EPR_time,
    single_qubit_gate_error=single_qubit_gate_error,
    two_qubit_gate_error=two_qubit_gate_error,
    meas_error=meas_error,
    EPR_error=EPR_error,
    T1=T1,
    T2=T2,
    link_width=link_width,
    proc_cycles=proc_cycles,
    clk_freq=clk_freq,
    topology='line',
    trials=trials
)
line_total_time = [[line_comp_times_list[i][j] + line_comm_times_list[i][j] for j in range(len(line_comp_times_list[i]))] for i in range(len(line_comp_times_list))]
line_op_fid = [[line_operational_fid_list[i][j] for j in range(len(line_operational_fid_list[i]))] for i in range(len(line_operational_fid_list))]
line_estimated_fid = [[line_operational_fid_list[i][j] * list_time_decoherence[i][j] for j in range(len(line_operational_fid_list[i]))] for i in range(len(line_operational_fid_list))]

print('Ring topology')
ring_comp_times_list, ring_comm_times_list, ring_operational_fid_list, ring_time_decoherence, ring_core_capacities_list, ring_non_local_comms, ring_non_local_slices = qlink_exploration(
    Gs=Gs_all,
    qubits_per_core=qubits_per_core,
    comm_qubits_per_link=comm_qubits_per_link,
    single_qubit_gate_time=single_qubit_gate_time,
    two_qubit_gate_time=two_qubit_gate_time,
    meas_time=meas_time,
    EPR_time=EPR_time,
    single_qubit_gate_error=single_qubit_gate_error,
    two_qubit_gate_error=two_qubit_gate_error,
    meas_error=meas_error,
    EPR_error=EPR_error,
    T1=T1,
    T2=T2,
    link_width=link_width,
    proc_cycles=proc_cycles,
    clk_freq=clk_freq,
    topology='ring',
    trials=trials
)
ring_total_time = [[ring_comp_times_list[i][j] + ring_comm_times_list[i][j] for j in range(len(ring_comp_times_list[i]))] for i in range(len(ring_comp_times_list))]
ring_op_fid = [[ring_operational_fid_list[i][j] for j in range(len(ring_operational_fid_list[i]))] for i in range(len(ring_operational_fid_list))]
ring_estimated_fid = [[ring_operational_fid_list[i][j] * ring_time_decoherence[i][j] for j in range(len(ring_operational_fid_list[i]))] for i in range(len(ring_operational_fid_list))]

print('Star topology')
star_comp_times_list, star_comm_times_list, star_operational_fid_list, star_time_decoherence, star_core_capacities_list, star_non_local_comms, star_non_local_slices = qlink_exploration(
    Gs=Gs_all,
    qubits_per_core=qubits_per_core,
    comm_qubits_per_link=comm_qubits_per_link,
    single_qubit_gate_time=single_qubit_gate_time,
    two_qubit_gate_time=two_qubit_gate_time,
    meas_time=meas_time,
    EPR_time=EPR_time,
    single_qubit_gate_error=single_qubit_gate_error,
    two_qubit_gate_error=two_qubit_gate_error,
    meas_error=meas_error,
    EPR_error=EPR_error,
    T1=T1,
    T2=T2,
    link_width=link_width,
    proc_cycles=proc_cycles,
    clk_freq=clk_freq,
    topology='star',
    trials=trials
)
star_total_time = [[star_comp_times_list[i][j] + star_comm_times_list[i][j] for j in range(len(star_comp_times_list[i]))] for i in range(len(star_comp_times_list))]
star_op_fid = [[star_operational_fid_list[i][j] for j in range(len(star_operational_fid_list[i]))] for i in range(len(star_operational_fid_list))]
star_estimated_fid = [[star_operational_fid_list[i][j] * star_time_decoherence[i][j] for j in range(len(star_operational_fid_list[i]))] for i in range(len(star_operational_fid_list))]

print('Grid topology')
grid_comp_times_list, grid_comm_times_list, grid_operational_fid_list, grid_time_decoherence, grid_core_capacities_list, grid_non_local_comms, grid_non_local_slices = qlink_exploration(
    Gs=Gs_all,
    qubits_per_core=qubits_per_core,
    comm_qubits_per_link=comm_qubits_per_link,
    single_qubit_gate_time=single_qubit_gate_time,
    two_qubit_gate_time=two_qubit_gate_time,
    meas_time=meas_time,
    EPR_time=EPR_time,
    single_qubit_gate_error=single_qubit_gate_error,
    two_qubit_gate_error=two_qubit_gate_error,
    meas_error=meas_error,
    EPR_error=EPR_error,
    T1=T1,
    T2=T2,
    link_width=link_width,
    proc_cycles=proc_cycles,
    clk_freq=clk_freq,
    topology='grid',
    trials=trials
)
grid_total_time = [[grid_comp_times_list[i][j] + grid_comm_times_list[i][j] for j in range(len(grid_comp_times_list[i]))] for i in range(len(grid_comp_times_list))]
grid_op_fid = [[grid_operational_fid_list[i][j] for j in range(len(grid_operational_fid_list[i]))] for i in range(len(grid_operational_fid_list))]
grid_estimated_fid = [[grid_operational_fid_list[i][j] * grid_time_decoherence[i][j] for j in range(len(grid_operational_fid_list[i]))] for i in range(len(grid_operational_fid_list))]

print('All-to-all topology')
all_to_all_comp_times_list, all_to_all_comm_times_list, all_to_all_operational_fid_list, all_to_all_time_decoherence, all_to_all_core_capacities_list, all_to_all_non_local_comms, all_to_all_non_local_slices = qlink_exploration(
    Gs=Gs_all,
    qubits_per_core=qubits_per_core,
    comm_qubits_per_link=comm_qubits_per_link,
    single_qubit_gate_time=single_qubit_gate_time,
    two_qubit_gate_time=two_qubit_gate_time,
    meas_time=meas_time,
    EPR_time=EPR_time,
    single_qubit_gate_error=single_qubit_gate_error,
    two_qubit_gate_error=two_qubit_gate_error,
    meas_error=meas_error,
    EPR_error=EPR_error,
    T1=T1,
    T2=T2,
    link_width=link_width,
    proc_cycles=proc_cycles,
    clk_freq=clk_freq,
    topology='all-to-all',
    trials=trials
)
all_to_all_total_time = [[all_to_all_comp_times_list[i][j] + all_to_all_comm_times_list[i][j] for j in range(len(all_to_all_comp_times_list[i]))] for i in range(len(all_to_all_comp_times_list))]
all_to_all_op_fid = [[all_to_all_operational_fid_list[i][j] for j in range(len(all_to_all_operational_fid_list[i]))] for i in range(len(all_to_all_operational_fid_list))]
all_to_all_estimated_fid = [[all_to_all_operational_fid_list[i][j] * all_to_all_time_decoherence[i][j] for j in range(len(all_to_all_operational_fid_list[i]))] for i in range(len(all_to_all_operational_fid_list))]


# Add empty lists to match the length of comm_qubits_per_link to topology capacities
line_capacities += [[] for _ in range(len(comm_qubits_per_link) - len(line_capacities))]
ring_capacities += [[] for _ in range(len(comm_qubits_per_link) - len(ring_capacities))]
star_capacities += [[] for _ in range(len(comm_qubits_per_link) - len(star_capacities))]
grid_capacities += [[] for _ in range(len(comm_qubits_per_link) - len(grid_capacities))]
all_to_all_capacities += [[] for _ in range(len(comm_qubits_per_link) - len(all_to_all_capacities))]

# Save results to CSV
with open(f'QLink_exploration_{circ_name}_{virtual_qubits}_{qubits_per_core}.csv', mode='w') as file:
    writer = csv.writer(file)

    writer.writerow([virtual_qubits])
    writer.writerow([qubits_per_core])
    writer.writerow(comm_qubits_per_link)
    writer.writerow([time_delta])
    writer.writerow([error_delta])
    writer.writerow([trials])

    writer.writerows(line_capacities)
    writer.writerows(ring_capacities)
    writer.writerows(star_capacities)
    writer.writerows(grid_capacities)
    writer.writerows(all_to_all_capacities)

    writer.writerows(line_total_time)
    writer.writerows(ring_total_time)
    writer.writerows(star_total_time)
    writer.writerows(grid_total_time)
    writer.writerows(all_to_all_total_time)

    writer.writerows(line_op_fid)
    writer.writerows(ring_op_fid)
    writer.writerows(star_op_fid)
    writer.writerows(grid_op_fid)
    writer.writerows(all_to_all_op_fid)

    writer.writerows(line_estimated_fid)
    writer.writerows(ring_estimated_fid)
    writer.writerows(star_estimated_fid)
    writer.writerows(grid_estimated_fid)
    writer.writerows(all_to_all_estimated_fid)

    writer.writerows(line_non_local_comms)
    writer.writerows(ring_non_local_comms)
    writer.writerows(star_non_local_comms)
    writer.writerows(grid_non_local_comms)
    writer.writerows(all_to_all_non_local_comms)

    writer.writerows(line_non_local_slices)
    writer.writerows(ring_non_local_slices)
    writer.writerows(star_non_local_slices)
    writer.writerows(grid_non_local_slices)
    writer.writerows(all_to_all_non_local_slices)
