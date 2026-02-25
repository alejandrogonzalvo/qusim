# Qusim Noise Model

The `qusim::noise` module provides a discrete-time fidelity simulator for evaluating the physical performance of hardware-aware compiling (HAC) pipelines and qubit routing.

Unlike full unitary state-vector simulators which scale exponentially and fail for $>30$ qubits, our fidelity model acts as an aggregated error budget tracker. It parses a mapped quantum DAG and assigns temporal penalties across the individual logical qubits as they wait in memory, execute local gates, or undergo state teleportation.

## Fidelity Equations

We model three disjoint hardware error mechanisms:

### 1. Algorithmic Gate Error 

Operational gate error measures standard depolarizing infidelity from single-qubit (`1Q`) and two-qubit (`2Q`) control limitations. 

For each native gate $g$ executed with error rate $\epsilon_{g}$, the participating qubits undergo fidelity degradation:
$$ F_{algo} = \prod_{g \in C} (1 - \epsilon_{g}) $$

### 2. Physical Routing Overhead

Moving quantum information across the constrained topological map degrades fidelity through network losses. We simulate routing from two distinct mechanisms:

#### A. SABRE Intra-Core SWAPs
SABRE iteratively routes logical qubits through classical adjacent physical links by inserting $SWAP$ gates. Since physical devices rarely support native SWAP instructions, they are typically decomposed into $3 \times CNOT$ gates. 
$$ F_{\text{SWAP}} = (1 - 3 \epsilon_{\text{2Q}} )^{N_{\text{swaps}}} $$
If $SWAP$ insertion causes cascading delays, this compounds the idle time in Eq 3.

#### B. Inter-Core Teleportation (HQA Boundaries)
When the Hungarian Qubit Assignment (HQA) algorithm elects to slice an entangled link across a multi-core array, it bridges the physical nodes via EPR teleportation. The fidelity of these external optical/microwave links scales exponentially with distance:
$$ F_{\text{Teleport}} = (1 - \epsilon_{\text{hop}})^{\text{distance}} $$

### 3. Idle Coherence (T1/T2 Decay)

Decoherence forces a unified "race against time". We continuously track the `busy_time` (nanoseconds) of each specific logical qubit. If a qubit is waiting for its dependency to route across the chip, or for an adjacent node to finish its operations, it accumulates `idle_time`. Using characteristic experimental timescales for energy relaxation ($T_{1}$) and phase relaxation ($T_{2}$):
$$ F_{coh} = \exp\left(-\frac{t_{\text{idle}}}{T_1}\right) \times \exp\left(-\frac{t_{\text{idle}}}{T_2}\right) $$

## Rust Implementation

You can find the core continuous-time summation loop inside `src/noise/mod.rs` $\rightarrow$ `estimate_fidelity(...)`. The results are wrapped into zero-cost `ndarray` tables and piped back up to the python frontend for heatmap rendering via `QusimResult`.
