use serde::{Serialize, Deserialize};

/// Defines the hardware-level role of the qubit within a QCore.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QubitRole {
    /// Used for executing the actual gates of the quantum algorithm.
    Computation,
    /// Dedicated to storing halves of EPR pairs for quantum links.
    Communication,
    /// Temporarily holds arriving quantum states during teleportation.
    Buffer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qubit {
    /// Global unique identifier for the physical qubit.
    pub id: usize,
    /// The ID of the QCore this qubit resides in.
    pub core_id: usize,
    /// The specific role this qubit plays in the architecture.
    pub role: QubitRole,
    
    // --- Simulation Tracking Fields ---
    
    /// Tracks the accumulated operational fidelity (starts at 1.0).
    pub operational_fidelity: f64,
    /// Tracks the accumulated coherence fidelity due to latency (starts at 1.0).
    pub coherence_fidelity: f64,
    /// The exact simulator timestamp when this qubit last finished an operation.
    /// Crucial for calculating how long the qubit sat idle (latency).
    pub last_active_time: f64,
    
    // --- State Management ---
    
    /// If true, the qubit is currently executing a gate or waiting for a network message.
    pub is_busy: bool,
    /// If this is a Communication qubit, this tracks the ID of the distant qubit it is entangled with.
    pub entangled_with: Option<usize>,
}

impl Qubit {
    pub fn new(id: usize, core_id: usize, role: QubitRole) -> Self {
        Self {
            id,
            core_id,
            role,
            operational_fidelity: 1.0,
            coherence_fidelity: 1.0,
            last_active_time: 0.0,
            is_busy: false,
            entangled_with: None,
        }
    }

    /// Calculates overall fidelity based on F = C(t) * F_op
    pub fn overall_fidelity(&self) -> f64 {
        self.operational_fidelity * self.coherence_fidelity
    }

    /// Applies operational noise from a quantum gate execution.
    pub fn apply_gate(&mut self, error_rate: f64, current_time: f64, gate_duration: f64) {
        // First, apply any decoherence that happened while the qubit was waiting for this gate
        self.apply_decoherence(current_time);

        // Multiply the current fidelity by the probability of the gate succeeding
        self.operational_fidelity *= 1.0 - error_rate;

        // Advance the qubit's local clock
        self.last_active_time = current_time + gate_duration;
    }

    /// Applies decoherence based on how long the qubit sat idle since its last operation.
    pub fn apply_decoherence(&mut self, current_time: f64) {
        let idle_time = current_time - self.last_active_time;
        
        if idle_time > 0.0 {
            // These should ideally be passed in or configured globally, 
            // set to T1=1.2ms and T2=1.16ms based on the paper's parameters.
            let t1 = 1_200_000.0; // in nanoseconds
            let t2 = 1_160_000.0; // in nanoseconds

            // C(t) = e^(-t/T1) * (0.5 * e^(-t/T2) + 0.5)
            let relaxation = (-idle_time / t1).exp();
            let dephasing = 0.5 * (-idle_time / t2).exp() + 0.5;
            
            let decoherence_drop = relaxation * dephasing;
            self.coherence_fidelity *= decoherence_drop;
            
            // Catch up the clock
            self.last_active_time = current_time;
        }
    }
}