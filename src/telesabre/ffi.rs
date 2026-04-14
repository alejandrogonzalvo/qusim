use std::os::raw::{c_char, c_int};

#[repr(C)]
pub struct TeleportEventT {
    pub vqubit:    c_int,
    pub from_core: c_int,
    pub to_core:   c_int,
    pub gate_idx:  c_int,
}

#[repr(C)]
pub struct SwapEventT {
    pub vqubit1:  c_int,
    pub vqubit2:  c_int,
    pub gate_idx: c_int,
}

/// Must exactly mirror the C `result_t` struct (same field order, same types).
/// count/capacity fields are size_t in C → usize in Rust.
#[repr(C)]
pub struct ResultT {
    pub num_teledata:  c_int,
    pub num_telegate:  c_int,
    pub num_swaps:     c_int,
    pub depth:         c_int,
    pub num_deadlocks: c_int,
    pub success:       bool,

    pub initial_virt_to_core:     *mut c_int,
    pub num_vqubits:              usize,
    pub teleport_events:          *mut TeleportEventT,
    pub num_teleport_events:      usize,
    pub teleport_events_capacity: usize,
    pub swap_events:              *mut SwapEventT,
    pub num_swap_events:          usize,
    pub swap_events_capacity:     usize,
}

#[repr(C)] pub struct ConfigT  { _private: [u8; 0] }
#[repr(C)] pub struct DeviceT  { _private: [u8; 0] }
#[repr(C)] pub struct CircuitT { _private: [u8; 0] }

extern "C" {
    pub fn config_from_json(filename: *const c_char) -> *mut ConfigT;
    pub fn config_set_parameter(config: *mut ConfigT, key: *const c_char, value: *const c_char);
    pub fn config_free(config: *mut ConfigT);

    pub fn device_from_json(filename: *const c_char) -> *mut DeviceT;
    pub fn device_free(device: *mut DeviceT);

    pub fn circuit_from_qasm(filename: *const c_char) -> *mut CircuitT;
    pub fn circuit_free(circuit: *mut CircuitT);

    pub fn telesabre_run(
        config:  *mut ConfigT,
        device:  *mut DeviceT,
        circuit: *mut CircuitT,
    ) -> ResultT;

    /// Frees heap memory inside `r` (does NOT free `r` itself).
    pub fn result_free(r: *mut ResultT);
}
