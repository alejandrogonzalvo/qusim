mod ffi;

use std::ffi::CString;

pub struct TeleSabreResult {
    pub num_teledata: i32,
    pub num_telegate: i32,
    pub num_swaps: i32,
    pub depth: i32,
    pub num_deadlocks: i32,
    pub success: bool,
    /// Core index for each virtual qubit at the start of routing.
    pub initial_virt_to_core: Vec<i32>,
    /// (vqubit, from_core, to_core, gate_idx) — one per teleportation.
    pub teleport_events: Vec<(i32, i32, i32, i32)>,
    /// (vq1, vq2, gate_idx) — one per SWAP (free-slot SWAPs excluded).
    pub swap_events: Vec<(i32, i32, i32)>,
}

/// Safe RAII wrapper around the three C-owned pointers required by `telesabre_run`.
///
/// `TeleSabre::new` parses all three input files via the C library and owns
/// the resulting heap-allocated structs.  `Drop` frees them via the matching
/// C destructors, so there is no memory leak.
///
/// # Thread safety
/// The underlying C library calls `srand`/`rand` (global state) and uses
/// `printf` extensively.  Do not share a `TeleSabre` across threads and run
/// any tests that call `run()` with `--test-threads=1`.
pub struct TeleSabre {
    config: *mut ffi::ConfigT,
    device: *mut ffi::DeviceT,
    circuit: *mut ffi::CircuitT,
}

impl TeleSabre {
    /// Load config, device and circuit from the given file paths and prepare
    /// the algorithm state.  Panics if any file cannot be parsed.
    pub fn new(config_path: &str, device_path: &str, circuit_path: &str) -> Self {
        let config_cstr = CString::new(config_path).expect("config path contains interior NUL");
        let device_cstr = CString::new(device_path).expect("device path contains interior NUL");
        let circuit_cstr =
            CString::new(circuit_path).expect("circuit path contains interior NUL");

        // Safety: all three C functions allocate and return a heap pointer that
        // we own.  We check for NULL and panic rather than proceed with an
        // invalid pointer.
        let (config, device, circuit) = unsafe {
            let config = ffi::config_from_json(config_cstr.as_ptr());
            assert!(!config.is_null(), "config_from_json returned NULL for {config_path}");

            let device = ffi::device_from_json(device_cstr.as_ptr());
            assert!(!device.is_null(), "device_from_json returned NULL for {device_path}");

            let circuit = ffi::circuit_from_qasm(circuit_cstr.as_ptr());
            assert!(!circuit.is_null(), "circuit_from_qasm returned NULL for {circuit_path}");

            (config, device, circuit)
        };

        TeleSabre { config, device, circuit }
    }

    /// Override a single config parameter by name (same key strings accepted
    /// by the CLI's `--key value` syntax).
    pub fn set_param(&mut self, key: &str, value: &str) {
        let key_cstr = CString::new(key).expect("key contains interior NUL");
        let value_cstr = CString::new(value).expect("value contains interior NUL");
        // Safety: self.config is a valid, owned pointer.
        unsafe {
            ffi::config_set_parameter(self.config, key_cstr.as_ptr(), value_cstr.as_ptr());
        }
    }

    /// Execute the TeleSABRE algorithm and return the routing result.
    ///
    /// This calls `telesabre_run` which internally calls `srand(config->seed)`.
    /// The config, device and circuit pointers remain valid afterwards; you
    /// may call `run` again if needed.
    pub fn run(&mut self) -> TeleSabreResult {
        // Safety: all three pointers are valid and owned by self.
        let mut r = unsafe { ffi::telesabre_run(self.config, self.device, self.circuit) };

        let num_vqubits = r.num_vqubits;

        let initial_virt_to_core: Vec<i32> = if r.initial_virt_to_core.is_null() || num_vqubits == 0 {
            vec![]
        } else {
            // Safety: pointer is non-null, length is > 0, and the slice is
            // valid until result_free (called below). .to_vec()/.collect()
            // copies data into Rust-owned heap before C memory is released.
            unsafe { std::slice::from_raw_parts(r.initial_virt_to_core, num_vqubits).to_vec() }
        };

        let teleport_events: Vec<(i32, i32, i32, i32)> =
            if r.teleport_events.is_null() || r.num_teleport_events == 0 {
                vec![]
            } else {
                // Safety: pointer is non-null, length is > 0, and the slice is
                // valid until result_free (called below). .to_vec()/.collect()
                // copies data into Rust-owned heap before C memory is released.
                unsafe {
                    std::slice::from_raw_parts(r.teleport_events, r.num_teleport_events)
                        .iter()
                        .map(|e| (e.vqubit, e.from_core, e.to_core, e.gate_idx))
                        .collect()
                }
            };

        let swap_events: Vec<(i32, i32, i32)> =
            if r.swap_events.is_null() || r.num_swap_events == 0 {
                vec![]
            } else {
                // Safety: pointer is non-null, length is > 0, and the slice is
                // valid until result_free (called below). .to_vec()/.collect()
                // copies data into Rust-owned heap before C memory is released.
                unsafe {
                    std::slice::from_raw_parts(r.swap_events, r.num_swap_events)
                        .iter()
                        .map(|e| (e.vqubit1, e.vqubit2, e.gate_idx))
                        .collect()
                }
            };

        // Capture all scalar fields before result_free, which may zero the struct.
        let num_teledata  = r.num_teledata;
        let num_telegate  = r.num_telegate;
        let num_swaps     = r.num_swaps;
        let depth         = r.depth;
        let num_deadlocks = r.num_deadlocks;
        let success       = r.success;

        // Free C-owned heap memory now that we've copied everything.
        unsafe { ffi::result_free(&mut r as *mut ffi::ResultT) };

        TeleSabreResult {
            num_teledata,
            num_telegate,
            num_swaps,
            depth,
            num_deadlocks,
            success,
            initial_virt_to_core,
            teleport_events,
            swap_events,
        }
    }
}

impl Drop for TeleSabre {
    fn drop(&mut self) {
        // Safety: we are the unique owner; pointers were set in new() and
        // can only be null if new() panicked before reaching the assignment,
        // in which case Drop is never reached.
        unsafe {
            ffi::circuit_free(self.circuit);
            ffi::device_free(self.device);
            ffi::config_free(self.config);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: our FFI wrapper must produce the same results as
    /// running the reference C binary directly.
    ///
    /// Golden values captured from:
    ///   cd telesabre && ./telesabre configs/default.json \
    ///     devices/A_grid_2_2_3_3.json \
    ///     circuits/qasm_25/qft_nativegates_ibm_qiskit_opt3_25.qasm
    ///
    /// Results are seed-independent for this circuit/device (verified with
    /// seeds 1, 2, 3, 42, 100, 999 — all produce identical counts).
    #[test]
    fn telesabre_qft25_matches_c_reference() {
        let mut ts = TeleSabre::new(
            "tests/fixtures/telesabre/configs/default.json",
            "tests/fixtures/telesabre/devices/A_grid_2_2_3_3.json",
            "tests/fixtures/telesabre/circuits/qasm_25/qft_nativegates_ibm_qiskit_opt3_25.qasm",
        );
        let result = ts.run();

        assert!(result.success, "TeleSABRE must complete successfully");
        assert_eq!(result.num_teledata, 156, "teledata ops must match C reference");
        assert_eq!(result.num_telegate, 24, "telegate ops must match C reference");
        assert_eq!(result.num_swaps, 494, "swap ops must match C reference");
    }

    /// Second golden vector: GHZ-25, same device.
    #[test]
    fn telesabre_ghz25_matches_c_reference() {
        let mut ts = TeleSabre::new(
            "tests/fixtures/telesabre/configs/default.json",
            "tests/fixtures/telesabre/devices/A_grid_2_2_3_3.json",
            "tests/fixtures/telesabre/circuits/qasm_25/ghz_nativegates_ibm_qiskit_opt3_25.qasm",
        );
        let result = ts.run();

        assert!(result.success);
        assert_eq!(result.num_teledata, 32);
        assert_eq!(result.num_telegate, 0);
        assert_eq!(result.num_swaps, 81);
    }
}
