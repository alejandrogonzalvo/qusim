fn main() {
    let src_dir = "csrc/telesabre";

    let sources = [
        "circuit.c",
        "config.c",
        "device.c",
        "graph.c",
        "heap.c",
        "json.c",
        "layout.c",
        "report.c",
        "telesabre.c",
        "utils.c",
    ];

    let mut build = cc::Build::new();
    build.include(src_dir);
    build.flag("-w"); // suppress warnings from vendored C code
    build.flag("-O2");

    for src in &sources {
        build.file(format!("{}/{}", src_dir, src));
        println!("cargo:rerun-if-changed={}/{}", src_dir, src);
    }

    build.compile("telesabre_c");

    // telesabre.c uses sqrt/fabs from libm
    println!("cargo:rustc-link-lib=m");
}
