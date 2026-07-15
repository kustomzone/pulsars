fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return; // kernels are CUDA/Linux; other hosts get an empty crate
    }
    println!("cargo:rerun-if-changed=cuda");
    // One fatbin for every NVIDIA generation the kernels can serve. The
    // floor is dp4a = sm_61 (Pascal / GTX 10-series); nothing newer is
    // required (no tensor cores, no async-copy, static <=48KB shared).
    //
    //   sm_61  SASS  GTX 10-series           + compute_61 PTX: JIT floor
    //   sm_75  SASS  GTX 16 / RTX 20-series    for anything unlisted
    //   sm_86  SASS  RTX 30-series             (sm_70 Volta, sm_80 A100,
    //   sm_89  SASS  RTX 40-series              Hopper, ...)
    //   compute_89 PTX: JIT for sm_90+ (Blackwell RTX 50 etc.) with the
    //   newest ISA the toolkit knows, instead of the sm_61 floor.
    //
    // PULSAR_CUDA_ARCH overrides (e.g. "89" for a fast dev build, or
    // "89,120" once the toolkit codegens Blackwell SASS natively).
    let archs = std::env::var("PULSAR_CUDA_ARCH").unwrap_or_else(|_| "61,75,86,89".into());
    let mut build = cc::Build::new();
    build.cuda(true).flag("-O3").flag("--use_fast_math");
    let list: Vec<&str> = archs.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    for (i, a) in list.iter().enumerate() {
        let first = i == 0;
        let last = i + 1 == list.len();
        // lowest arch also embeds its PTX (universal JIT floor); highest
        // embeds its PTX too (best ISA for future GPUs); middles are SASS-only
        let code = if first || last {
            format!("arch=compute_{a},code=[sm_{a},compute_{a}]")
        } else {
            format!("arch=compute_{a},code=sm_{a}")
        };
        build.flag("-gencode").flag(&code);
    }
    build.file("cuda/pulsar_kernels.cu").compile("pulsar_kernels");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
}
