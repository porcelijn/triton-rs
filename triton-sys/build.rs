use std::env;
use std::path::PathBuf;

fn main() {

    let target_path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("src/bindings.rs");

    if target_path.exists() {
        println!("cargo:warning=bindings.rs already exist, jump generate step!");
        return;
    }

    let bindings = bindgen::Builder::default()
        .clang_arg("-Ideps/core/include")
        .clang_arg("-I/opt/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/libc/usr/include/")
        .clang_arg("-xc++")
        .clang_arg("--target=aarch64-unknown-linux-gnu")
        .header("deps/core/include/triton/core/tritonbackend.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    bindings.write_to_file(target_path).expect("Write failed");
}
