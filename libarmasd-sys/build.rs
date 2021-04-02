
extern crate autotools;

use std::path::Path;
use std::process::Command;

pub fn main() {

    if !Path::new("armas/.git").exists() {
        let _ = Command::new("git")
            .args(&["submodule", "update", "--init", "armas"])
            .status();
    }

    if !Path::new("armas/configure").exists() {
        let _ = Command::new("sh")
            .current_dir("./armas")
            .arg("./bootstrap.sh")
            .status();
    }

    let mut config = autotools::Config::new("armas");

    config.enable("float64", None)
        .enable("notypenames", None)
        .disable("ext-precision", None)
        .disable("accelerators", None)
        .disable("compat", None)
        .disable("sparse", None)
        .cflag("-O3");

    config.make_target("all").build();
    let dst = config.make_target("install").build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=armasd");
}
