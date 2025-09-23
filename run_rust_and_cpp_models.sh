#!/usr/bin/env python3
import os
import sys
import subprocess
import urllib.request
import tarfile
import sysconfig
from pathlib import Path

# This script builds the Boost.ICL-backed C++ extension and runs the benchmarks
# comparing the Rust baseline and the C++-accelerated Python model.
#
# Usage:
#   python3 run_rust_and_cpp_models.sh
#
# Requirements:
#   - A C++17 compiler (clang++ or g++)
#   - Python with pip

def main():
    """Main function to setup, build, and run benchmarks."""
    project_root = Path(__file__).parent.resolve()
    os.chdir(project_root)

    # 1) Ensure we are in the repo root and Python folder exists
    if not (project_root / "python").is_dir():
        print(
            "Error: expected 'python' directory at repo root. "
            "Run this script from the repo root.",
            file=sys.stderr
        )
        sys.exit(1)

    # 2) Setup: download Boost headers locally and install pybind11
    boost_version = "1.83.0"
    boost_ver_underscores = "1_83_0"
    build_dir = project_root / ".build"
    boost_dir = build_dir / f"boost_{boost_ver_underscores}"
    build_dir.mkdir(exist_ok=True)

    if not boost_dir.is_dir():
        boost_tgz = build_dir / f"boost_{boost_ver_underscores}.tar.gz"
        print(f"Downloading Boost {boost_version} headers...")
        boost_url = (
            f"https://archives.boost.io/release/{boost_version}/source/"
            f"boost_{boost_ver_underscores}.tar.gz"
        )
        urllib.request.urlretrieve(boost_url, boost_tgz)

        print("Extracting Boost...")
        with tarfile.open(boost_tgz, "r:gz") as tar:
            tar.extractall(path=build_dir)
        boost_tgz.unlink()  # remove tarball after extraction

    print("Installing pybind11 (local user)...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--user", "--quiet", "pybind11"],
        check=True
    )

    # 3) Build the C++ extension in place
    print("Building C++ extension (Boost.ICL RangeSet) ...")
    python_dir = project_root / "python"
    os.chdir(python_dir)

    # Compute extension suffix and includes via pybind11
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    pybind_includes_str = subprocess.check_output(
        [sys.executable, "-m", "pybind11", "--includes"],
        encoding='utf-8'
    ).strip()
    pybind_includes = pybind_includes_str.split()

    cxx = os.environ.get("CXX", "c++")

    source_file = "aug25/models/icl_rangeset.cpp"
    output_file = f"aug25/models/icl_rangeset{ext_suffix}"

    compile_command = [
        cxx,
        "-O3",
        "-std=c++17",
        "-shared",
        "-fPIC",
        *pybind_includes,
        f"-I{boost_dir}",
        source_file,
        "-o",
        output_file,
    ]

    subprocess.run(compile_command, check=True, text=True)

    print(f"Build complete: {python_dir / output_file}")

    # 4) Run benchmarks comparing Rust baseline vs C++-accelerated model
    print("\nRunning benchmarks...")
    # Assumes run_benchmarks.sh is in the 'python' directory and model paths
    # should be relative to it.
    subprocess.run(
        [
            "bash",
            "run_benchmarks.sh",
            "aug25/models/rust_model.py",
            "aug25/models/precompute3_model_cpp.py"
        ],
        check=True
    )

if __name__ == "__main__":
    main()
