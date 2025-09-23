cmake_minimum_required(VERSION 3.14)
project(gss_cpp_modules)

find_package(pybind11 REQUIRED)

# For precompute3_engine
pybind11_add_module(precompute3_engine precompute3_engine.cpp)

# For LeveledGSS wrapper
pybind11_add_module(leveled_gss_cpp leveled_gss_py.cpp)
