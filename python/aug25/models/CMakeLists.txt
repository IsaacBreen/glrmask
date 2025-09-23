cmake_minimum_required(VERSION 3.14)
project(gss_cpp_modules)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11. If not installed, download it.
find_package(pybind11)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found. Downloading from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1 # A recent stable version
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# For precompute3_engine
pybind11_add_module(precompute3_engine precompute3_engine.cpp)

# For LeveledGSS wrapper
pybind11_add_module(leveled_gss_cpp leveled_gss_py.cpp)