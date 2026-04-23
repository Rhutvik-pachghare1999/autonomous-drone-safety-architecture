cmake_minimum_required(VERSION 3.18)
project(aisp_hocbf VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# pybind11 from system package
find_package(pybind11 REQUIRED PATHS /usr/lib/python3/dist-packages/pybind11/share/cmake/pybind11)

# Python bindings target
pybind11_add_module(hocbf src/control/hocbf.cpp)

target_compile_definitions(hocbf PRIVATE BUILD_PYTHON_BINDINGS)

target_compile_options(hocbf PRIVATE
    -O3 -march=native
    -Wall -Wextra
)

# Install .so next to the source for direct import
install(TARGETS hocbf LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/src/control)
