cmake_minimum_required(VERSION 3.18)
project(aisp_hocbf VERSION 2.0.0 LANGUAGES CXX C)

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

# Watchdog C test — exercises the real src/rt/watchdog.c (single source of truth)
enable_testing()
add_executable(watchdog_test tests/watchdog_test.c src/rt/watchdog.c)
target_include_directories(watchdog_test PRIVATE src/rt)
target_link_libraries(watchdog_test m rt)
add_test(NAME watchdog_test COMMAND watchdog_test)

# SIL_REPLAY runner — software-in-the-loop replay for CI/verification
# Builds safety_filter.c with SIL_REPLAY mode (reads /dev/shm/aisp_vla_cmd,
# runs watchdog + HOCBF, logs CSV). Used by sim/sil_harness.py.
add_executable(sil_runner src/rt/safety_filter.c src/rt/watchdog.c)
target_compile_definitions(sil_runner PRIVATE SIL_REPLAY NO_ONNX)
target_include_directories(sil_runner PRIVATE src/rt)
target_link_libraries(sil_runner m rt)
# Note: actual SIL integration is tested via Python harness (sim/sil_harness.py)
# which creates the shared memory. CTest only verifies the binary builds.

# Production RT safety filter — periodic 1 kHz deadline-driven loop with
# SCHED_FIFO, mlockall, CPU affinity, and deadline miss detection.
# Requires CAP_SYS_NICE and CAP_IPC_LOCK to run.
add_executable(safety_filter src/rt/safety_filter.c src/rt/watchdog.c)
target_compile_definitions(safety_filter PRIVATE NO_ONNX)
target_include_directories(safety_filter PRIVATE src/rt)
target_link_libraries(safety_filter m rt)
# Note: NO_ONNX — ONNX optional at runtime via --sitl flag
