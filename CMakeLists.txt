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
