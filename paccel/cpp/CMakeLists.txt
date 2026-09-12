cmake_minimum_required(VERSION 3.20)
project(libpaccel_accelerator LANGUAGES CXX)

# The minimal, dependency-free PAccel quantization core. No onnxruntime, no
# protobuf, no BLAS — just C++20.
add_library(paccel_accelerator STATIC src/accelerator.cpp)
add_library(power::inference::accelerator ALIAS paccel_accelerator)

target_include_directories(paccel_accelerator PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_compile_features(paccel_accelerator PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(paccel_accelerator PRIVATE /W4)
else()
    target_compile_options(paccel_accelerator PRIVATE -Wall -Wextra -Wpedantic)
endif()

option(PACCEL_BUILD_SELFTEST "Build the accelerator self-test" ON)
if(PACCEL_BUILD_SELFTEST)
    enable_testing()
    add_executable(paccel_selftest test/selftest.cpp)
    target_link_libraries(paccel_selftest PRIVATE power::inference::accelerator)
    add_test(NAME paccel_selftest COMMAND paccel_selftest)
endif()
