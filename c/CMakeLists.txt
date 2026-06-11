cmake_minimum_required(VERSION 3.18)
project(numleanopencl_c LANGUAGES C)

find_package(OpenCL REQUIRED)

add_library(numleanopencl_opencl STATIC
    opencl_lean_context.c)

target_link_libraries(numleanopencl_opencl PUBLIC OpenCL::OpenCL)

add_executable(opencl_smoke opencl_smoke.c)
target_link_libraries(opencl_smoke PRIVATE OpenCL::OpenCL m)
