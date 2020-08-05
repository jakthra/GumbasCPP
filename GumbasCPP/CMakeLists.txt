# CMakeList.txt : CMake project for GumbasCPP, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("GumbasCPP")

list(APPEND CMAKE_PREFIX_PATH C:\\libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory (src)


# TODO: Add tests and install targets if needed.

