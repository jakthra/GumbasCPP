# CMakeList.txt : CMake project for GumbasCPP, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("GumbasCPP")

list(APPEND CMAKE_PREFIX_PATH C:\\libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add source to this project's executable.
add_executable (GumbasCPP "GumbasCPP.cpp" "GumbasCPP.h")

# TODO: Add tests and install targets if needed.



target_link_libraries(GumbasCPP "${TORCH_LIBRARIES}")
set_property(TARGET GumbasCPP PROPERTY CXX_STANDARD 14)

add_custom_command(TARGET GumbasCPP POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                       ${CMAKE_SOURCE_DIR}/Resources/ $<TARGET_FILE_DIR:GumbasCPP>)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET GumbasCPP
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:GumbasCPP>)
endif (MSVC)
