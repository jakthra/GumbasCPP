add_library (GumbasCPP GumbasCPP.cpp GumbasCPP.h)
add_executable (main "main.cpp")
target_link_libraries (main GumbasCPP)
set(CMAKE_SUPPRESS_REGENERATION true)



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
