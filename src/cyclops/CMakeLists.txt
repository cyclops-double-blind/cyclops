target_compile_features(cyclops PUBLIC cxx_std_17)
target_include_directories(
  cyclops PUBLIC
  $<BUILD_INTERFACE:${cyclops_src_dir}>
  $<INSTALL_INTERFACE:include>)

target_compile_definitions(cyclops PUBLIC ${cyclops_eigen3_align_bytes})
if (cyclops_native_build)
  # This should be set as PUBLIC to ensure the ABI compatibility of the
  # Eigen-related objects.
  target_compile_options(cyclops PUBLIC -march=native)
endif()

if (cyclops_enable_compiler_warnings)
  target_compile_options(
    cyclops PRIVATE
    -Wuninitialized -Wmaybe-uninitialized -Winit-self
    -Wunused -Wunused-value -Wno-unused-variable -Wunused-const-variable=1
    -Wunused-function -Wunused-label
    -Wmissing-declarations -Wmissing-field-initializers
    -Wimplicit-fallthrough
    -Wuseless-cast
    -Wbool-compare -Wenum-compare
  )
endif()

if (cyclops_activate_sanitizers)
  target_compile_options(
    cyclops PUBLIC
    -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined)
  target_link_libraries(cyclops PUBLIC -fsanitize=address -fsanitize=undefined)
endif()

target_sources(cyclops
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/cyclops.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cyclops.hpp
)
add_subdirectory(details)
