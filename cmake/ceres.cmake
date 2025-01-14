include(ProcessorCount)
ProcessorCount(JOBS)

if (cyclops_native_build)
  set(
    cyclops_ceres_compile_flags
    "-DEIGEN_MAX_ALIGN_BYTES=32 -DEIGEN_MAX_STATIC_ALIGN_BYTES=32 -march=native")
else()
  set(
    cyclops_ceres_compile_flags
    "-DEIGEN_MAX_ALIGN_BYTES=16 -DEIGEN_MAX_STATIC_ALIGN_BYTES=16")
endif()

configure_file(
  cmake/ceres.cmake.in
  ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/ceres-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/${PROJECT_NAME}/ceres-download")
execute_process(COMMAND "${CMAKE_COMMAND}" --build . -- -j${JOBS}
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/${PROJECT_NAME}/ceres-download")

find_package(Ceres REQUIRED
  PATHS ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/ceres NO_DEFAULT_PATH)
