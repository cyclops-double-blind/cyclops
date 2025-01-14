include(ProcessorCount)
ProcessorCount(JOBS)

configure_file(
  cmake/range-v3.cmake.in
  ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/range-v3-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/${PROJECT_NAME}/range-v3-download")
execute_process(COMMAND "${CMAKE_COMMAND}" --build . -- -j${JOBS}
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/${PROJECT_NAME}/range-v3-download")

find_package(range-v3 REQUIRED
  PATHS ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/range-v3
  NO_DEFAULT_PATH)
