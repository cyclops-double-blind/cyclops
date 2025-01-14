include(ProcessorCount)
ProcessorCount(JOBS)

configure_file(
  cmake/doctest.cmake.in
  ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/doctest-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/${PROJECT_NAME}/doctest-download")
execute_process(COMMAND "${CMAKE_COMMAND}" --build . -- -j${JOBS}
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/${PROJECT_NAME}/doctest-download")

find_package(doctest REQUIRED
  PATHS ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/doctest NO_DEFAULT_PATH)
