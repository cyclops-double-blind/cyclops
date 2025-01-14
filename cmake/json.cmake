include(ProcessorCount)
ProcessorCount(JOBS)

configure_file(
  cmake/json.cmake.in
  ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/json-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/${PROJECT_NAME}/json-download")
execute_process(COMMAND "${CMAKE_COMMAND}" --build . -- -j${JOBS}
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/${PROJECT_NAME}/json-download")

find_package(nlohmann_json 3.9.1 REQUIRED
  PATHS ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/json NO_DEFAULT_PATH)
