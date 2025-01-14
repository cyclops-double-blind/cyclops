include(ProcessorCount)
ProcessorCount(JOBS)

configure_file(
  cmake/spdlog.cmake.in
  ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/spdlog-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/${PROJECT_NAME}/spdlog-download")
execute_process(COMMAND "${CMAKE_COMMAND}" --build . -- -j${JOBS}
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/${PROJECT_NAME}/spdlog-download")

find_package(spdlog REQUIRED
  PATHS ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/spdlog NO_DEFAULT_PATH)
