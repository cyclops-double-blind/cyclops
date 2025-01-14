include(ProcessorCount)
ProcessorCount(JOBS)

configure_file(
  cmake/eigen.cmake.in
  ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/eigen-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/${PROJECT_NAME}/eigen-download")
execute_process(COMMAND "${CMAKE_COMMAND}" --build . -- -j${JOBS}
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/${PROJECT_NAME}/eigen-download")

find_package(Eigen3 REQUIRED
  PATHS ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/eigen NO_DEFAULT_PATH)
