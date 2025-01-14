add_library(cyclops_testlib "")
add_executable(cyclops_test "")

target_link_libraries(
  cyclops_testlib
  PUBLIC Eigen3::Eigen range-v3::range-v3 nlohmann_json::nlohmann_json)
target_link_libraries(
  cyclops_test PRIVATE doctest::doctest ceres cyclops_testlib
  -Wl,--whole-archive cyclops -Wl,--no-whole-archive)  # XXX

add_subdirectory(cyclops_tests)
add_subdirectory(testcases)

if (cyclops_test_dump)
  target_compile_definitions(
    cyclops_test PRIVATE CYCLOPS_TEST_DUMP_RESULTS)
endif()

add_custom_target(cyclops_run_test DEPENDS cyclops_test)
add_custom_command(
  TARGET cyclops_run_test
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/cyclops_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  POST_BUILD)
