include(CMakeDependentOption)

option(cyclops_enable_compiler_warnings "Enable compiler warnings" ON)
option(cyclops_test "Build unit test" OFF)
option(cyclops_test_dump "Dump test results for the visualization" OFF)

option(
  cyclops_native_build
  "Activate native build optimization (-march=native)" OFF)  # FIXME

option(
  cyclops_activate_sanitizers
  "Activate sanitizers (asan/ubsan)" OFF)

option(
  cyclops_configure_bundle_dependencies
  "Configure bundle dependencies (eigen/ceres/range-v3/spdlog)" ON)

option(
  cyclops_configure_test_dependencies
  "Configure test dependencies (doctest/nlohmann-json)" ON)

option(
  cyclops_test_with_boost
  "Allow testcases that depend on boost" OFF)

cmake_dependent_option(
  cyclops_install
  "Generate an install target for cyclops" ON "${cyclops_is_standalone}" OFF)
