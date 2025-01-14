cmake_minimum_required(VERSION 3.5)
project(cyclops)

get_directory_property(cyclops_is_subproject PARENT_DIRECTORY)

if (NOT cyclops_is_subproject)
  set(cyclops_is_standalone YES)
else()
  set(cyclops_is_standalone NO)
endif()

include(cmake/utils/options.cmake)

set(cyclops_root "${CMAKE_CURRENT_SOURCE_DIR}")
set(cyclops_src_dir "${cyclops_root}/src")
set(cyclops_dependencies
  PUBLIC Eigen3::Eigen ceres
  PRIVATE range-v3::range-v3 spdlog::spdlog)

if (cyclops_configure_bundle_dependencies)
  include(cmake/eigen.cmake)
  include(cmake/ceres.cmake)
  include(cmake/range-v3.cmake)
  include(cmake/spdlog.cmake)
else()
  find_package(Ceres REQUIRED)
  find_package(Eigen3 REQUIRED)
  find_package(range-v3 REQUIRED)
  find_package(spdlog REQUIRED)
endif()

if (cyclops_test)
  if (cyclops_configure_test_dependencies)
    include(cmake/doctest.cmake)
    include(cmake/json.cmake)
  else()
    find_package(doctest REQUIRED)
    find_package(nlohmann_json REQUIRED)
  endif()
endif()

if (cyclops_native_build)
  set(cyclops_eigen3_align_bytes
    -DEIGEN_MAX_ALIGN_BYTES=32 -DEIGEN_MAX_STATIC_ALIGN_BYTES=32)
else()
  set(cyclops_eigen3_align_bytes
    -DEIGEN_MAX_ALIGN_BYTES=16 -DEIGEN_MAX_STATIC_ALIGN_BYTES=16)
endif()

include(cmake/macro/cyclops_unittest.cmake)

add_library(cyclops STATIC "")
target_link_libraries(cyclops ${cyclops_dependencies})

add_subdirectory(src)

cyclops_make_unittest_runner_target()

if (cyclops_install)
  # Remove range-v3 from the interface link library list of the exported target.
  get_target_property(
    CYCLOPS_INTERFACE_LINK_LIB cyclops INTERFACE_LINK_LIBRARIES)
  list(REMOVE_ITEM CYCLOPS_INTERFACE_LINK_LIB "$<LINK_ONLY:range-v3::range-v3>")
  set_target_properties(
    cyclops PROPERTIES INTERFACE_LINK_LIBRARIES "${CYCLOPS_INTERFACE_LINK_LIB}")

  # Installation.
  include(CMakePackageConfigHelpers)
  configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cyclopsConfig.cmake
    INSTALL_DESTINATION lib/cmake/cyclops)
  install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/cyclopsConfig.cmake"
    DESTINATION lib/cmake/cyclops)

  install(
    TARGETS cyclops
    EXPORT cyclops-targets LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
  install(
    EXPORT cyclops-targets
    FILE cyclopsTargets.cmake NAMESPACE cyclops:: DESTINATION lib/cmake/cyclops)

  install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/src/cyclops
    DESTINATION include FILES_MATCHING PATTERN "*.hpp")

  if (cyclops_configure_bundle_dependencies)
    install(
      DIRECTORY
      ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/spdlog/include DESTINATION .)
    install(
      DIRECTORY ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/spdlog/lib DESTINATION .)

    install(
      DIRECTORY ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/ceres/include DESTINATION .)
    install(
      DIRECTORY ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/ceres/lib DESTINATION .)

    install(
      DIRECTORY ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/eigen/include DESTINATION .)
    install(
      DIRECTORY ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/eigen/share DESTINATION .)
  endif()
endif()
