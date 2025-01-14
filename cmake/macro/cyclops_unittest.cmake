set(cyclops_unittest_list "" CACHE INTERNAL "List of cyclops unittests")

function(cyclops_add_unittest test_name)
  if (cyclops_test)
    set(
      cyclops_unittest_list "${cyclops_unittest_list};${test_name}"
      CACHE INTERNAL "List of cyclops unittests")

    set(options "")  # empty.
    set(single_value_args "")  # empty.
    set(multi_value_args SOURCES DEPENDS)
    cmake_parse_arguments(
      cyclops_unittest "${options}" "${single_value_args}" "${multi_value_args}"
      ${ARGN})

    set(unittest_bin_target_name cyclops_unittest_${test_name})
    set(unittest_run_target_name cyclops_run_unittest_${test_name})

    add_executable(${unittest_bin_target_name} "")
    target_sources(
      ${unittest_bin_target_name}
      PRIVATE
      ${cyclops_unittest_SOURCES}
      ${cyclops_src_dir}/cyclops/details/logging.cpp
      ${cyclops_src_dir}/tests/testcases/_main.cpp
    )
    target_include_directories(
      ${unittest_bin_target_name} PRIVATE ${cyclops_src_dir})
    target_link_libraries(
      ${unittest_bin_target_name} ${cyclops_dependencies}
      ${cyclops_unittest_DEPENDS} doctest::doctest)
    target_compile_features(
      ${unittest_bin_target_name} PRIVATE cxx_std_17)

    if (cyclops_activate_sanitizers)
      target_compile_options(
        ${unittest_bin_target_name} PUBLIC
        -fno-omit-frame-pointer
        -fsanitize=address
        -fsanitize=undefined
      )
      target_link_libraries(
        ${unittest_bin_target_name} PUBLIC
        -fsanitize=address
        -fsanitize=undefined
      )
    endif()

    add_custom_target(
      ${unittest_run_target_name} DEPENDS ${unittest_bin_target_name})
    add_custom_command(
      TARGET ${unittest_run_target_name}
      COMMAND $<TARGET_FILE:${unittest_bin_target_name}>
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      POST_BUILD
    )
  endif()
endfunction()

function(cyclops_make_unittest_runner_target)
  if (cyclops_test)
    set(unittest_list_repr "")
    foreach(testname ${cyclops_unittest_list})
      string(APPEND unittest_list_repr \\n-${testname})
    endforeach(testname)

    add_custom_target(cyclops_list_unittests)
    add_custom_command(
      TARGET cyclops_list_unittests
      COMMAND echo \"[cyclops] unittests:${unittest_list_repr}\"
    )

    set(unittest_all_bin_target_names "")
    foreach(testname ${cyclops_unittest_list})
      list(
        APPEND unittest_all_bin_target_names "cyclops_unittest_${testname}")
    endforeach(testname)
    add_custom_target(
      cyclops_build_all_unittests DEPENDS ${unittest_all_bin_target_names})

    set(unittest_all_run_target_names "")
    foreach(testname ${cyclops_unittest_list})
      list(
        APPEND unittest_all_run_target_names "cyclops_run_unittest_${testname}")
    endforeach(testname)
    add_custom_target(
      cyclops_run_unittests DEPENDS
      cyclops_build_all_unittests ${unittest_all_run_target_names})
  endif()
endfunction()
