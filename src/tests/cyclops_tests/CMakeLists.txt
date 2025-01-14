target_include_directories(
  cyclops_testlib PUBLIC ${cyclops_src_dir} ${cyclops_src_dir}/tests)

target_compile_features(cyclops_testlib PUBLIC cxx_std_17)
target_compile_definitions(cyclops_testlib PUBLIC ${cyclops_eigen3_align_bytes})

target_sources(
  cyclops_testlib
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/default.cpp
  ${CMAKE_CURRENT_LIST_DIR}/default.hpp
  ${CMAKE_CURRENT_LIST_DIR}/random.cpp
  ${CMAKE_CURRENT_LIST_DIR}/random.hpp
  ${CMAKE_CURRENT_LIST_DIR}/random.ipp
  ${CMAKE_CURRENT_LIST_DIR}/range.ipp
  ${CMAKE_CURRENT_LIST_DIR}/serialize.cpp
  ${CMAKE_CURRENT_LIST_DIR}/serialize.hpp
  ${CMAKE_CURRENT_LIST_DIR}/signal.cpp
  ${CMAKE_CURRENT_LIST_DIR}/signal.hpp
  ${CMAKE_CURRENT_LIST_DIR}/signal.ipp
  ${CMAKE_CURRENT_LIST_DIR}/state.hpp
  ${CMAKE_CURRENT_LIST_DIR}/trajectory.cpp
  ${CMAKE_CURRENT_LIST_DIR}/trajectory.hpp)
add_subdirectory(data)
add_subdirectory(mockups)
