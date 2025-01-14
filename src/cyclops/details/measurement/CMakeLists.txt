target_sources(cyclops
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/data_provider.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_provider.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_queue.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_queue.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_updater.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_updater.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/keyframe.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/keyframe.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/preintegration.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/preintegration.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/preintegration.ipp
  ${CMAKE_CURRENT_SOURCE_DIR}/type.hpp
)

cyclops_add_unittest(
  measurement_data_provider
  SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/data_provider.test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/preintegration.cpp
  ${cyclops_src_dir}/cyclops/details/config.cpp
  ${cyclops_src_dir}/cyclops/details/estimation/state/accessor_read.cpp
  ${cyclops_src_dir}/cyclops/details/estimation/state/state_block.cpp
  ${cyclops_src_dir}/cyclops/details/estimation/state/state_internal.cpp

  DEPENDS cyclops_testlib
)

cyclops_add_unittest(
  measurement_data_queue
  SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/data_queue.test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_provider.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/preintegration.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/keyframe.cpp
  ${cyclops_src_dir}/cyclops/details/config.cpp
  ${cyclops_src_dir}/cyclops/details/estimation/state/accessor_read.cpp
  ${cyclops_src_dir}/cyclops/details/estimation/state/state_block.cpp
  ${cyclops_src_dir}/cyclops/details/estimation/state/state_internal.cpp
  ${cyclops_src_dir}/cyclops/details/telemetry/keyframe.cpp
  ${cyclops_src_dir}/cyclops/details/utils/vision.cpp

  DEPENDS cyclops_testlib
)

cyclops_add_unittest(
  measurement_preintegration
  SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/preintegration.test.cpp
  ${cyclops_src_dir}/cyclops/details/config.cpp

  DEPENDS cyclops_testlib
)
