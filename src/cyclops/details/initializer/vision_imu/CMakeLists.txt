target_sources(cyclops
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/acceptance.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/acceptance.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/camera_motion_prior.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/camera_motion_prior.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rotation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rotation.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/translation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/translation.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/translation_analysis.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/translation_analysis.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/translation_cache.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/translation_cache.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/translation_evaluation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/translation_evaluation.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/translation_refinement.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/translation_refinement.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/translation_sample.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/translation_sample.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/uncertainty.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/uncertainty.hpp
)

cyclops_add_unittest(
  initializer_camera_motion_prior
  SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/camera_motion_prior.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/camera_motion_prior.test.cpp
)

cyclops_add_unittest(
  initializer_rotation_matching
  SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/rotation.test.cpp
  ${cyclops_src_dir}/cyclops/details/measurement/preintegration.cpp
  ${cyclops_src_dir}/cyclops/details/utils/math.cpp
)

cyclops_add_unittest(
  initializer_translation
  SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/translation.test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/acceptance.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/translation_analysis.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/translation_cache.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/translation_evaluation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/translation_refinement.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/translation_sample.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/uncertainty.cpp
  ${cyclops_src_dir}/cyclops/details/utils/debug.cpp
  ${cyclops_src_dir}/cyclops/details/utils/math.cpp
  ${cyclops_src_dir}/cyclops/details/utils/qcqp1.cpp
  ${cyclops_src_dir}/cyclops/details/telemetry/initializer.cpp

  DEPENDS cyclops_testlib
)

cyclops_add_unittest(
  initializer_translation_analysis
  SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/translation_analysis.test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/translation_analysis.cpp
  ${cyclops_src_dir}/cyclops/details/measurement/preintegration.cpp
)

cyclops_add_unittest(
  initializer_translation_primal
  SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/translation_evaluation.test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/translation_evaluation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/translation_cache.cpp
  ${cyclops_src_dir}/cyclops/details/utils/qcqp1.cpp
)
