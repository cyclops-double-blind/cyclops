target_sources(cyclops
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/cost.gaussian_prior.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cost.gaussian_prior.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cost.imu_bias_prior.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cost.imu_bias_walk.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cost.imu_preintegration.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cost.imu_preintegration.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cost.landmark.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/manifold.se3.hpp
)

cyclops_add_unittest(
  estimator_prior_ceres_cost
  SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/cost.gaussian_prior.test.cpp
  ${cyclops_src_dir}/cyclops/details/config.cpp
  ${cyclops_src_dir}/cyclops/details/estimation/graph/node.cpp
  ${cyclops_src_dir}/cyclops/details/estimation/state/state_block.cpp

  DEPENDS cyclops_testlib ceres
)

cyclops_add_unittest(
  estimator_imu_ceres_cost
  SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/cost.imu_preintegration.test.cpp
  ${cyclops_src_dir}/cyclops/details/config.cpp
  ${cyclops_src_dir}/cyclops/details/measurement/preintegration.cpp

  DEPENDS cyclops_testlib
)

cyclops_add_unittest(
  estimator_landmark_ceres_cost
  SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/cost.landmark.test.cpp
  ${cyclops_src_dir}/cyclops/details/config.cpp

  DEPENDS cyclops_testlib
)
