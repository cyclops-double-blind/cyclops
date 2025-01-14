target_sources(cyclops
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/bundle_adjustment.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bundle_adjustment.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bundle_adjustment_factors.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bundle_adjustment_factors.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bundle_adjustment_states.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bundle_adjustment_states.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/epipolar.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/epipolar.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/epipolar_refine.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/epipolar_refine.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/epnp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/epnp.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/homography.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/homography.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/homography_refinement.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/homography_refinement.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hypothesis.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hypothesis.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/multiview.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/multiview.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/triangulation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/triangulation.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/twoview.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/twoview.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/twoview_selection.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/twoview_selection.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/type.hpp
)

cyclops_add_unittest(
  initializer_vision_epipolar
  SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/epipolar.test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/epipolar_refine.cpp
  ${cyclops_src_dir}/cyclops/details/utils/math.cpp

  DEPENDS cyclops_testlib
)

cyclops_add_unittest(
  initializer_vision_epipolar_refinement
  SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/epipolar_refine.test.cpp
  ${cyclops_src_dir}/cyclops/details/utils/math.cpp
)

cyclops_add_unittest(
  initializer_vision_homography
  SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/homography.test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/homography_refinement.cpp
  ${cyclops_src_dir}/cyclops/details/utils/debug.cpp
  ${cyclops_src_dir}/cyclops/details/utils/math.cpp

  DEPENDS cyclops_testlib
)

cyclops_add_unittest(
  initializer_vision_epnp
  SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/epnp.test.cpp

  DEPENDS cyclops_testlib
)

cyclops_add_unittest(
  initializer_bundle_adjustment_factors
  SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/bundle_adjustment_factors.test.cpp
)

cyclops_add_unittest(
  initializer_bundle_adjustment
  SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/bundle_adjustment.test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bundle_adjustment_states.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bundle_adjustment_factors.cpp
  ${cyclops_src_dir}/cyclops/details/utils/math.cpp
  ${cyclops_src_dir}/cyclops/details/config.cpp
  ${cyclops_src_dir}/cyclops/details/type.cpp

  DEPENDS cyclops_testlib
)
