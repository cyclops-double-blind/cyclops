target_sources(cyclops
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/debug.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/debug.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/debug.ipp
  ${CMAKE_CURRENT_SOURCE_DIR}/math.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/math.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/qcqp1.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/qcqp1.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/type.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/vision.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/vision.hpp
)

cyclops_add_unittest(
  utils_vision
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/vision.test.cpp
  DEPENDS cyclops_testlib
)

if (cyclops_test AND cyclops_test_with_boost)
  find_package(Boost REQUIRED)
  cyclops_add_unittest(
    utils_math
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/math.test.cpp
    DEPENDS cyclops_testlib Boost::boost
  )
endif()
