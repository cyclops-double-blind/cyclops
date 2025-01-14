add_subdirectory(estimation)
add_subdirectory(initializer)
add_subdirectory(measurement)
add_subdirectory(telemetry)
add_subdirectory(utils)

target_sources(cyclops
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/config.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/config.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/logging.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/logging.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/type.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/type.hpp
)
