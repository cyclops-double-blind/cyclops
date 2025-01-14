#pragma once

#include <string>
#include <memory>

namespace spdlog {
  struct logger;
}

namespace cyclops {
  void init_logger(int log_level = 1);
  void init_logger(std::string path, int log_level = 1);

  extern std::shared_ptr<spdlog::logger> __logger__;
}  // namespace cyclops
