#include "cyclops/details/logging.hpp"

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <cstdlib>

namespace cyclops {
  std::shared_ptr<spdlog::logger> __logger__ = spdlog::default_logger();

  static std::string log_level_name(spdlog::level::level_enum const level) {
    switch (level) {
    case spdlog::level::trace:
      return "TRACE";
    case spdlog::level::debug:
      return "DEBUG";
    case spdlog::level::info:
      return "INFO";
    case spdlog::level::warn:
      return "WARN";
    case spdlog::level::critical:
      return "CRITICAL";
    default:
      __logger__->critical("Unknown log level: {}", level);
      return "UNKNOWN";
    }
  }

  static void set_log_level(int log_level) {
    auto&& set_log_level_ = [](spdlog::level::level_enum const level) {
      __logger__->info("Setting log level to {}", log_level_name(level));
      __logger__->set_level(level);
    };

    if (log_level > spdlog::level::critical) {
      __logger__->critical(
        "Setting logger level over than CRITICAL; which is not allowed");
      set_log_level_(spdlog::level::info);
      return;
    }
    if (log_level < spdlog::level::trace) {
      __logger__->critical(
        "Setting logger level below than TRACE; which is not allowed");
      set_log_level_(spdlog::level::info);
      return;
    }
    set_log_level_(spdlog::level::level_enum(log_level));
  }

  static std::string expand_user_path(std::string user_path) {
    if (user_path.empty())
      return "";

    if (user_path.at(0) == '~') {
      if (user_path.size() > 1 && user_path.at(1) != '/')
        return user_path;

      char const* home = std::getenv("HOME");
      if (home)
        user_path.replace(0, 1, home);
    }
    return user_path;
  }

  static void init_logger__internal__(
    std::optional<std::string> maybe_path, int log_level) {
    if (maybe_path) {
      auto expanded_path = expand_user_path(*maybe_path);
      try {
        __logger__ = spdlog::basic_logger_mt("cyclops", expanded_path);
        __logger__->flush_on(spdlog::level::debug);
        __logger__->info("Successed to set file logger to {}", expanded_path);
        set_log_level(log_level);
        return;
      } catch (spdlog::spdlog_ex const& err) {
        spdlog::critical(
          "failed to open logger file: {}. trying stdout instead...",
          err.what());
      }
    }

    __logger__ = spdlog::stdout_color_mt("cyclops");
    set_log_level(log_level);
  }

  void init_logger(int log_level) {
    init_logger__internal__(std::nullopt, log_level);
  }

  void init_logger(std::string path, int log_level) {
    init_logger__internal__(path, log_level);
  }
}  // namespace cyclops
