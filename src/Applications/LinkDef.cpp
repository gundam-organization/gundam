//
// Created by Adrien BLANCHET on 23/11/2022.
//

#include "Logger.h"

#if __cplusplus < 201703L && defined(__CINT__)
// Out of line declaration of non-const static variable (before C++17 -> now "inline" member work)
// Need to declare even default init variables to avoid warning "has internal linkage but is not defined"

// parameters
bool Logger::_enableColors_{LOGGER_ENABLE_COLORS};
bool Logger::_propagateColorsOnUserHeader_{LOGGER_ENABLE_COLORS_ON_USER_HEADER};
bool Logger::_cleanLineBeforePrint_{LOGGER_WRITE_OUTFILE};
bool Logger::_disablePrintfLineJump_{false};
bool Logger::_writeInOutputFile_{false};
Logger::LogLevel Logger::_maxLogLevel_{static_cast<Logger::LogLevel>(LOGGER_MAX_LOG_LEVEL_PRINTED)};
Logger::PrefixLevel Logger::_prefixLevel_{static_cast<Logger::PrefixLevel>(LOGGER_PREFIX_LEVEL)};
std::string Logger::_userHeaderStr_{};
std::string Logger::_prefixFormat_{};
std::string Logger::_indentStr_{};

// internal
bool Logger::_isNewLine_{true};
int Logger::_currentLineNumber_{-1};
std::string Logger::_currentFileName_{};
std::string Logger::_currentPrefix_{};
std::string Logger::_outputFileName_{};
std::mutex Logger::_loggerMutex_{};
LoggerUtils::StreamBufferSupervisor* Logger::_streamBufferSupervisorPtr_{nullptr};
LoggerUtils::StreamBufferSupervisor Logger::_streamBufferSupervisor_{};
Logger::LogLevel Logger::_currentLogLevel_{Logger::LogLevel::TRACE};
Logger::Color Logger::_currentColor_{Logger::Color::RESET};
#endif
