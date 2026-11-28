#ifndef GUNDAM_BACKEND_EVENT_CACHE_PROGRESS_H
#define GUNDAM_BACKEND_EVENT_CACHE_PROGRESS_H

#include "Logger.h"

#include <chrono>
#include <cstddef>

namespace Backends {
  class EventCacheProgress {
  public:
    EventCacheProgress(const char* phase_, std::size_t total_)
        : _phase_(phase_), _total_(total_) {}

    void advance() {
      // Check the clock only at fixed intervals in the event loop.
      if( ++_processed_ % 8192 != 0 or _processed_ == _total_ ){ return; }
      const auto now = Clock::now();
      if( now - _lastReport_ < std::chrono::seconds(2) ){ return; }
      _lastReport_ = now;
      LogInfo << _phase_ << " (" << _processed_ << "/" << _total_ << " event cache entries, "
              << int(100. * double(_processed_) / double(_total_)) << "%)..." << std::endl;
    }

    [[nodiscard]] double elapsedSeconds() const { return std::chrono::duration<double>(Clock::now() - _start_).count(); }

  private:
    using Clock = std::chrono::steady_clock;
    const char* _phase_;
    std::size_t _total_;
    std::size_t _processed_{0};
    Clock::time_point _start_{Clock::now()};
    Clock::time_point _lastReport_{_start_};
  };
}

#endif
