#include "CpuWeightProbe.h"

#include <cerrno>
#include <cstring>
#include <utility>

#ifdef __linux__
#include <linux/perf_event.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#endif

Backends::CpuWeightProbe::~CpuWeightProbe() { closeCounters(); }

void Backends::CpuWeightProbe::closeCounters() {
#ifdef __linux__
  for( auto& counter : _counters_ ){
    if( counter.fd >= 0 ){ close(counter.fd); }
    counter = Counter{};
  }
#endif
}

Backends::CpuWeightProbe::Snapshot Backends::CpuWeightProbe::snapshot() {
  Snapshot out;
  out.wall = std::chrono::steady_clock::now();
#ifdef __linux__
  timespec cpuTime{};
  if( clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpuTime) == 0 ){
    out.cpuSeconds = double(cpuTime.tv_sec) + 1e-9 * double(cpuTime.tv_nsec);
  }
  rusage usage{};
  if( getrusage(RUSAGE_THREAD, &usage) == 0 ){
    out.voluntarySwitches = usage.ru_nvcsw;
    out.involuntarySwitches = usage.ru_nivcsw;
    out.minorFaults = usage.ru_minflt;
    out.majorFaults = usage.ru_majflt;
  }
  out.cpu = sched_getcpu();
#endif
  return out;
}

void Backends::CpuWeightProbe::begin() {
  _measured_ = false;
#ifdef __linux__
  const long thread = syscall(SYS_gettid);
  if( thread != _threadId_ ){
    // ParallelWorker implementations may replace threads between jobs.
    closeCounters();
    _threadId_ = thread;
    constexpr std::uint64_t readMiss = (std::uint64_t(PERF_COUNT_HW_CACHE_OP_READ) << 8)
                                     | (std::uint64_t(PERF_COUNT_HW_CACHE_RESULT_MISS) << 16);
    const std::uint32_t types[]{PERF_TYPE_HARDWARE, PERF_TYPE_HARDWARE, PERF_TYPE_HARDWARE,
                               PERF_TYPE_HARDWARE, PERF_TYPE_HW_CACHE, PERF_TYPE_HW_CACHE};
    const std::uint64_t configs[]{PERF_COUNT_HW_CPU_CYCLES, PERF_COUNT_HW_INSTRUCTIONS,
        PERF_COUNT_HW_CACHE_REFERENCES, PERF_COUNT_HW_CACHE_MISSES,
        PERF_COUNT_HW_CACHE_LL | readMiss, PERF_COUNT_HW_CACHE_DTLB | readMiss};
    for( std::size_t i = 0; i < _counters_.size(); ++i ){
      perf_event_attr attr{};
      attr.size = sizeof(attr);
      attr.type = types[i];
      attr.config = configs[i];
      attr.disabled = 1;
      attr.exclude_kernel = 1;
      attr.exclude_hv = 1;
      attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
      auto& counter = _counters_[i];
      counter.fd = int(syscall(SYS_perf_event_open, &attr, 0, -1, -1, PERF_FLAG_FD_CLOEXEC));
      counter.error = counter.fd < 0 ? errno : 0;
    }
  }
  for( auto& counter : _counters_ ){
    counter.active = counter.valid = false;
    if( counter.fd < 0 ){ continue; }
    // Counts/times are cumulative: subtract both snapshots. RESET alone would
    // reset the count but not the enabled/running times across repeated probes.
    const auto bytes = read(counter.fd, &counter.before, sizeof(Reading));
    if( bytes != sizeof(Reading) ){
      counter.error = bytes < 0 ? errno : EIO;
      continue;
    }
    if( ioctl(counter.fd, PERF_EVENT_IOC_ENABLE, 0) != 0 ){
      counter.error = errno;
      continue;
    }
    counter.error = 0;
    counter.active = true;
  }
#endif
  _begin_ = snapshot();
}

void Backends::CpuWeightProbe::end() {
  _end_ = snapshot();
#ifdef __linux__
  for( auto& counter : _counters_ ){
    if( not counter.active ){ continue; }
    counter.active = false;
    if( ioctl(counter.fd, PERF_EVENT_IOC_DISABLE, 0) != 0 ){
      counter.error = errno;
      // Do not retain a counter that could keep running outside the weight job.
      close(counter.fd);
      counter.fd = -1;
      continue;
    }
    const auto bytes = read(counter.fd, &counter.after, sizeof(Reading));
    if( bytes != sizeof(Reading) ){
      counter.error = bytes < 0 ? errno : EIO;
      continue;
    }
    counter.valid = counter.after.value >= counter.before.value
                    and counter.after.enabled >= counter.before.enabled
                    and counter.after.running >= counter.before.running;
  }
#endif
  _measured_ = true;
}

Backends::CpuWeightProbe::Json Backends::CpuWeightProbe::report() const {
  const char* names[]{"cycles", "instructions", "cacheReferences", "cacheMisses",
                      "llcReadMisses", "dtlbReadMisses"};
  Json out{{"measured", _measured_}, {"events", Json::object()},
           {"threadId", _threadId_ < 0 ? Json(nullptr) : Json(_threadId_)}};
#ifdef __linux__
  out["platformSupported"] = true;
#else
  out["platformSupported"] = false;
#endif
  for( std::size_t i = 0; i < _counters_.size(); ++i ){
    const auto& counter = _counters_[i];
    Json event{{"available", counter.fd >= 0}, {"valid", counter.valid},
               {"errno", counter.error}, {"count", nullptr}, {"scaledCount", nullptr},
               {"enabledSeconds", nullptr}, {"runningSeconds", nullptr}, {"runningFraction", nullptr}};
    if( counter.error ){ event["error"] = std::strerror(counter.error); }
    if( counter.valid ){
      const auto count = counter.after.value - counter.before.value;
      const auto enabled = counter.after.enabled - counter.before.enabled;
      const auto running = counter.after.running - counter.before.running;
      event["count"] = count;
      event["enabledSeconds"] = double(enabled) * 1e-9;
      event["runningSeconds"] = double(running) * 1e-9;
      if( running > 0 and enabled > 0 ){
        event["runningFraction"] = double(running) / double(enabled);
        event["scaledCount"] = double(count) * double(enabled) / double(running);
      }
    }
    out["events"][names[i]] = std::move(event);
  }
  if( not _measured_ ){ return out; }
  out["wallSeconds"] = std::chrono::duration<double>(_end_.wall - _begin_.wall).count();
  out["cpuSeconds"] = _begin_.cpuSeconds < 0 or _end_.cpuSeconds < 0 ? Json(nullptr)
                         : Json(_end_.cpuSeconds - _begin_.cpuSeconds);
  out["cpuAtStart"] = _begin_.cpu < 0 ? Json(nullptr) : Json(_begin_.cpu);
  out["cpuAtEnd"] = _end_.cpu < 0 ? Json(nullptr) : Json(_end_.cpu);
  out["voluntarySwitches"] = _begin_.voluntarySwitches < 0 ? Json(nullptr)
                         : Json(_end_.voluntarySwitches - _begin_.voluntarySwitches);
  out["involuntarySwitches"] = _begin_.involuntarySwitches < 0 ? Json(nullptr)
                         : Json(_end_.involuntarySwitches - _begin_.involuntarySwitches);
  out["minorFaults"] = _begin_.minorFaults < 0 ? Json(nullptr) : Json(_end_.minorFaults - _begin_.minorFaults);
  out["majorFaults"] = _begin_.majorFaults < 0 ? Json(nullptr) : Json(_end_.majorFaults - _begin_.majorFaults);
  return out;
}
