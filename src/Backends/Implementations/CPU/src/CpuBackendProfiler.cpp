#include "CpuBackend.h"
#include "CpuBackendProfiler.h"

#include "VersionConfig.h"
#include "Logger.h"

#include <algorithm>
#include <filesystem>
#include <map>
#include <set>
#include <sstream>
#include <type_traits>
#include <stdexcept>
#include <thread>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#include <sys/utsname.h>
#include <unistd.h>
#endif
#ifdef __linux__
#include <sched.h>
#endif

namespace {
  using Json = Backends::CpuBackendProfiler::Json;
  std::string readFile(const std::string& path_, std::size_t limit_ = 65536) {
    std::ifstream input(path_);
    std::string value;
    char buffer[4096];
    while( input and value.size() < limit_ ){
      input.read(buffer, std::min(sizeof(buffer), limit_ - value.size()));
      value.append(buffer, std::size_t(input.gcount()));
    }
    return value;
  }
  void addFile(Json& files_, const std::string& path_) {
    auto value = readFile(path_);
    if( not value.empty() ){ files_[path_] = value; }
  }
  Json cgroupFiles() {
    Json result = Json::object();
#ifdef __linux__
    // Record both conventional mount roots and process-relative paths. Paths
    // remain explicit: a root limit must not be mistaken for the job's limit.
    const auto membership = readFile("/proc/self/cgroup");
    const std::vector<std::string> mounts{"/sys/fs/cgroup", "/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct",
                                         "/sys/fs/cgroup/memory", "/sys/fs/cgroup/cpuset"};
    std::set<std::string> roots(mounts.begin(), mounts.end());
    std::istringstream lines(membership);
    std::string line;
    while( std::getline(lines, line) ){
      auto first = line.find(':');
      auto last = line.rfind(':');
      if( first == std::string::npos or first == last ){ continue; }
      const auto controllers = line.substr(first + 1, last - first - 1);
      if( not controllers.empty() and controllers.find("cpu") == std::string::npos
          and controllers.find("memory") == std::string::npos ){ continue; }
      auto group = line.substr(last + 1);
      if( group.find("..") != std::string::npos ){ continue; }
      // CPU quotas are often imposed on a job/step parent rather than the
      // leaf task cgroup. Include ancestors up to the conventional mount root.
      for( int depth = 0; depth < 16; ++depth ){
        for( const auto& mount : mounts ){ roots.insert(mount + group); }
        const auto slash = group.rfind('/');
        if( slash == std::string::npos or group == "/" ){ break; }
        group = slash == 0 ? "/" : group.substr(0, slash);
      }
    }
    for( const auto& root : roots ){
      for( const auto* file : {"cpu.max", "cpu.stat", "cpu.cfs_quota_us", "cpu.cfs_period_us",
                              "cpuset.cpus.effective", "cpuset.mems.effective", "memory.current",
                              "memory.max", "memory.events", "memory.numa_stat"} ){
        addFile(result, root + "/" + file);
      }
    }
#endif
    return result;
  }
  Json systemMetadata() {
    Json out = Json::object();
    out["fileReadLimitBytes"] = 65536;
    out["hardwareConcurrency"] = std::thread::hardware_concurrency();
#if defined(__unix__) || defined(__APPLE__)
    struct utsname info{};
    if( uname(&info) == 0 ){
      out["host"] = info.nodename;
      out["system"] = info.sysname;
      out["release"] = info.release;
      out["architecture"] = info.machine;
    }
    out["pid"] = getpid();
#endif
#ifdef __linux__
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if( sched_getaffinity(0, sizeof(affinity), &affinity) == 0 ){
      out["allowedCpus"] = Json::array();
      out["cpuTopology"] = Json::array();
      for( int cpu = 0; cpu < CPU_SETSIZE; ++cpu ){
        if( not CPU_ISSET(cpu, &affinity) ){ continue; }
        out["allowedCpus"].push_back(cpu);
        const std::string root = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
        Json item{{"cpu", cpu}};
        for( const auto* field : {"core_id", "physical_package_id", "thread_siblings_list"} ){
          item[field] = readFile(root + "/topology/" + field);
        }
        out["cpuTopology"].push_back(item);
      }
    }
    Json files = Json::object();
    for( const auto* path : {"/proc/cpuinfo", "/proc/self/status", "/proc/self/cgroup", "/proc/self/mountinfo",
                             "/proc/self/numa_maps", "/sys/devices/system/node/online",
                             "/sys/devices/system/cpu/smt/active"} ){ addFile(files, path); }
    std::istringstream nodes(readFile("/sys/devices/system/node/online"));
    std::string range;
    while( std::getline(nodes, range, ',') ){
      std::istringstream bounds(range);
      int first{-1}, last{-1};
      if( not (bounds >> first) or first < 0 ){ continue; }
      last = first;
      if( bounds.peek() == '-' ){ bounds.get(); bounds >> last; }
      if( last < first or last > 4095 ){ continue; }
      for( int node = first; node <= last; ++node ){
        const auto root = "/sys/devices/system/node/node" + std::to_string(node);
        addFile(files, root + "/cpulist");
        addFile(files, root + "/distance");
      }
    }
    const int cpu = sched_getcpu();
    if( cpu >= 0 ){
      const std::string root = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
      for( const auto* field : {"scaling_driver", "scaling_governor", "scaling_cur_freq", "cpuinfo_max_freq", "base_frequency"} ){
        addFile(files, root + "/cpufreq/" + field);
      }
      for( int index = 0; index < 8; ++index ){
        for( const auto* field : {"level", "type", "size", "shared_cpu_list"} ){
          addFile(files, root + "/cache/index" + std::to_string(index) + "/" + field);
        }
      }
    }
    out["files"] = std::move(files);
#endif
    out["cgroupFiles"] = cgroupFiles();
    return out;
  }
  Json resourceDelta(const Backends::CpuBackendProfiler::Snapshot& begin_,
                     const Backends::CpuBackendProfiler::Snapshot& end_) {
    Json out;
    out["wallSeconds"] = std::chrono::duration<double>(end_.wall - begin_.wall).count();
    out["cpuSeconds"] = begin_.cpuSeconds < 0 or end_.cpuSeconds < 0
                         ? Json(nullptr) : Json(end_.cpuSeconds - begin_.cpuSeconds);
    auto delta = [](long a_, long b_){ return a_ < 0 or b_ < 0 ? Json(nullptr) : Json(b_ - a_); };
    out["minorFaults"] = delta(begin_.minorFaults, end_.minorFaults);
    out["majorFaults"] = delta(begin_.majorFaults, end_.majorFaults);
    out["voluntarySwitches"] = delta(begin_.voluntarySwitches, end_.voluntarySwitches);
    out["involuntarySwitches"] = delta(begin_.involuntarySwitches, end_.involuntarySwitches);
    return out;
  }
}

Backends::CpuBackendProfiler::Snapshot Backends::CpuBackendProfiler::snapshot(bool thread_) {
  Snapshot out;
  out.wall = Clock::now();
#if defined(__unix__) || defined(__APPLE__)
  struct rusage usage{};
  int who = RUSAGE_SELF;
#ifdef __linux__
  if( thread_ ){ who = RUSAGE_THREAD; }
#else
  // Process accounting must not be presented as per-worker accounting.
  if( thread_ ){ return out; }
#endif
  if( getrusage(who, &usage) == 0 ){
    out.cpuSeconds = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec * 1e-6
                   + usage.ru_stime.tv_sec + usage.ru_stime.tv_usec * 1e-6;
    out.minorFaults = usage.ru_minflt;
    out.majorFaults = usage.ru_majflt;
    out.voluntarySwitches = usage.ru_nvcsw;
    out.involuntarySwitches = usage.ru_nivcsw;
  }
#endif
#ifdef __linux__
  out.cpu = sched_getcpu();
#endif
  return out;
}

Backends::CpuBackendProfiler::WorkerScope::WorkerScope(CpuBackendProfiler* profiler_, Stage stage_, int thread_) {
  if( profiler_ == nullptr or not profiler_->_active_ ){ return; }
  _record_ = &profiler_->_workers_[std::size_t(stage_)][thread_];
  _record_->begin = snapshot(true);
  _record_->ran = true;
}
Backends::CpuBackendProfiler::WorkerScope::~WorkerScope() {
  if( _record_ != nullptr ){ _record_->end = snapshot(true); }
}

Backends::CpuBackendProfiler::CpuBackendProfiler(const std::string& path_, std::size_t interval_,
    std::size_t maxSamples_, std::size_t threads_, Json metadata_)
    : _interval_(interval_), _maxSamples_(maxSamples_) {
  for( auto& workers : _workers_ ){ workers.resize(threads_); }
  auto system = systemMetadata();
  auto outputPath = std::filesystem::path(path_);
  outputPath.replace_filename(outputPath.stem().string() + "-" + std::to_string(system.value("pid", 0))
                              + outputPath.extension().string());
  _output_.open(outputPath, std::ios::app);
  if( not _output_ ){ throw std::runtime_error("Cannot open CPU profile output: " + outputPath.string()); }
  metadata_["configuredOutputPath"] = path_;
  metadata_["outputPath"] = outputPath.string();
  // Discover readable CPU accounting files once. Memory/NUMA accounting and
  // repeated discovery can be expensive on large batch nodes.
  std::set<std::string> cpuStatPaths;
  for( auto it = system["cgroupFiles"].begin(); it != system["cgroupFiles"].end(); ++it ){
    auto path = it.key();
    if( path.size() < 9 or path.substr(path.size() - 9) != "/cpu.stat" ){ continue; }
    std::size_t duplicate;
    while( (duplicate = path.find("//")) != std::string::npos ){ path.erase(duplicate, 1); }
    cpuStatPaths.insert(path);
  }
  _cgroupCpuStatPaths_.assign(cpuStatPaths.begin(), cpuStatPaths.end());
  metadata_["fullSystemSnapshotEverySamples"] = 50;
  _runId_ = std::to_string(std::chrono::system_clock::now().time_since_epoch().count())
            + "-" + std::to_string(system.value("pid", 0));
  metadata_["profileStartedUnixSeconds"] = std::chrono::duration<double>(
      std::chrono::system_clock::now().time_since_epoch()).count();
  metadata_["kind"] = "metadata";
  metadata_["system"] = std::move(system);
  metadata_["sampleInterval"] = interval_;
  metadata_["maxSamples"] = maxSamples_;
  write(std::move(metadata_));
  LogInfo << "CPU profiling enabled: " << outputPath.string() << " (first 5 calls, then every " << interval_
          << "; at most " << maxSamples_ << " samples)." << std::endl;
}

void Backends::CpuBackendProfiler::write(Json record_) {
  record_["schemaVersion"] = 1;
  record_["runId"] = _runId_;
  _output_ << record_.dump() << '\n';
  _output_.flush();
  if( not _output_ ){ throw std::runtime_error("Failed to write CPU profile output."); }
}

void Backends::CpuBackendProfiler::begin(std::uint64_t call_, const std::vector<double>& parameters_,
    const std::vector<double>& previous_, bool primed_) {
  _active_ = _samples_ < _maxSamples_ and (call_ <= 5 or call_ % _interval_ == 0);
  if( not _active_ ){ return; }
  _begin_ = snapshot(false);
  _call_ = call_;
  _cachePrimed_ = primed_;
  _changedParameters_.clear();
  for( std::size_t i = 0; i < parameters_.size(); ++i ){
    if( i >= previous_.size() or parameters_[i] != previous_[i] ){ _changedParameters_.push_back(i); }
  }
  for( auto& workers : _workers_ ){ std::fill(workers.begin(), workers.end(), WorkerRecord{}); }
  _stageSeconds_.fill(0.);
  _dirtyDials_ = 0;
  _cachePreparationSeconds_ = 0.;
}
void Backends::CpuBackendProfiler::beginStage(Stage stage_) {
  if( _active_ ){ _stageBegin_[std::size_t(stage_)] = Clock::now(); }
}
void Backends::CpuBackendProfiler::endStage(Stage stage_) {
  if( _active_ ){
    auto index = std::size_t(stage_);
    _stageSeconds_[index] = std::chrono::duration<double>(Clock::now() - _stageBegin_[index]).count();
  }
}
void Backends::CpuBackendProfiler::cachePrepared(std::size_t dirtyDials_) {
  if( not _active_ ){ return; }
  _dirtyDials_ = dirtyDials_;
  _cachePreparationSeconds_ = std::chrono::duration<double>(Clock::now() - _stageBegin_[0]).count();
}
void Backends::CpuBackendProfiler::end(const BackendTimingSummary& timing_) {
  if( not _active_ ){ return; }
  const auto end = snapshot(false);
  Json out{{"kind", "sample"}, {"call", _call_}, {"cacheWasPrimed", _cachePrimed_},
           {"changedParameters", _changedParameters_}, {"dirtyCachedDials", _dirtyDials_},
           {"cachePreparationSeconds", _cachePreparationSeconds_}, {"process", resourceDelta(_begin_, end)}};
  out["unixTimeSeconds"] = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
  out["secondsSinceProfileStart"] = std::chrono::duration<double>(_begin_.wall - _created_).count();
  out["previousSampleExportSeconds"] = _previousExportSeconds_;
  out["stages"] = Json::array();
  const char* names[]{"cache", "weights", "histograms"};
  for( std::size_t stage = 0; stage < 3; ++stage ){
    Json item{{"name", names[stage]}, {"wallSeconds", _stageSeconds_[stage]}, {"workers", Json::array()}};
    double latestEnd{0};
    for( std::size_t thread = 0; thread < _workers_[stage].size(); ++thread ){
      const auto& worker = _workers_[stage][thread];
      if( not worker.ran ){ continue; }
      auto data = resourceDelta(worker.begin, worker.end);
      const double start = std::chrono::duration<double>(worker.begin.wall - _stageBegin_[stage]).count();
      const double finish = std::chrono::duration<double>(worker.end.wall - _stageBegin_[stage]).count();
      latestEnd = std::max(latestEnd, finish);
      data["thread"] = thread;
      data["startOffsetSeconds"] = start;
      data["endOffsetSeconds"] = finish;
      data["cpuAtStart"] = worker.begin.cpu < 0 ? Json(nullptr) : Json(worker.begin.cpu);
      data["cpuAtEnd"] = worker.end.cpu < 0 ? Json(nullptr) : Json(worker.end.cpu);
      if( stage == 0 ){
        data["evaluatedDials"] = worker.evaluatedDials;
        data["unchangedInputs"] = worker.unchangedInputs;
        data["evaluatedDialTypes"] = worker.evaluatedDialTypes;
      }
      if( stage == 1 ){
        data["weightEvents"] = worker.weightEvents;
        data["weightBlocks"] = worker.weightBlocks;
      }
      item["workers"].push_back(data);
    }
    item["joinTailSeconds"] = item["workers"].empty() ? Json(nullptr)
                                : Json(std::max(0., _stageSeconds_[stage] - latestEnd));
    out["stages"].push_back(item);
  }
  out["backendTimingSeconds"] = {{"cache", timing_.cachedDialStageSeconds},
      {"weights", timing_.eventWeightsStageSeconds}, {"histograms", timing_.histogramStageSeconds},
      {"variables", timing_.variableStageSeconds}, {"binning", timing_.binningStageSeconds},
      {"likelihood", timing_.likelihoodHostSeconds}};
  out["cacheTraversal"] = timing_.linearCacheTraversal ? "Linear" : "Indexed";
  out["cacheInputTransformCount"] = timing_.cacheInputTransformCount;
  out["sharedCacheInputs"] = timing_.sharedCacheInputs;
  out["cacheUpdateTimingSeconds"] = {{"preparation", timing_.cachePreparationSeconds},
                                     {"execution", timing_.cacheExecutionSeconds}};
#ifdef __linux__
  // Heavy memory accounting is deliberately sparse. On large nodes, a full
  // export was observed to cost hundreds of milliseconds, outside the kernel
  // timers but still delaying the fit. Keep CPU throttling counters frequent.
  const bool fullSnapshot = _samples_ == 0 or _samples_ == 5 or (_samples_ + 1) % 50 == 0;
  out["systemSnapshot"] = fullSnapshot ? "full" : "light";
  auto readStart = Clock::now();
  Json files = Json::object();
  for( const auto* path : {"/proc/self/status", "/proc/pressure/cpu", "/proc/pressure/memory"} ){
    addFile(files, path);
  }
  if( fullSnapshot ){ addFile(files, "/proc/self/smaps_rollup"); }
  if( _samples_ == 0 or _samples_ == 5 ){ addFile(files, "/proc/self/numa_maps"); }
  out["systemFiles"] = std::move(files);
  out["exportReadSeconds"]["processFiles"] = std::chrono::duration<double>(Clock::now() - readStart).count();
  readStart = Clock::now();
  Json cgroups = Json::object();
  if( fullSnapshot ){ cgroups = cgroupFiles(); }
  else{
    for( const auto& path : _cgroupCpuStatPaths_ ){ addFile(cgroups, path); }
  }
  out["cgroupFiles"] = std::move(cgroups);
  out["exportReadSeconds"]["cgroupFiles"] = std::chrono::duration<double>(Clock::now() - readStart).count();
#endif
  write(std::move(out));
  _previousExportSeconds_ = std::chrono::duration<double>(Clock::now() - end.wall).count();
  ++_samples_;
  _active_ = false;
}

Backends::CpuBackendProfiler::Json Backends::CpuBackend::getWeightSchedulingDiagnostics() const {
  LogThrowIf(not _isBuilt_, "No built CPU model for scheduling diagnostics.");
  const auto& model = _engineView_->propagation;
  CpuBackendProfiler::Json out{
      {"weightScheduling", _usesDynamicWeights_ ? "Dynamic" : "Fixed"},
      {"weightBlocksPerThread", _weightBlocksPerThread_}, {"weightBlockCount", _weightBlocks_.size()},
      {"cacheTraversalRequested", _cacheTraversal_.toString()},
      {"cacheInputEvaluation", _inputEvaluation_.toString()},
      {"cacheInputGroups", _cachedInputGroups_.size()},
      {"cacheInputStorageBytes", getCachedInputStorageBytes()},
      {"perDialInputStorageBytes", model.dials.size() * sizeof(double)},
      {"threads", _threadPool_.getNbThreads()}, {"events", model.events.size()},
      {"eventDialReferences", model.eventDialIndices.size()},
      {"commit", GundamVersionConfig::getVersionPostCommitHash()}};
#ifdef __OPTIMIZE__
  out["compilerOptimizationEnabled"] = true;
#else
  out["compilerOptimizationEnabled"] = false;
#endif
  return out;
}

void Backends::CpuBackend::initializeProfiler() {
  _profiler_.reset();
  if( _profileOutputPath_.empty() ){ return; }
  const auto& model = _engineView_->propagation;
  using Json = CpuBackendProfiler::Json;
  Json metadata{{"backend", "CPU"}, {"threads", _threadPool_.getNbThreads()},
      {"version", GundamVersionConfig::getVersionStr()}, {"commit", GundamVersionConfig::getVersionPostCommitHash()},
      {"branch", GundamVersionConfig::getVersionBranch()}, {"compiler", __VERSION__},
      {"buildType", GUNDAM_CPU_BUILD_TYPE}, {"requiresEventBinning", _requiresEventBinning_},
      {"hasReweightCap", model.hasReweightCap}, {"parameters", model.parameterCount},
      {"events", model.events.size()}, {"uniqueDials", model.dials.size()},
      {"eventDialReferences", model.eventDialIndices.size()}, {"cachedDials", _cachedDialCount_},
      {"weightSchedulingRequested", _weightScheduling_.toString()},
      {"cacheTraversalRequested", _cacheTraversal_.toString()},
      {"cacheInputEvaluation", _inputEvaluation_.toString()},
      {"cacheInputGroups", _cachedInputGroups_.size()},
      {"cacheInputStorageBytes", getCachedInputStorageBytes()},
      {"weightScheduling", _usesDynamicWeights_ ? "Dynamic" : "Fixed"},
      {"weightBlocksPerThread", _weightBlocksPerThread_},
      {"dynamicBlockTarget", _weightBlocksPerThread_ * _threadPool_.getNbThreads()},
      {"weightRangesRole", _usesDynamicWeights_ ? "fixedReference" : "executed"},
      {"histogramBins", model.totalBins}, {"dialTypes", {"Norm", "Shift", "CompactSpline", "UniformSpline",
        "MonotonicSpline", "GeneralSpline", "Graph", "ExternalWeight"}}};
#ifdef __OPTIMIZE__
  metadata["compilerOptimizationEnabled"] = true;
#else
  metadata["compilerOptimizationEnabled"] = false;
#endif
#ifdef NDEBUG
  metadata["ndebug"] = true;
#else
  metadata["ndebug"] = false;
#endif
  Json buffers = Json::object();
  auto buffer = [&](const std::string& name_, const auto& values_){
    std::ostringstream address;
    address << static_cast<const void*>(values_.data());
    buffers[name_] = {{"dataAddress", address.str()}, {"elements", values_.size()}, {"elementBytes", sizeof(typename std::decay_t<decltype(values_)>::value_type)},
        {"sizeBytes", values_.size() * sizeof(values_[0])}, {"capacityBytes", values_.capacity() * sizeof(values_[0])}};
  };
  buffer("eventMetadata", model.events);
  buffer("weightDescriptors", model.eventWeights);
  buffer("eventDialIndices", model.eventDialIndices);
  buffer("dialDescriptors", model.dials);
  buffer("dialInputs", model.dialInputs);
  buffer("dialPayloads", model.dialPayloads);
  buffer("cachedResponses", _cachedDialResponses_);
  buffer("cachedInputs", _cachedDialInputs_);
  buffer("cachedInputGroups", _cachedInputGroups_);
  buffer("cachedDialInputGroups", _cachedDialInputGroups_);
  buffer("cachedFlags", _isCachedDial_);
  buffer("cacheValidity", _isCachedDialResponseValid_);
  buffer("previousParameters", _lastParameterValues_);
  metadata["buffers"] = std::move(buffers);
  metadata["eventWeightOutputBytes"] = model.events.size() * sizeof(double);
  metadata["threadHistogramBytes"] = 2 * _threadPool_.getNbThreads()
      * (model.totalBins + model.samples.size()) * sizeof(double);
  std::array<std::size_t, 8> types{}, cachedTypes{}, referencedTypes{};
  for( std::size_t i = 0; i < model.dials.size(); ++i ){
    ++types[std::size_t(model.dials[i].type)];
    if( _isCachedDial_[i] ){ ++cachedTypes[std::size_t(model.dials[i].type)]; }
  }
  for( auto index : model.eventDialIndices ){ ++referencedTypes[std::size_t(model.dials[index].type)]; }
  metadata["uniqueDialTypes"] = types;
  metadata["cachedDialTypes"] = cachedTypes;
  metadata["referencedDialTypes"] = referencedTypes;
  std::vector<std::size_t> parameterDials;
  for( const auto& dials : _cachedDialsByParameter_ ){ parameterDials.push_back(dials.size()); }
  metadata["cachedDialsPerParameter"] = parameterDials;
  auto describeRanges = [&](const std::vector<EventRange>& ranges_){
    auto result = Json::array();
    for( const auto& range : ranges_ ){
      if( range.begin == range.end ){ continue; }
      std::size_t dials{0};
      for( auto i = range.begin; i < range.end; ++i ){ dials += model.eventWeights[i].dialCount; }
      result.push_back({{"begin", range.begin}, {"end", range.end}, {"events", range.end - range.begin},
                        {"dialReferences", dials}});
    }
    return result;
  };
  metadata["weightRanges"] = describeRanges(_weightRanges_);
  metadata["weightBlocks"] = describeRanges(_usesDynamicWeights_ ? _weightBlocks_ : _weightRanges_);
  metadata["weightBlockCount"] = metadata["weightBlocks"].size();
  std::map<std::size_t, std::size_t> multiplicities;
  for( const auto& event : model.eventWeights ){ ++multiplicities[event.dialCount]; }
  metadata["eventDialCountDistribution"] = Json::array();
  for( const auto& entry : multiplicities ){
    metadata["eventDialCountDistribution"].push_back({{"dials", entry.first}, {"events", entry.second}});
  }
  _profiler_ = std::make_unique<CpuBackendProfiler>(_profileOutputPath_, _profileSampleInterval_,
      _profileMaxSamples_, _threadPool_.getNbThreads(), std::move(metadata));
}
