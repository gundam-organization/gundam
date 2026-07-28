#include "ExternalWeightDialFactory.h"

#include "ExternalWeight.h"
#include "Event.h"
#include "GenericToolbox.Utils.h"
#include "Logger.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sstream>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <utility>
#include <unistd.h>

namespace {
  std::string trimString(const std::string& str_){
    auto begin = std::find_if_not(str_.begin(), str_.end(), [](unsigned char c){ return std::isspace(c); });
    auto end = std::find_if_not(str_.rbegin(), str_.rend(), [](unsigned char c){ return std::isspace(c); }).base();
    if( begin >= end ){ return {}; }
    return {begin, end};
  }

  std::string buildSharedMemoryName(const std::string& tag_){
    std::stringstream ss;
    ss << "gdmEW_" << getpid() << "_" << tag_;
    return ss.str();
  }

  std::string toPosixSharedMemoryName(const std::string& name_){
    return "/" + name_;
  }
}

ExternalWeightDialFactory::PythonWorker::SharedMemoryBuffer::SharedMemoryBuffer(
    std::string name_,
    std::size_t nbDoubles_)
    : name(std::move(name_)), nbDoubles(nbDoubles_), nbBytes(std::max<std::size_t>(nbDoubles_, 1)*sizeof(double)) {
  auto posixName = toPosixSharedMemoryName(name);
  fd = shm_open(posixName.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
  LogThrowIf(fd == -1, "Could not create shared memory \"" << posixName << "\": " << std::strerror(errno));
  LogThrowIf(ftruncate(fd, off_t(nbBytes)) == -1,
             "Could not resize shared memory \"" << posixName << "\": " << std::strerror(errno));
  ptr = static_cast<double*>(mmap(nullptr, nbBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
  LogThrowIf(ptr == MAP_FAILED,
             "Could not map shared memory \"" << posixName << "\": " << std::strerror(errno));
}

ExternalWeightDialFactory::PythonWorker::SharedMemoryBuffer::~SharedMemoryBuffer(){
  if( ptr != nullptr and ptr != MAP_FAILED ){ munmap(ptr, nbBytes); }
  if( fd != -1 ){ close(fd); }
  if( not name.empty() ){ shm_unlink(toPosixSharedMemoryName(name).c_str()); }
}

ExternalWeightDialFactory::ExternalWeightDialFactory(const ConfigReader& config_) {
  ConfigReader config(config_);

  if( config.hasField("externalWeight") ){
    config = config.fetchValue<ConfigReader>("externalWeight");
  }

  config.defineFields({
      {"pythonExecutable"},
      {"pythonVenv"},
      {"initScript"},
      {"evalScript"},
      {"scriptArgs"},
      {"inputList"},
    });
  config.checkConfiguration();

  config.fillValue(_inputNameList_, "inputList");
  LogThrowIf(_inputNameList_.empty(), "ExternalWeight requires a non-empty inputList.");

  for( auto& inputName : _inputNameList_ ){
    inputName = normalizeInputName(inputName);
    LogThrowIf(inputName.empty(), "ExternalWeight inputList contains an empty input name.");
  }
  _inputValueList_.resize(_inputNameList_.size());

  PythonWorkerConfig workerConfig{};
  config.fillValue(workerConfig.pythonExecutable, "pythonExecutable");
  config.fillValue(workerConfig.pythonVenv, "pythonVenv");
  config.fillValue(workerConfig.initScript, "initScript");
  config.fillValue(workerConfig.evalScript, "evalScript");
  config.fillValue(workerConfig.scriptArgs, "scriptArgs");

  if( workerConfig.pythonExecutable.empty() and not workerConfig.pythonVenv.empty() ){
    workerConfig.pythonExecutable = workerConfig.pythonVenv + "/bin/python";
  }

  _worker_.configure(workerConfig);
  _worker_.initialize();
}

DialBase* ExternalWeightDialFactory::makeDial(const Event& event_) {
  std::lock_guard<std::mutex> lock(_eventRegistrationMutex_);

  const std::size_t eventIndex = _weightList_.size();

  for( std::size_t iInput = 0 ; iInput < _inputNameList_.size() ; ++iInput ){
    _inputValueList_[iInput].emplace_back(event_.getVariables().fetchVariable(_inputNameList_[iInput]).getVarAsDouble());
  }

  _weightList_.emplace_back(1.);
  return new ExternalWeight(&_weightList_, eventIndex);
}

void ExternalWeightDialFactory::updateWeights(DialInputBuffer& inputBuffer_) {
  if( not _eventsLoadedInWorker_ ){
    this->finalizeEventLoading();
  }
  else if( not inputBuffer_.isDialUpdateRequested() ){
    return;
  }

  for( const auto& values : _inputValueList_ ){
    LogThrowIf(values.size() != _weightList_.size(),
               "ExternalWeight internal size mismatch: "
               << _weightList_.size() << " weights for "
               << values.size() << " registered input values.");
  }
  _worker_.evaluate(inputBuffer_, _weightList_);
}

void ExternalWeightDialFactory::finalizeEventLoading() {
  if( _eventsLoadedInWorker_ ){ return; }

  for( auto& values : _inputValueList_ ){
    LogThrowIf(values.size() != _weightList_.size(),
               "ExternalWeight input size mismatch: "
               << values.size() << " values for " << _weightList_.size() << " registered events.");
    values.shrink_to_fit();
  }
  _weightList_.shrink_to_fit();
  _worker_.loadEvents(_inputNameList_, _inputValueList_, _weightList_.size());
  _eventsLoadedInWorker_ = true;
}

std::string ExternalWeightDialFactory::normalizeInputName(const std::string& inputName_) {
  auto out = trimString(inputName_);
  if( out.size() >= 2 and out.front() == '[' and out.back() == ']' ){
    out = out.substr(1, out.size() - 2);
    out = trimString(out);
  }
  return out;
}

void ExternalWeightDialFactory::PythonWorker::configure(const PythonWorkerConfig& config_) {
  _config_ = config_;
}

ExternalWeightDialFactory::PythonWorker::~PythonWorker(){
  this->stopWorkerProcess();
}

void ExternalWeightDialFactory::PythonWorker::initialize() {
  if( _isInitialized_ ){ return; }

  LogThrowIf(_config_.pythonExecutable.empty(),
             "ExternalWeight pythonExecutable is not configured. Provide pythonExecutable or pythonVenv.");
  LogThrowIf(_config_.evalScript.empty(),
             "ExternalWeight evalScript is not configured.");
  LogThrowIf(access(_config_.pythonExecutable.c_str(), X_OK) != 0,
             "ExternalWeight pythonExecutable is not executable: \"" << _config_.pythonExecutable
             << "\" / " << std::strerror(errno));
  LogThrowIf(access(_config_.evalScript.c_str(), R_OK) != 0,
             "ExternalWeight evalScript is not readable: \"" << _config_.evalScript
             << "\" / " << std::strerror(errno));

  if( not _config_.initScript.empty() ){
    LogWarning << "ExternalWeight initScript is currently ignored by the shared-memory worker. "
               << "Put initialization logic in the worker initialize command handler instead." << std::endl;
  }

  this->validateEvalScript();
  _isInitialized_ = true;
}

void ExternalWeightDialFactory::PythonWorker::validateEvalScript() {
  int stdoutPipe[2];
  LogThrowIf(pipe(stdoutPipe) == -1,
             "Could not create ExternalWeight py_compile pipe: " << std::strerror(errno));

  const pid_t pid = fork();
  LogThrowIf(pid == -1, "Could not fork ExternalWeight py_compile process: " << std::strerror(errno));

  if( pid == 0 ){
    dup2(stdoutPipe[1], STDOUT_FILENO);
    dup2(stdoutPipe[1], STDERR_FILENO);
    close(stdoutPipe[0]);
    close(stdoutPipe[1]);

    execl(
        _config_.pythonExecutable.c_str(),
        _config_.pythonExecutable.c_str(),
        "-m",
        "py_compile",
        _config_.evalScript.c_str(),
        static_cast<char*>(nullptr)
    );
    _exit(127);
  }

  close(stdoutPipe[1]);

  std::string output;
  char buffer[1024];
  while( true ){
    ssize_t nRead = read(stdoutPipe[0], buffer, sizeof(buffer));
    if( nRead == 0 ){ break; }
    LogThrowIf(nRead < 0,
               "Could not read ExternalWeight py_compile output: " << std::strerror(errno));
    output.append(buffer, std::size_t(nRead));
  }
  close(stdoutPipe[0]);

  int status = 0;
  waitpid(pid, &status, 0);
  LogThrowIf(
      not WIFEXITED(status) or WEXITSTATUS(status) != 0,
      "ExternalWeight evalScript failed `python -m py_compile`: \"" << _config_.evalScript << "\""
      << (output.empty() ? "" : std::string{"\n"} + output)
  );
}

void ExternalWeightDialFactory::PythonWorker::loadEvents(
    const std::vector<std::string>& inputNameList_,
    const std::vector<std::vector<double>>& inputValueList_,
    std::size_t eventCount_) {
  LogThrowIf(not _isInitialized_, "ExternalWeight PythonWorker is not initialized.");
  LogThrowIf(inputNameList_.size() != inputValueList_.size(),
             "ExternalWeight worker received " << inputNameList_.size()
             << " input names but " << inputValueList_.size() << " input arrays.");
  _loadedInputNameList_ = inputNameList_;
  _eventCount_ = eventCount_;

  _inputBufferList_.clear();
  _inputBufferList_.reserve(inputValueList_.size());
  for( std::size_t iInput = 0 ; iInput < inputValueList_.size() ; ++iInput ){
    LogThrowIf(inputValueList_[iInput].size() != eventCount_,
               "ExternalWeight input \"" << inputNameList_[iInput] << "\" has "
               << inputValueList_[iInput].size() << " entries for " << eventCount_ << " events.");
    _inputBufferList_.emplace_back(
        std::make_unique<SharedMemoryBuffer>(
            buildSharedMemoryName("i" + std::to_string(iInput)),
            eventCount_
        )
    );
    std::copy(inputValueList_[iInput].begin(), inputValueList_[iInput].end(), _inputBufferList_.back()->ptr);
  }

  _weightBuffer_ = std::make_unique<SharedMemoryBuffer>(
      buildSharedMemoryName("w"),
      eventCount_
  );
  std::fill(_weightBuffer_->ptr, _weightBuffer_->ptr + eventCount_, 1.);

  LogInfo << "ExternalWeight worker registered " << eventCount_
          << " events with inputs " << GenericToolbox::toString(inputNameList_) << "." << std::endl;
}

void ExternalWeightDialFactory::PythonWorker::evaluate(
    const DialInputBuffer& inputBuffer_,
    std::vector<double>& weightList_) {
  LogThrowIf(not _isInitialized_, "ExternalWeight PythonWorker is not initialized.");
  LogThrowIf(_weightBuffer_ == nullptr, "ExternalWeight worker has no weight buffer.");
  LogThrowIf(_eventCount_ != weightList_.size(),
             "ExternalWeight has " << _eventCount_ << " shared-memory weights for "
             << weightList_.size() << " registered dials.");

  if( not _isWorkerStarted_ ){
    this->startWorkerProcess(inputBuffer_);
  }

  LogThrowIf(_parameterBuffer_ == nullptr, "ExternalWeight worker has no parameter buffer.");
  LogThrowIf(_parameterBuffer_->nbDoubles != std::size_t(inputBuffer_.getInputSize()),
             "ExternalWeight parameter buffer size mismatch.");

  for( int iPar = 0 ; iPar < inputBuffer_.getInputSize() ; ++iPar ){
    _parameterBuffer_->ptr[iPar] = inputBuffer_.getInputBuffer().at(iPar);
  }

  JsonType command;
  command["command"] = "evaluate";
  this->sendWorkerCommand(command);
  auto response = this->readWorkerResponse();
  LogThrowIf(response.value("status", std::string{}) != "ok",
             "ExternalWeight worker evaluation failed: " << response.dump());

  std::copy(_weightBuffer_->ptr, _weightBuffer_->ptr + _eventCount_, weightList_.begin());
}

void ExternalWeightDialFactory::PythonWorker::startWorkerProcess(const DialInputBuffer& inputBuffer_) {
  if( _isWorkerStarted_ ){ return; }

  _parameterBuffer_ = std::make_unique<SharedMemoryBuffer>(
      buildSharedMemoryName("parameters"),
      std::size_t(inputBuffer_.getInputSize())
  );

  int stdinPipe[2];
  int stdoutPipe[2];
  LogThrowIf(pipe(stdinPipe) == -1, "Could not create ExternalWeight worker stdin pipe: " << std::strerror(errno));
  LogThrowIf(pipe(stdoutPipe) == -1, "Could not create ExternalWeight worker stdout pipe: " << std::strerror(errno));

  const pid_t pid = fork();
  LogThrowIf(pid == -1, "Could not fork ExternalWeight worker: " << std::strerror(errno));

  if( pid == 0 ){
    dup2(stdinPipe[0], STDIN_FILENO);
    dup2(stdoutPipe[1], STDOUT_FILENO);
    close(stdinPipe[0]);
    close(stdinPipe[1]);
    close(stdoutPipe[0]);
    close(stdoutPipe[1]);

    std::vector<std::string> argvStorage;
    argvStorage.reserve(3 + _config_.scriptArgs.size());
    argvStorage.emplace_back(_config_.pythonExecutable);
    argvStorage.emplace_back(_config_.evalScript);
    for( const auto& arg : _config_.scriptArgs ){ argvStorage.emplace_back(arg); }
    argvStorage.emplace_back("--worker");

    std::vector<char*> argv;
    argv.reserve(argvStorage.size() + 1);
    for( auto& arg : argvStorage ){ argv.emplace_back(arg.data()); }
    argv.emplace_back(nullptr);

    execv(_config_.pythonExecutable.c_str(), argv.data());
    _exit(127);
  }

  close(stdinPipe[0]);
  close(stdoutPipe[1]);
  _workerInputFd_ = stdinPipe[1];
  _workerOutputFd_ = stdoutPipe[0];
  _workerPid_ = int(pid);
  _isWorkerStarted_ = true;

  JsonType initCommand;
  initCommand["command"] = "initialize";
  initCommand["nEvents"] = _eventCount_;
  initCommand["inputs"] = JsonType::object();
  for( std::size_t iInput = 0 ; iInput < _loadedInputNameList_.size() ; ++iInput ){
    initCommand["inputs"][_loadedInputNameList_[iInput]] = {
        {"shmName", _inputBufferList_.at(iInput)->name},
        {"dtype", "float64"},
        {"shape", {_eventCount_}}
    };
  }

  initCommand["parameters"] = JsonType::array();
  for( int iPar = 0 ; iPar < inputBuffer_.getInputSize() ; ++iPar ){
    initCommand["parameters"].push_back({
        {"name", inputBuffer_.getParameter(iPar).getName()},
        {"title", inputBuffer_.getParameter(iPar).getFullTitle()},
        {"index", iPar}
    });
  }
  initCommand["parameterBuffer"] = {
      {"shmName", _parameterBuffer_->name},
      {"dtype", "float64"},
      {"shape", {std::size_t(inputBuffer_.getInputSize())}}
  };
  initCommand["weights"] = {
      {"shmName", _weightBuffer_->name},
      {"dtype", "float64"},
      {"shape", {_eventCount_}}
  };

  this->sendWorkerCommand(initCommand);
  auto response = this->readWorkerResponse();
  LogThrowIf(response.value("status", std::string{}) != "ok",
             "ExternalWeight worker initialization failed: " << response.dump());
}

void ExternalWeightDialFactory::PythonWorker::sendWorkerCommand(const JsonType& command_) {
  LogThrowIf(_workerInputFd_ == -1, "ExternalWeight worker input pipe is not open.");

  const std::string line = command_.dump() + "\n";
  const char* data = line.data();
  std::size_t remaining = line.size();
  while( remaining != 0 ){
    ssize_t nWritten = write(_workerInputFd_, data, remaining);
    LogThrowIf(nWritten <= 0, "Could not write to ExternalWeight worker: " << std::strerror(errno));
    data += nWritten;
    remaining -= std::size_t(nWritten);
  }
}

JsonType ExternalWeightDialFactory::PythonWorker::readWorkerResponse() {
  LogThrowIf(_workerOutputFd_ == -1, "ExternalWeight worker output pipe is not open.");

  std::string line;
  char c;
  while( true ){
    ssize_t nRead = read(_workerOutputFd_, &c, 1);
    LogThrowIf(nRead <= 0, "Could not read from ExternalWeight worker: " << std::strerror(errno));
    if( c == '\n' ){ break; }
    line += c;
  }

  try{
    return JsonType::parse(line);
  }
  catch( const std::exception& e ){
    LogThrow("ExternalWeight worker response is not valid JSON: \"" << line << "\" / " << e.what());
  }
  return {};
}

void ExternalWeightDialFactory::PythonWorker::stopWorkerProcess() {
  if( _isWorkerStarted_ and _workerInputFd_ != -1 ){
    try{
      JsonType command;
      command["command"] = "shutdown";
      this->sendWorkerCommand(command);
      auto response = this->readWorkerResponse();
      LogThrowIf(response.value("status", std::string{}) != "ok",
                 "ExternalWeight worker shutdown failed: " << response.dump());
    }
    catch(...){}
  }

  if( _workerInputFd_ != -1 ){ close(_workerInputFd_); _workerInputFd_ = -1; }
  if( _workerOutputFd_ != -1 ){ close(_workerOutputFd_); _workerOutputFd_ = -1; }

  if( _workerPid_ > 0 ){
    int status;
    waitpid(pid_t(_workerPid_), &status, 0);
    _workerPid_ = -1;
  }

  _isWorkerStarted_ = false;
}
