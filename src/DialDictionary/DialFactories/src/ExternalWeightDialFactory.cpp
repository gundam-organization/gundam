#include "ExternalWeightDialFactory.h"

#include "ExternalWeightDispatcher.h"
#include "Event.h"
#include "GenericToolbox.Utils.h"
#include "Logger.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <utility>
#include <unistd.h>

ExternalWeightWorker::SharedMemoryBuffer::SharedMemoryBuffer(
    std::string name_,
    std::size_t nbDoubles_)
    : name(std::move(name_)), nbDoubles(nbDoubles_), nbBytes(std::max<std::size_t>(nbDoubles_, 1)*sizeof(double)) {
  const std::string posixName = "/" + name;
  fd = shm_open(posixName.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
  LogThrowIf(fd == -1, "Could not create shared memory \"" << posixName << "\": " << std::strerror(errno));
  LogThrowIf(ftruncate(fd, off_t(nbBytes)) == -1,
             "Could not resize shared memory \"" << posixName << "\": " << std::strerror(errno));
  ptr = static_cast<double*>(mmap(nullptr, nbBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
  LogThrowIf(ptr == MAP_FAILED,
             "Could not map shared memory \"" << posixName << "\": " << std::strerror(errno));
}

ExternalWeightWorker::SharedMemoryBuffer::~SharedMemoryBuffer(){
  if( ptr != nullptr and ptr != MAP_FAILED ){ munmap(ptr, nbBytes); }
  if( fd != -1 ){ close(fd); }
  if( not name.empty() ){ shm_unlink(("/" + name).c_str()); }
}

std::string ExternalWeightWorker::normalizeInputName(const std::string& inputName_) {
  auto out = GenericToolbox::trimString(inputName_, " ");
  if( out.size() >= 2 and out.front() == '[' and out.back() == ']' ){
    out = out.substr(1, out.size() - 2);
    out = GenericToolbox::trimString(out, " ");
  }
  return out;
}

void ExternalWeightWorker::configureImpl() {

  _config_.fillValue(_inputEventVarNameList_, "inputEventVarList");
  for( auto& inputName : _inputEventVarNameList_ ){
    inputName = normalizeInputName(inputName);
    LogThrowIf(inputName.empty(), "ExternalWeight inputEventVarList contains an empty input name.");
  }
  _inputEventValueList_.resize(_inputEventVarNameList_.size());

}

std::size_t ExternalWeightWorker::registerEvent(const Event& event_) {
  std::lock_guard<std::mutex> lock(_eventRegistrationMutex_);
  LogThrowIf(_areEventsLoaded_, "Cannot register an event after loading ExternalWeight buffers.");

  const std::size_t eventIndex = _weightList_->size();
  for( std::size_t iInput = 0 ; iInput < _inputEventVarNameList_.size() ; ++iInput ){
    _inputEventValueList_[iInput].emplace_back(
        event_.getVariables().fetchVariable(_inputEventVarNameList_[iInput]).getVarAsDouble()
    );
  }
  _weightList_->emplace_back(1.);
  return eventIndex;
}

void ExternalWeightWorker::finalizeEventLoading() {
  if( _areEventsLoaded_ ){ return; }

  _eventCount_ = _weightList_->size();
  _inputBufferList_.clear();
  _inputBufferList_.reserve(_inputEventValueList_.size());
  for( std::size_t iInput = 0 ; iInput < _inputEventValueList_.size() ; ++iInput ){
    LogThrowIf(_inputEventValueList_[iInput].size() != _eventCount_,
               "ExternalWeight input \"" << _inputEventVarNameList_[iInput] << "\" has "
               << _inputEventValueList_[iInput].size() << " entries for " << _eventCount_ << " events.");
    _inputEventValueList_[iInput].shrink_to_fit();
    _inputBufferList_.emplace_back(
        std::make_unique<SharedMemoryBuffer>(
            "gdmEW_" + std::to_string(getpid()) + "_i" + std::to_string(iInput),
            _eventCount_
        )
    );
    std::copy(_inputEventValueList_[iInput].begin(), _inputEventValueList_[iInput].end(), _inputBufferList_.back()->ptr);
  }

  _weightList_->shrink_to_fit();
  _weightBuffer_ = std::make_unique<SharedMemoryBuffer>(
      "gdmEW_" + std::to_string(getpid()) + "_w",
      _eventCount_
  );
  std::fill(_weightBuffer_->ptr, _weightBuffer_->ptr + _eventCount_, 1.);
  _areEventsLoaded_ = true;

  LogInfo << "ExternalWeight worker registered " << _eventCount_
          << " events with inputs " << GenericToolbox::toString(_inputEventVarNameList_) << "." << std::endl;
}

void ExternalWeightWorker::updateWeights(DialInputBuffer& inputBuffer_) {
  if( not _areEventsLoaded_ ){
    this->finalizeEventLoading();
  }
  else if( not inputBuffer_.isDialUpdateRequested() ){
    return;
  }
  this->evaluate(inputBuffer_);
}

void ExternalWeightWorker::evaluate(const DialInputBuffer& inputBuffer_) {
  LogThrowIf(not this->isInitialized(), "ExternalWeight worker is not initialized.");
  LogThrowIf(_weightBuffer_ == nullptr, "ExternalWeight worker has no weight buffer.");
  LogThrowIf(_eventCount_ != _weightList_->size(),
             "ExternalWeight has " << _eventCount_ << " shared-memory weights for "
             << _weightList_->size() << " registered dials.");

  if( _parameterBuffer_ == nullptr ){
    _parameterBuffer_ = std::make_unique<SharedMemoryBuffer>(
        "gdmEW_" + std::to_string(getpid()) + "_parameters",
        std::size_t(inputBuffer_.getInputSize())
    );
  }
  LogThrowIf(_parameterBuffer_->nbDoubles != std::size_t(inputBuffer_.getInputSize()),
             "ExternalWeight parameter buffer size mismatch.");
  for( int iPar = 0 ; iPar < inputBuffer_.getInputSize() ; ++iPar ){
    _parameterBuffer_->ptr[iPar] = inputBuffer_.getInputBuffer().at(iPar);
  }

  this->evaluateImpl(inputBuffer_);
  std::copy(_weightBuffer_->ptr, _weightBuffer_->ptr + _eventCount_, _weightList_->begin());
}

void ExternalWeightDialFactory::configureImpl() {
  _config_.clearFields();
  _config_.defineFields({
      {FieldFlag::MANDATORY, "type"},
      {"inputEventVarList"},
      {"workerConfig"},
    });
  _config_.checkConfiguration();

  const auto workerType = _config_.fetchValue<std::string>("type");
  if( workerType == "PythonWorker" ){
    _worker_ = std::make_unique<ExternalWeightPythonWorker>();
    _worker_->configure(_config_);
  }
  else{
    LogThrow("Unsupported ExternalWeight worker type: \"" << workerType << "\".");
  }

  // The worker owns the event inputs and the calculated weights.
}

void ExternalWeightDialFactory::initializeImpl() {
  LogThrowIf(_worker_ == nullptr, "ExternalWeight worker is not configured.");
  _worker_->initialize();
}

DialBase* ExternalWeightDialFactory::makeDial(const Event& event_) {
  const std::size_t eventIndex = _worker_->registerEvent(event_);
  return new ExternalWeightDispatcher(_worker_->getWeightList(), eventIndex);
}

void ExternalWeightPythonWorker::configureImpl() {
  ExternalWeightWorker::configureImpl();

  auto pythonConfig = _config_.fetchValue<ConfigReader>("workerConfig");
  pythonConfig.defineFields({
      {FieldFlag::MANDATORY, "pythonExecutable"},
      {FieldFlag::MANDATORY, "evalScript"},
      {"initScript"},
      {"scriptArgs"},
    });
  pythonConfig.checkConfiguration();

  pythonConfig.fillValue(_pythonExecutable_, "pythonExecutable");
  pythonConfig.fillValue(_initScript_, "initScript");
  pythonConfig.fillValue(_evalScript_, "evalScript");
  pythonConfig.fillValue(_scriptArgs_, "scriptArgs");

  LogThrowIf(_pythonExecutable_.empty(),
             "ExternalWeight PythonWorker requires either pythonExecutable.");
}

ExternalWeightPythonWorker::~ExternalWeightPythonWorker(){
  this->stopWorkerProcess();
}

void ExternalWeightPythonWorker::initializeImpl() {

  LogThrowIf(_pythonExecutable_.empty(),
             "ExternalWeight pythonExecutable is not configured. Provide pythonExecutable or pythonVenv.");
  LogThrowIf(_evalScript_.empty(),
             "ExternalWeight evalScript is not configured.");
  LogThrowIf(access(_pythonExecutable_.c_str(), X_OK) != 0,
             "ExternalWeight pythonExecutable is not executable: \"" << _pythonExecutable_
             << "\" / " << std::strerror(errno));
  LogThrowIf(access(_evalScript_.c_str(), R_OK) != 0,
             "ExternalWeight evalScript is not readable: \"" << _evalScript_
             << "\" / " << std::strerror(errno));

  if( not _initScript_.empty() ){
    LogWarning << "ExternalWeight initScript is currently ignored by the shared-memory worker. "
               << "Put initialization logic in the worker initialize command handler instead." << std::endl;
  }

  this->validateEvalScript();
}

void ExternalWeightPythonWorker::validateEvalScript() {
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
        _pythonExecutable_.c_str(),
        _pythonExecutable_.c_str(),
        "-m",
        "py_compile",
        _evalScript_.c_str(),
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
      "ExternalWeight evalScript failed `python -m py_compile`: \"" << _evalScript_ << "\""
      << (output.empty() ? "" : std::string{"\n"} + output)
  );
}

void ExternalWeightPythonWorker::evaluateImpl(const DialInputBuffer& inputBuffer_) {
  if( not _isWorkerStarted_ ){
    this->startWorkerProcess(inputBuffer_);
  }
  JsonType command;
  command["command"] = "evaluate";
  this->sendWorkerCommand(command);
  auto response = this->readWorkerResponse();
  LogThrowIf(response.value("status", std::string{}) != "ok",
             "ExternalWeight worker evaluation failed: " << response.dump());
}

void ExternalWeightPythonWorker::startWorkerProcess(const DialInputBuffer& inputBuffer_) {
  if( _isWorkerStarted_ ){ return; }

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

    // The bootstrap owns the wire protocol.  A user script only needs to
    // expose run(command), with the shared-memory descriptions already
    // converted to NumPy arrays by the bootstrap.
    static const char* bootstrap = R"PY(
import json
import runpy
import sys
import traceback
from multiprocessing import shared_memory

import numpy as np


def _attach(description):
    try:
        shm = shared_memory.SharedMemory(name=description["shmName"], track=False)
    except TypeError:
        shm = shared_memory.SharedMemory(name=description["shmName"])
        try:
            from multiprocessing import resource_tracker
            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception:
            pass
    return shm, np.ndarray(tuple(description["shape"]), dtype=np.float64, buffer=shm.buf)


def _respond(payload):
    sys.stdout.write(json.dumps(payload if payload is not None else {"status": "ok"}) + "\n")
    sys.stdout.flush()


def _worker():
    script = sys.argv[1]
    user_args = sys.argv[2:-1]
    # Make argv useful both while importing the script and in configure().
    sys.argv = [script] + user_args
    namespace = runpy.run_path(script, run_name="__external_weight_user__")

    # Keep the old protocol usable while users migrate to run(command).
    if not callable(namespace.get("run")):
        legacy_worker = namespace.get("run_worker")
        if not callable(legacy_worker):
            raise RuntimeError("ExternalWeight evalScript must define run(command_) or run_worker()")
        sys.argv.append("--worker")
        return legacy_worker()

    run = namespace["run"]
    configure = namespace.get("configure")
    if callable(configure):
        configure(user_args)

    shared_memory_list = []
    inputs = {}
    parameter_names = []
    parameter_buffer = None
    weights = None
    parameters = {}
    user_command = None
    read_line = sys.stdin.buffer.readline

    while True:
        line = read_line()
        if not line:
            break
        command = json.loads(line)
        command_name = command.get("command")

        if command_name == "initialize":
            user_command = command
            user_command["parameterInfo"] = command["parameters"]
            inputs.clear()
            for name, description in command["inputs"].items():
                shm, array = _attach(description)
                shared_memory_list.append(shm)
                inputs[name] = array
            shm, parameter_buffer = _attach(command["parameterBuffer"])
            shared_memory_list.append(shm)
            shm, weights = _attach(command["weights"])
            shared_memory_list.append(shm)
            parameter_names[:] = [entry["name"] for entry in command["parameters"]]
            user_command["inputs"] = inputs
            user_command["parameters"] = parameters
            user_command["weights"] = weights
        elif command_name == "evaluate":
            # Reuse the command and parameter dictionary. This avoids
            # allocating three dictionaries on every propagation.
            user_command["command"] = "evaluate"
            parameters.clear()
            parameters.update(zip(parameter_names, parameter_buffer))
        elif command_name == "shutdown":
            user_command["command"] = "shutdown"

        result = run(user_command)
        _respond(result)

        if command_name == "shutdown":
            for shm in shared_memory_list:
                shm.close()
            return 0
    return 0


try:
    sys.exit(_worker())
except Exception as error:
    traceback.print_exc(file=sys.stderr)
    _respond({"status": "error", "message": str(error)})
    sys.exit(1)
)PY";

    std::vector<std::string> argvStorage;
    argvStorage.reserve(4 + _scriptArgs_.size());
    argvStorage.emplace_back(_pythonExecutable_);
    argvStorage.emplace_back("-c");
    argvStorage.emplace_back(bootstrap);
    argvStorage.emplace_back(_evalScript_);
    for( const auto& arg : _scriptArgs_ ){ argvStorage.emplace_back(arg); }
    argvStorage.emplace_back("--worker");

    std::vector<char*> argv;
    argv.reserve(argvStorage.size() + 1);
    // std::string::data() returns const char* before C++17, while execv
    // requires a mutable argv array even though it does not modify its entries.
    for( auto& arg : argvStorage ){ argv.emplace_back(const_cast<char*>(arg.c_str())); }
    argv.emplace_back(nullptr);

    execv(_pythonExecutable_.c_str(), argv.data());
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
  initCommand["nEvents"] = getEventCount();
  initCommand["inputs"] = JsonType::object();
  const auto& inputNameList = getLoadedInputNameList();
  const auto& inputBufferList = getInputBufferList();
  for( std::size_t iInput = 0 ; iInput < inputNameList.size() ; ++iInput ){
    initCommand["inputs"][inputNameList[iInput]] = {
        {"shmName", inputBufferList.at(iInput)->name},
        {"dtype", "float64"},
        {"shape", {getEventCount()}}
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
  const auto* parameterBuffer = getParameterBuffer();
  const auto* weightBuffer = getWeightBuffer();
  LogThrowIf(parameterBuffer == nullptr or weightBuffer == nullptr,
             "ExternalWeight worker shared-memory buffers are not initialized.");
  initCommand["parameterBuffer"] = {
      {"shmName", parameterBuffer->name},
      {"dtype", "float64"},
      {"shape", {std::size_t(inputBuffer_.getInputSize())}}
  };
  initCommand["weights"] = {
      {"shmName", weightBuffer->name},
      {"dtype", "float64"},
      {"shape", {getEventCount()}}
  };

  this->sendWorkerCommand(initCommand);
  auto response = this->readWorkerResponse();
  LogThrowIf(response.value("status", std::string{}) != "ok",
             "ExternalWeight worker initialization failed: " << response.dump());
}

void ExternalWeightPythonWorker::sendWorkerCommand(const JsonType& command_) {
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

JsonType ExternalWeightPythonWorker::readWorkerResponse() {
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

void ExternalWeightPythonWorker::stopWorkerProcess() {
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
