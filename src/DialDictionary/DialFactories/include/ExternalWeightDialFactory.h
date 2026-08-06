#ifndef GUNDAM_EXTERNAL_WEIGHT_DIAL_FACTORY_H
#define GUNDAM_EXTERNAL_WEIGHT_DIAL_FACTORY_H

#include "DialFactoryBase.h"

#include "ConfigUtils.h"

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

class ExternalWeightWorker : public JsonBaseClass {
public:
  ~ExternalWeightWorker() override = default;

  virtual void loadEvents(const std::vector<std::string>& inputNameList_, const std::vector<std::vector<double>>& inputValueList_, std::size_t eventCount_) = 0;
  virtual void evaluate(const DialInputBuffer& inputBuffer_, std::vector<double>& weightList_) = 0;

  [[nodiscard]] virtual const std::vector<std::string>& getInputNameList() const = 0;
};

class ExternalWeightPythonWorker : public ExternalWeightWorker {
public:
  ~ExternalWeightPythonWorker() override;

  void loadEvents(const std::vector<std::string>& inputNameList_, const std::vector<std::vector<double>>& inputValueList_, std::size_t eventCount_) override;
  void evaluate(const DialInputBuffer& inputBuffer_, std::vector<double>& weightList_) override;

  [[nodiscard]] const std::vector<std::string>& getInputNameList() const override { return _inputNameList_; }

private:
  struct SharedMemoryBuffer {
    explicit SharedMemoryBuffer(std::string name_, std::size_t nbDoubles_);
    ~SharedMemoryBuffer();

    SharedMemoryBuffer(const SharedMemoryBuffer&) = delete;
    SharedMemoryBuffer& operator=(const SharedMemoryBuffer&) = delete;

    std::string name{};
    std::size_t nbDoubles{0};
    std::size_t nbBytes{0};
    int fd{-1};
    double* ptr{nullptr};
  };

  static std::string normalizeInputName(const std::string& inputName_);
  void startWorkerProcess(const DialInputBuffer& inputBuffer_);
  void validateEvalScript();
  void sendWorkerCommand(const JsonType& command_);
  JsonType readWorkerResponse();
  void stopWorkerProcess();
  void configureImpl() override;
  void initializeImpl() override;

  std::string _pythonExecutable_{};
  std::string _pythonVenv_{};
  std::string _initScript_{};
  std::string _evalScript_{};
  std::vector<std::string> _scriptArgs_{};
  std::vector<std::string> _inputNameList_{};
  std::vector<std::string> _loadedInputNameList_{};
  std::vector<std::unique_ptr<SharedMemoryBuffer>> _inputBufferList_{};
  std::unique_ptr<SharedMemoryBuffer> _parameterBuffer_{nullptr};
  std::unique_ptr<SharedMemoryBuffer> _weightBuffer_{nullptr};
  std::size_t _eventCount_{0};
  bool _isWorkerStarted_{false};
  int _workerInputFd_{-1};
  int _workerOutputFd_{-1};
  int _workerPid_{-1};
};

class ExternalWeightDialFactory : public DialFactoryBase, public JsonBaseClass {

public:
  ExternalWeightDialFactory() = default;

protected:
  void configureImpl() override;
  void initializeImpl() override;

public:

  [[nodiscard]] DialBase* makeDial(const Event& event_) override;

  void updateWeights(DialInputBuffer& inputBuffer_);
  void finalizeEventLoading();

  [[nodiscard]] const std::vector<std::string>& getInputNameList() const{ return _inputNameList_; }
  [[nodiscard]] const std::vector<double>& getWeightList() const{ return _weightList_; }

private:
  std::vector<std::string> _inputNameList_{};
  std::vector<std::vector<double>> _inputValueList_{};
  std::vector<double> _weightList_{};
  std::unique_ptr<ExternalWeightWorker> _worker_{nullptr};
  bool _eventsLoadedInWorker_{false};
  mutable std::mutex _eventRegistrationMutex_{};

};

#endif // GUNDAM_EXTERNAL_WEIGHT_DIAL_FACTORY_H
