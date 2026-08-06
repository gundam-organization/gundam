#ifndef GUNDAM_EXTERNAL_WEIGHT_DIAL_FACTORY_H
#define GUNDAM_EXTERNAL_WEIGHT_DIAL_FACTORY_H

#include "DialFactoryBase.h"

#include "ConfigUtils.h"

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

class Event;

class ExternalWeightWorker : public JsonBaseClass {
public:
  ~ExternalWeightWorker() override = default;

  std::size_t registerEvent(const Event& event_);
  void updateWeights(DialInputBuffer& inputBuffer_);

  [[nodiscard]] const std::vector<std::string>& getInputNameList() const { return _inputEventVarNameList_; }
  [[nodiscard]] const std::shared_ptr<std::vector<double>>& getWeightList() const { return _weightList_; }

protected:
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

  [[nodiscard]] const std::vector<std::string>& getLoadedInputNameList() const { return _inputEventVarNameList_; }
  [[nodiscard]] const std::vector<std::unique_ptr<SharedMemoryBuffer>>& getInputBufferList() const { return _inputBufferList_; }
  [[nodiscard]] const SharedMemoryBuffer* getParameterBuffer() const { return _parameterBuffer_.get(); }
  [[nodiscard]] const SharedMemoryBuffer* getWeightBuffer() const { return _weightBuffer_.get(); }
  [[nodiscard]] std::size_t getEventCount() const { return _eventCount_; }

protected:
  static std::string normalizeInputName(const std::string& inputName_);
  void configureImpl() override;

private:
  virtual void evaluateImpl(const DialInputBuffer& inputBuffer_) = 0;
  void evaluate(const DialInputBuffer& inputBuffer_);
  void finalizeEventLoading();

  std::vector<std::string> _inputEventVarNameList_{};
  std::vector<std::vector<double>> _inputEventValueList_{};
  std::shared_ptr<std::vector<double>> _weightList_{std::make_shared<std::vector<double>>()};
  std::vector<std::unique_ptr<SharedMemoryBuffer>> _inputBufferList_{};
  std::unique_ptr<SharedMemoryBuffer> _parameterBuffer_{nullptr};
  std::unique_ptr<SharedMemoryBuffer> _weightBuffer_{nullptr};
  std::size_t _eventCount_{0};
  bool _areEventsLoaded_{false};
  mutable std::mutex _eventRegistrationMutex_{};
};

class ExternalWeightPythonWorker : public ExternalWeightWorker {
public:
  ~ExternalWeightPythonWorker() override;

protected:
  void configureImpl() override;
  void initializeImpl() override;

private:
  void startWorkerProcess(const DialInputBuffer& inputBuffer_);
  void validateEvalScript();
  void sendWorkerCommand(const JsonType& command_);
  JsonType readWorkerResponse();
  void stopWorkerProcess();
  void evaluateImpl(const DialInputBuffer& inputBuffer_) override;
  std::string _pythonExecutable_{};
  std::string _pythonVenv_{};
  std::string _initScript_{};
  std::string _evalScript_{};
  std::vector<std::string> _scriptArgs_{};
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

  [[nodiscard]] const std::vector<std::string>& getInputNameList() const{ return _worker_->getInputNameList(); }

private:
  std::unique_ptr<ExternalWeightWorker> _worker_{nullptr};

};

#endif // GUNDAM_EXTERNAL_WEIGHT_DIAL_FACTORY_H
