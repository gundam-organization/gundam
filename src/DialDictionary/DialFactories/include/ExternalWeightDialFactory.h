#ifndef GUNDAM_EXTERNAL_WEIGHT_DIAL_FACTORY_H
#define GUNDAM_EXTERNAL_WEIGHT_DIAL_FACTORY_H

#include "DialFactoryBase.h"

#include "ConfigUtils.h"

#include <mutex>
#include <string>
#include <vector>

class ExternalWeightDialFactory : public DialFactoryBase {

public:
  struct PythonWorkerConfig {
    std::string pythonExecutable{};
    std::string pythonVenv{};
    std::string initScript{};
    std::string evalScript{};
  };

  class PythonWorker {
  public:
    void configure(const PythonWorkerConfig& config_);
    void initialize();
    void loadEvents(const std::vector<std::string>& inputNameList_, const std::vector<std::vector<double>>& inputValueList_, std::size_t eventCount_);
    void evaluate(const DialInputBuffer& inputBuffer_, std::vector<double>& weightList_);

    [[nodiscard]] const PythonWorkerConfig& getConfig() const{ return _config_; }

  private:
    PythonWorkerConfig _config_{};
    std::vector<std::string> _loadedInputNameList_{};
    std::vector<std::vector<double>> _loadedInputValueList_{};
    bool _isInitialized_{false};
  };

public:
  ExternalWeightDialFactory() = default;
  explicit ExternalWeightDialFactory(const ConfigReader& config_);

  [[nodiscard]] DialBase* makeDial(const Event& event_) override;

  void updateWeights(DialInputBuffer& inputBuffer_);
  void finalizeEventLoading();

  [[nodiscard]] const std::vector<std::string>& getInputNameList() const{ return _inputNameList_; }
  [[nodiscard]] const std::vector<double>& getWeightList() const{ return _weightList_; }

private:
  static std::string normalizeInputName(const std::string& inputName_);

  std::vector<std::string> _inputNameList_{};
  std::vector<std::vector<double>> _inputValueList_{};
  std::vector<double> _weightList_{};
  PythonWorker _worker_{};
  bool _eventsLoadedInWorker_{false};
  mutable std::mutex _eventRegistrationMutex_{};

};

#endif // GUNDAM_EXTERNAL_WEIGHT_DIAL_FACTORY_H
