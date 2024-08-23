//
// Created by Nadrino on 13/10/2023.
//

#ifndef GUNDAM_PARAMETERS_MANAGER_H
#define GUNDAM_PARAMETERS_MANAGER_H

#include "ParameterSet.h"
#include "Parameter.h"

#include "TMatrixD.h"

#include <vector>
#include <memory>


class ParametersManager : public JsonBaseClass  {

protected:
  // called through public JsonBaseClass::readConfig() and JsonBaseClass::initialize()
  void readConfigImpl() override;
  void initializeImpl() override;

public:
  // setters
  void setParameterSetListConfig(const JsonType& parameterSetListConfig_){ _parameterSetListConfig_ = parameterSetListConfig_; }
  // Deprecated because out of bounds parameters are illegal and will result
  // in a crash.  An out of bounds parameter is always rethrown and this value
  // is ignored.
  [[deprecated("Must always be true")]] void setReThrowParSetIfOutOfBounds(bool reThrowParSetIfOutOfBounds_){ _reThrowParSetIfOutOfBounds_ = reThrowParSetIfOutOfBounds_; }
  // If set to true, parameters that are outside of the physical bounds will
  // be rethrown.  The physical bounds are always contained in the overall
  // parameter bounds, and the throw value will always be inside of the
  // overall parameter bounds (the overall bounds are the "domain" of the
  // parameter).  See the Parameter::setMinValue() and
  // Parameter::setMinPhysical for more details.
  void setReThrowParSetIfOutOfPhysical(bool reThrowParSetIfOutOfPhysical_){ _reThrowParSetIfOutOfPhysical_ = reThrowParSetIfOutOfPhysical_; }
  void setThrowToyParametersWithGlobalCov(bool throwToyParametersWithGlobalCov_){ _throwToyParametersWithGlobalCov_ = throwToyParametersWithGlobalCov_; }
  void setGlobalCovarianceMatrix(const std::shared_ptr<TMatrixD> &globalCovarianceMatrix){ _globalCovarianceMatrix_ = globalCovarianceMatrix; }

  // const getters
  [[nodiscard]] const std::shared_ptr<TMatrixD> &getGlobalCovarianceMatrix() const{ return _globalCovarianceMatrix_; }
  [[nodiscard]] const std::shared_ptr<TMatrixD> &getStrippedCovarianceMatrix() const{ return _strippedCovarianceMatrix_; }
  [[nodiscard]] const std::vector<ParameterSet> &getParameterSetsList() const{ return _parameterSetList_; }

  // getters
  std::shared_ptr<TMatrixD> &getGlobalCovarianceMatrix(){ return _globalCovarianceMatrix_; }
  std::vector<ParameterSet> &getParameterSetsList(){ return _parameterSetList_; }

  // const core
  [[nodiscard]] std::string getParametersSummary( bool showEigen_ = true ) const;
  [[nodiscard]] JsonType exportParameterInjectorConfig() const;
  [[nodiscard]] const ParameterSet* getFitParameterSetPtr(const std::string& name_) const;

  // core
  void moveParametersToPrior();
  void injectParameterValues(const JsonType &config_);
  void throwParameters();
  void throwParametersFromParSetCovariance();
  void throwParametersFromGlobalCovariance(bool quietVerbose_ = true);
  ParameterSet* getFitParameterSetPtr(const std::string& name_);

  // Logger related
  static void muteLogger();
  static void unmuteLogger();

private:
  // config
  bool _reThrowParSetIfOutOfBounds_{true};  // NO LONGER USE. ALWAYS TRUE.
  bool _reThrowParSetIfOutOfPhysical_{true};
  bool _throwToyParametersWithGlobalCov_{false};
  JsonType _parameterSetListConfig_{};

  // internals
  std::vector<ParameterSet> _parameterSetList_{};
  std::vector<Parameter*> _globalCovParList_{};
  std::vector<Parameter*> _strippedParameterList_{};
  std::shared_ptr<TMatrixD> _globalCovarianceMatrix_{nullptr};
  std::shared_ptr<TMatrixD> _strippedCovarianceMatrix_{nullptr};
  std::shared_ptr<TMatrixD> _choleskyMatrix_{nullptr};

};
#endif //GUNDAM_PARAMETERS_MANAGER_H
