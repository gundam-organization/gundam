//
// Created by Adrien Blanchet on 13/10/2023.
//

#ifndef GUNDAM_PARAMETERSMANAGER_H
#define GUNDAM_PARAMETERSMANAGER_H

#include "FitParameterSet.h"
#include "FitParameter.h"

#include "TMatrixD.h"

#include <vector>
#include <memory>


class ParametersManager : public JsonBaseClass  {

protected:
  // called through public JsonBaseClass::readConfig() and JsonBaseClass::initialize()
  void readConfigImpl() override;
  void initializeImpl() override;

public:
  ParametersManager() = default;

  // setters
  void setReThrowParSetIfOutOfBounds(bool reThrowParSetIfOutOfBounds_){ _reThrowParSetIfOutOfBounds_ = reThrowParSetIfOutOfBounds_; }
  void setThrowToyParametersWithGlobalCov(bool throwToyParametersWithGlobalCov_){ _throwToyParametersWithGlobalCov_ = throwToyParametersWithGlobalCov_; }
  void setGlobalCovarianceMatrix(const std::shared_ptr<TMatrixD> &globalCovarianceMatrix){ _globalCovarianceMatrix_ = globalCovarianceMatrix; }

  // const getters
  [[nodiscard]] const std::shared_ptr<TMatrixD> &getGlobalCovarianceMatrix() const{ return _globalCovarianceMatrix_; }
  [[nodiscard]] const std::shared_ptr<TMatrixD> &getStrippedCovarianceMatrix() const{ return _strippedCovarianceMatrix_; }
  [[nodiscard]] const std::vector<FitParameterSet> &getParameterSetsList() const{ return _parameterSetList_; }

  // getters
  std::shared_ptr<TMatrixD> &getGlobalCovarianceMatrix(){ return _globalCovarianceMatrix_; }
  std::vector<FitParameterSet> &getParameterSetsList(){ return _parameterSetList_; }

  // const core
  [[nodiscard]] std::string getParametersSummary( bool showEigen_ = true ) const;
  [[nodiscard]] nlohmann::json exportParameterInjectorConfig() const;
  [[nodiscard]] const FitParameterSet* getFitParameterSetPtr(const std::string& name_) const;

  // core
  void injectParameterValues(const nlohmann::json &config_);
  void throwParameters();
  void throwParametersFromParSetCovariance();
  void throwParametersFromGlobalCovariance(bool quietVerbose_ = true);
  FitParameterSet* getFitParameterSetPtr(const std::string& name_);

private:
  // config
  bool _reThrowParSetIfOutOfBounds_{true};
  bool _throwToyParametersWithGlobalCov_{false};

  // internals
  std::vector<FitParameterSet> _parameterSetList_{};
  std::vector<FitParameter*> _globalCovParList_{};
  std::vector<FitParameter*> _strippedParameterList_{};
  std::shared_ptr<TMatrixD> _globalCovarianceMatrix_{nullptr};
  std::shared_ptr<TMatrixD> _strippedCovarianceMatrix_{nullptr};
  std::shared_ptr<TMatrixD> _choleskyMatrix_{nullptr};

};


#endif //GUNDAM_PARAMETERSMANAGER_H
