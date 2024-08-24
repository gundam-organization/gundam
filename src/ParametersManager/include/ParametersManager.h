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

  /// Define the type of validity that needs to be required by
  /// hasValidParameterValues.  The validity is propagated to each
  /// ParameterSet.  The validity is:
  ///
  ///  "range" (default) -- Between the parameter minimum and maximum values.
  ///  "norange"         -- Do not require parameters in the valid range
  ///  "mirror"          -- Between the mirrored values (if parameter has
  ///                       mirroring).
  ///  "nomirror"        -- Do not require parameters in the mirrored range
  ///  "physical"        -- Only physically meaningful values.
  ///  "nophysical"      -- Do not require parameters in the physical range.
  ///
  /// Example: setParameterValidity("range,mirror,physical")
  void setParameterValidity(const std::string& validity);

  /// Check that the parameters in all of the enabled ParameterSets are valid.
  [[nodiscard]] bool hasValidParameterSets() const;

private:
  // config
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
