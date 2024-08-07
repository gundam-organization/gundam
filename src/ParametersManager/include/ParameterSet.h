//
// Created by Nadrino on 21/05/2021.
//

#ifndef GUNDAM_PARAMETERSET_H
#define GUNDAM_PARAMETERSET_H

#include "Parameter.h"
#include "JsonBaseClass.h"
#include "ParameterThrowerMarkHarz.h"

#include "Logger.h"
#include "GenericToolbox.CorrelatedVariablesSampler.h"

#include "nlohmann/json.hpp"
#include "TMatrixDSym.h"
#include "TVectorT.h"
#include "TFile.h"
#include "TObjArray.h"
#include "TVectorT.h"
#include "TMatrixDSymEigen.h"

#include <vector>
#include <string>


/*
 * \class FitParameterSet is a class which aims at handling a set of parameters bond together with a covariance matrix
 * User parameters:
 * - Covariance matrix (dim N)
 * - N Fit Parameters (handing dials)
 *
 * */

class ParameterSet : public JsonBaseClass  {

protected:
  // called through public JsonBaseClass::readConfig() and JsonBaseClass::initialize()
  void readConfigImpl() override;
  void initializeImpl() override;

public:
  // in src dependent
  static void muteLogger();
  static void unmuteLogger();

  // Post-init
  void processCovarianceMatrix(); // invert the matrices, and make sure fixed parameters are detached from correlations

  // Setters
  void setMaskedForPropagation(bool maskedForPropagation_){ _maskedForPropagation_ = maskedForPropagation_; }

  // Getters
  [[nodiscard]] bool isEnabled() const{ return _isEnabled_; }
  [[nodiscard]] bool isEnablePca() const{ return _enablePca_; }
  [[nodiscard]] bool isUseEigenDecompInFit() const{ return _useEigenDecompInFit_; }
  [[nodiscard]] bool isEnabledThrowToyParameters() const{ return _enabledThrowToyParameters_; }
  [[nodiscard]] bool isMaskForToyGeneration() const { return _maskForToyGeneration_; }
  [[nodiscard]] bool isMaskedForPropagation() const{ return _maskedForPropagation_; }
  [[nodiscard]] bool isUseOnlyOneParameterPerEvent() const{ return _useOnlyOneParameterPerEvent_; }
  [[nodiscard]] int getNbEnabledEigenParameters() const{ return _nbEnabledEigen_; }
  [[nodiscard]] double getPenaltyChi2Buffer() const{ return _penaltyChi2Buffer_; }
  [[nodiscard]] size_t getNbParameters() const{ return _parameterList_.size(); }
  [[nodiscard]] const std::string &getName() const{ return _name_; }
  [[nodiscard]] const nlohmann::json &getDialSetDefinitions() const{ return _dialSetDefinitions_; }
  [[nodiscard]] const TMatrixD* getInvertedEigenVectors() const{ return _eigenVectorsInv_.get(); }
  [[nodiscard]] const TMatrixD* getEigenVectors() const{ return _eigenVectors_.get(); }
  [[nodiscard]] const std::vector<nlohmann::json>& getCustomFitParThrow() const{ return _customFitParThrow_; }
  [[nodiscard]] const std::shared_ptr<TMatrixDSym> &getPriorCorrelationMatrix() const{ return _priorCorrelationMatrix_; }
  [[nodiscard]] const std::shared_ptr<TMatrixDSym> &getPriorCovarianceMatrix() const { return _priorCovarianceMatrix_; }

  /// Get the vector of parameters for this parameter set in the real
  /// parameter space.  These parameters are not eigendecomposed.
  [[nodiscard]] const std::vector<Parameter> &getParameterList() const{ return _parameterList_; }
  [[nodiscard]] std::vector<Parameter> &getParameterList(){ return _parameterList_; }

  /// Get the vector of parameters for this parameter set in the
  /// eigendecomposed basis.
  [[nodiscard]] const std::vector<Parameter> &getEigenParameterList() const{ return _eigenParameterList_; }
  [[nodiscard]] std::vector<Parameter> &getEigenParameterList(){ return _eigenParameterList_; }

  /// Get the vector of parameters for this parameter set that is applicable
  /// for the current stage of the fit.  This will either be the
  /// eigendecomposed parameters, or the parameters in the non-decomposed
  /// basis.
  [[nodiscard]] const std::vector<Parameter>& getEffectiveParameterList() const;
  [[nodiscard]] std::vector<Parameter>& getEffectiveParameterList();

  double getPenaltyChi2();

  /// Set all of the parameters to their prior values.
  void moveFitParametersToPrior();

  /// Set the parameter values based on a random throw with fluctuations
  /// determined by the striped covariance matrix.  If the first parameter,
  /// rethrowIfNotInbounds, is true, then the throw is retried untill all of
  /// the parameters are within the allowed bounds.  if the second parameter,
  /// gain_, is set, it determines the variance of the thrown distribution
  /// relative to the stripped covariance matrix.  A value larger than one
  /// will increase the thrown variance.
  void throwFitParameters(bool rethrowIfNotInbounds_ = true, double gain_ = 1);

  /// Update the parameter values in the set based on the parameter values in
  /// the eigen decomposed basis.
  void propagateEigenToOriginal();

  /// Update the parameters in the eigen decomposed basis based on the
  /// parameter values in non-decomposed basis.
  void propagateOriginalToEigen();

  // Misc
  [[nodiscard]] std::string getSummary() const;
  [[nodiscard]] nlohmann::json exportInjectorConfig() const;
  void injectParameterValues(const nlohmann::json& config_);
  Parameter* getParameterPtr(const std::string& parName_);
  Parameter* getParameterPtrWithTitle(const std::string& parTitle_);

  // statics
  static double toNormalizedParRange(double parRange, const Parameter& par);
  static double toNormalizedParValue(double parValue, const Parameter& par);
  static double toRealParValue(double normParValue, const Parameter& par);
  static double toRealParRange(double normParRange, const Parameter& par);
  static bool isValidCorrelatedParameter(const Parameter& par_);

protected:
  void readParameterDefinitionFile();
  void defineParameters();
  void fillDeltaParameterList();

private:
  // Internals
  std::vector<Parameter> _parameterList_;

  // JSON
  std::string _name_{};
  std::string _parameterDefinitionFilePath_{};
  std::string _covarianceMatrixTMatrixD_{};
  std::string _parameterPriorTVectorD_{};
  std::string _parameterNameTObjArray_{};
  std::string _parameterLowerBoundsTVectorD_{};
  std::string _parameterUpperBoundsTVectorD_{};
  std::string _throwEnabledListPath_{};
  nlohmann::json _parameterDefinitionConfig_{};
  nlohmann::json _dialSetDefinitions_{};
  bool _isEnabled_{};
  bool _useMarkGenerator_{false};
  bool _useEigenDecompForThrows_{false};
  bool _maskedForPropagation_{false};
  bool _printDialSetsSummary_{false};
  bool _printParametersSummary_{false};
  bool _releaseFixedParametersOnHesse_{false};
  bool _devUseParLimitsOnEigen_{false};
  bool _maskForToyGeneration_{false};
  int _nbParameterDefinition_{-1};
  double _nominalStepSize_{std::nan("unset")};
  int _maxNbEigenParameters_{-1};
  double _maxEigenFraction_{1};

  double _globalParameterMinValue_{std::nan("unset")};
  double _globalParameterMaxValue_{std::nan("unset")};
  std::pair<double, double> _eigenParBounds_{std::nan("unset"), std::nan("unset")};

  double _penaltyChi2Buffer_{std::nan("unset")};

  std::vector<nlohmann::json> _enableOnlyParameters_{};
  std::vector<nlohmann::json> _disableParameters_{};
  std::vector<nlohmann::json> _customFitParThrow_{};

  // Eigen objects
  int _nbEnabledEigen_{0};
  bool _enablePca_{false};
  bool _useEigenDecompInFit_{false};
  bool _useOnlyOneParameterPerEvent_{false};
  std::vector<Parameter> _eigenParameterList_{};
  std::shared_ptr<TMatrixDSymEigen> _eigenDecomp_{nullptr};

  // Toy throwing
  bool _enabledThrowToyParameters_{true};
  std::shared_ptr<TVectorD> _throwEnabledList_{nullptr};

  // Used for base swapping
  std::shared_ptr<TVectorD> _eigenValues_{nullptr};
  std::shared_ptr<TVectorD> _eigenValuesInv_{nullptr};
  std::shared_ptr<TMatrixD> _eigenVectors_{nullptr};
  std::shared_ptr<TMatrixD> _eigenVectorsInv_{nullptr};
  std::shared_ptr<TVectorD> _eigenParBuffer_{nullptr};
  std::shared_ptr<TVectorD> _originalParBuffer_{nullptr};
  std::shared_ptr<TMatrixD> _projectorMatrix_{nullptr};


  std::shared_ptr<TMatrixDSym> _priorCovarianceMatrix_{nullptr};        // matrix coming from the file
  std::shared_ptr<TMatrixDSym> _priorCorrelationMatrix_{nullptr};        // matrix coming from the file
  std::shared_ptr<TMatrixDSym> _strippedCovarianceMatrix_{nullptr};        // matrix stripped from fixed/freed parameters
  std::shared_ptr<TMatrixD>    _inverseStrippedCovarianceMatrix_{nullptr}; // inverse matrix used for chi2

  std::shared_ptr<TVectorD>  _parameterPriorList_{nullptr};
  std::shared_ptr<TVectorD>  _parameterLowerBoundsList_{nullptr};
  std::shared_ptr<TVectorD>  _parameterUpperBoundsList_{nullptr};
  std::shared_ptr<TObjArray> _parameterNamesList_{nullptr};

  std::shared_ptr<TVectorD>  _deltaParameterList_{nullptr}; // difference from prior

  std::shared_ptr<TMatrixD> _choleskyMatrix_{nullptr};
  GenericToolbox::CorrelatedVariablesSampler _correlatedVariableThrower_{};
  std::shared_ptr<ParameterThrowerMarkHarz> _markHartzGen_{nullptr};

};


#endif //GUNDAM_PARAMETERSET_H
