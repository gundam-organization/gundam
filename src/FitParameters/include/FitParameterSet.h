//
// Created by Nadrino on 21/05/2021.
//

#ifndef GUNDAM_FITPARAMETERSET_H
#define GUNDAM_FITPARAMETERSET_H

#include "FitParameter.h"
#ifndef USE_NEW_DIALS
#include "NestedDialTest.h"
#endif
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

#include "vector"
#include "string"


/*
 * \class FitParameterSet is a class which aims at handling a set of parameters bond together with a covariance matrix
 * User parameters:
 * - Covariance matrix (dim N)
 * - N Fit Parameters (handing dials)
 *
 * */

class FitParameterSet : public JsonBaseClass  {

public:
  // Post-init
  void processCovarianceMatrix(); // invert the matrices, and make sure fixed parameters are detached from correlations

  // Setters
  void setMaskedForPropagation(bool maskedForPropagation_);

  // Getters
  [[nodiscard]] bool isEnabled() const;
  [[nodiscard]] bool isEnablePca() const;
  [[nodiscard]] bool isUseEigenDecompInFit() const;
  [[nodiscard]] bool isEnabledThrowToyParameters() const;
  [[nodiscard]] bool isUseOnlyOneParameterPerEvent() const;
  [[nodiscard]] bool isMaskedForPropagation() const;
  [[nodiscard]] int getNbEnabledEigenParameters() const;
  [[nodiscard]] double getPenaltyChi2Buffer() const;
  [[nodiscard]] size_t getNbParameters() const;
  [[nodiscard]] const std::string &getName() const;
  [[nodiscard]] const nlohmann::json &getDialSetDefinitions() const;
  [[nodiscard]] const TMatrixD* getInvertedEigenVectors() const;
  [[nodiscard]] const TMatrixD* getEigenVectors() const;
  [[nodiscard]] const std::vector<nlohmann::json>& getCustomFitParThrow() const;
  [[nodiscard]] const std::shared_ptr<TMatrixDSym> &getPriorCorrelationMatrix() const;
  [[nodiscard]] const std::shared_ptr<TMatrixDSym> &getPriorCovarianceMatrix() const;
  [[nodiscard]] const std::vector<FitParameter> &getParameterList() const;
  [[nodiscard]] const std::vector<FitParameter>& getEffectiveParameterList() const;

  // non-const Getters
  std::vector<FitParameter> &getParameterList();
  std::vector<FitParameter> &getEigenParameterList();
  std::vector<FitParameter>& getEffectiveParameterList();

  // Core
  double getPenaltyChi2();

  // Throw / Shifts
  void moveFitParametersToPrior();
  void throwFitParameters(double gain_ = 1);

  void propagateEigenToOriginal();
  void propagateOriginalToEigen();

  // Misc
  FitParameter* getParameterPtr(const std::string& parName_);
  FitParameter* getParameterPtrWithTitle(const std::string& parTitle_);
  [[nodiscard]] std::string getSummary() const;

  static double toNormalizedParRange(double parRange, const FitParameter& par);
  static double toNormalizedParValue(double parValue, const FitParameter& par);
  static double toRealParValue(double normParValue, const FitParameter& par);
  static double toRealParRange(double normParRange, const FitParameter& par);
  static bool isValidCorrelatedParameter(const FitParameter& par_);

protected:
  void readConfigImpl() override;
  void initializeImpl() override;

  void readParameterDefinitionFile();
  void defineParameters();
  void fillDeltaParameterList();

private:
  // Internals
  std::vector<FitParameter> _parameterList_;
#ifndef USE_NEW_DIALS
  std::vector<NestedDialTest> _nestedDialList_;
#endif

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
  int _nbParameterDefinition_{-1};
  double _nominalStepSize_{std::nan("unset")};
  int _maxNbEigenParameters_{-1};
  double _maxEigenFraction_{1};

  double _globalParameterMinValue_{std::nan("unset")};
  double _globalParameterMaxValue_{std::nan("unset")};

  double _penaltyChi2Buffer_{std::nan("unset")};

  std::vector<nlohmann::json> _enableOnlyParameters_{};
  std::vector<nlohmann::json> _disableParameters_{};
  std::vector<nlohmann::json> _customFitParThrow_{};

  // Eigen objects
  int _nbEnabledEigen_{0};
  bool _enablePca_{false};
  bool _useEigenDecompInFit_{false};
  bool _useOnlyOneParameterPerEvent_{false};
  std::vector<FitParameter> _eigenParameterList_{};
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


#endif //GUNDAM_FITPARAMETERSET_H
