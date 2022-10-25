//
// Created by Nadrino on 21/05/2021.
//

#ifndef GUNDAM_FITPARAMETERSET_H
#define GUNDAM_FITPARAMETERSET_H

#include "FitParameter.h"
#include "NestedDialTest.h"
#include "JsonBaseClass.h"

#include "Logger.h"

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
  // Setters
  void setSaveDir(TDirectory* saveDir_);

  // Post-init
  void prepareFitParameters(); // invert the matrices, and make sure fixed parameters are detached from correlations

  // Getters
  bool isEnabled() const;
  bool isEnabledThrowToyParameters() const;
  bool isUseOnlyOneParameterPerEvent() const;
  const std::string &getName() const;
  std::vector<FitParameter> &getParameterList();
  std::vector<FitParameter> &getEigenParameterList();
  const std::vector<FitParameter> &getParameterList() const;
  const std::shared_ptr<TMatrixDSym> &getPriorCorrelationMatrix() const;
  const std::shared_ptr<TMatrixDSym> &getPriorCovarianceMatrix() const;
  std::vector<FitParameter>& getEffectiveParameterList();
  const std::vector<FitParameter>& getEffectiveParameterList() const;
  bool isMaskedForPropagation() const;

  // Core
  size_t getNbParameters() const;
  double getPenaltyChi2();

  // Throw / Shifts
  void moveFitParametersToPrior();
  void throwFitParameters(double gain_ = 1);
  const std::vector<nlohmann::json>& getCustomFitParThrow() const;

  bool isUseEigenDecompInFit() const;
  int getNbEnabledEigenParameters() const;
  const TMatrixD* getInvertedEigenVectors() const;
  const TMatrixD* getEigenVectors() const;
  void propagateEigenToOriginal();
  void propagateOriginalToEigen();

  // Misc
  std::string getSummary() const;

  static double toNormalizedParRange(double parRange, const FitParameter& par);
  static double toNormalizedParValue(double parValue, const FitParameter& par);
  static double toRealParValue(double normParValue, const FitParameter& par);
  static double toRealParRange(double normParRange, const FitParameter& par);

  void setMaskedForPropagation(bool maskedForPropagation);

protected:
  void readConfigImpl() override;
  void initializeImpl() override;

  void readParameterDefinitionFile();
  void defineParameters();
  void fillDeltaParameterList();

private:
  // Internals
  std::vector<FitParameter> _parameterList_;
  std::vector<NestedDialTest> _nestedDialList_;
  TDirectory* _saveDir_{nullptr};

  // JSON
  std::string _name_{};
  std::string _parameterDefinitionFilePath_{};
  bool _isEnabled_{};
  bool _maskedForPropagation_{false};
  bool _printDialSetsSummary_{false};
  bool _releaseFixedParametersOnHesse_{false};
  int _nbParameterDefinition_{-1};
  double _nominalStepSize_{std::nan("unset")};
  int _maxNbEigenParameters_{-1};
  double _maxEigenFraction_{1};

  double _globalParameterMinValue_{std::nan("UNSET")};
  double _globalParameterMaxValue_{std::nan("UNSET")};

  std::vector<nlohmann::json> _customFitParThrow_{};

  // Eigen objects
  int _nbEnabledEigen_{0};
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

};


#endif //GUNDAM_FITPARAMETERSET_H
