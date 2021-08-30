//
// Created by Nadrino on 21/05/2021.
//

#ifndef XSLLHFITTER_FITPARAMETERSET_H
#define XSLLHFITTER_FITPARAMETERSET_H

#include "vector"
#include "string"

#include "json.hpp"
#include "TMatrixDSym.h"
#include "TVectorT.h"
#include "TFile.h"
#include "TObjArray.h"
#include "TVectorT.h"
#include "TMatrixDSymEigen.h"

#include "Logger.h"

#include "FitParameter.h"


/*
 * FitParameterSet is a class which aims at handling a set of parameters bond together with a covariance matrix
 * User parameters:
 * - Covariance matrix (dim N)
 * - N Fit Parameters (handing dials)
 *
 * */

class FitParameterSet {

public:
  FitParameterSet();
  virtual ~FitParameterSet();

  void reset();

  // Setters
  void setJsonConfig(const nlohmann::json &jsonConfig);

  // Init
  void initialize();

  // Getters
  bool isEnabled() const;
  const std::string &getName() const;
  std::vector<FitParameter> &getParameterList();
  const std::vector<FitParameter> &getParameterList() const;
  TMatrixDSym *getOriginalCovarianceMatrix() const;

  // Core
  size_t getNbParameters() const;
  FitParameter& getFitParameter( size_t iPar_ );
  double getChi2() const;

  // Eigen decomposition
  bool isUseEigenDecompInFit() const;
  int getNbEnabledEigenParameters() const;
  double getEigenParameter(int iPar_) const;
  double getEigenSigma(int iPar_) const;
  void setEigenParameter( int iPar_, double value_ );
  const TMatrixD* getInvertedEigenVectors() const;
  const TMatrixD* getEigenVectors() const;
  void propagateEigenToOriginal();
  void propagateOriginalToEigen();

  // Misc
  std::string getSummary() const;

protected:
  void passIfInitialized(const std::string& methodName_) const;

  void readCovarianceMatrix();

private:
  // User parameters
  nlohmann::json _jsonConfig_;

  // Internals
  bool _isInitialized_{false};
  std::vector<FitParameter> _parameterList_;

  // JSON
  std::string _name_;
  bool _isEnabled_{};
  double _maxEigenFraction_{1};
  TFile* _covarianceMatrixFile_{nullptr};
  TMatrixDSym* _originalCovarianceMatrix_{nullptr};
  TVectorD* _parameterPriorList_{nullptr};
  TObjArray* _parameterNamesList_{nullptr};


  int _nbEnabledEigen_{0};
  bool _useEigenDecompInFit_{false};
  std::shared_ptr<TMatrixDSymEigen> _eigenDecomp_{nullptr};
  std::shared_ptr<TVectorD> _eigenValues_{nullptr};
  std::shared_ptr<TMatrixD> _eigenVectors_{nullptr};
  std::shared_ptr<TMatrixD> _invertedEigenVectors_{nullptr};
  std::shared_ptr<TMatrixDSym> _projectorMatrix_{nullptr};
  std::shared_ptr<TMatrixDSym> _inverseCovarianceMatrix_{nullptr};
  std::shared_ptr<TMatrixDSym> _originalCorrelationMatrix_{nullptr};
  std::shared_ptr<TMatrixDSym> _effectiveCovarianceMatrix_{nullptr};
  std::shared_ptr<TMatrixDSym> _effectiveCorrelationMatrix_{nullptr};

  std::shared_ptr<TVectorD> _eigenParValues_;
  std::shared_ptr<TVectorD> _originalParValues_;
  std::shared_ptr<TVectorD> _eigenParPriorValues_;

};


#endif //XSLLHFITTER_FITPARAMETERSET_H
