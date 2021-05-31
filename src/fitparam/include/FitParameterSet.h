//
// Created by Adrien BLANCHET on 21/05/2021.
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

  void setJsonConfig(const nlohmann::json &jsonConfig);

  void initialize();

  std::vector<FitParameter> &getParameterList();

private:
  // User parameters
  nlohmann::json _jsonConfig_;

  // Internals
  bool _isInitialized_{false};
  std::vector<FitParameter> _parameterList_;

  // JSON
  std::string _name_;
  bool _isEnabled_;
  TFile* _covarianceMatrixFile_{nullptr};
  TMatrixDSym* _covarianceMatrix_{nullptr};
  TVectorD* _parameterPriorList_{nullptr};
  TObjArray* _parameterNamesList_{nullptr};

};


#endif //XSLLHFITTER_FITPARAMETERSET_H
