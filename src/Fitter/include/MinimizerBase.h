//
// Created by Clark McGrew on 25/01/2023.
//

#ifndef GUNDAM_MinimizerBase_h
#define GUNDAM_MinimizerBase_h

#include "JsonBaseClass.h"
#include "GenericToolbox.VariablesMonitor.h"

#include <string>

class TDirectory;
class FitterEngine;
class FitParameter;
class Propagator;
class LikelihoodInterface;

/// An (almost) abstract base class for minimizer interfaces.  Classes derived
/// from MinimizerBase are used by the FitterEngine to run different types of
/// fits (primarily a MINUIT based maximization of the likelihood the fits).
/// Classes need to implement two worker methods.  The minimize() method is
/// expected to find the minimim of the LLH function (or Chi-Squared), and the
/// calcErrors() is expected to calculate the covariance of the LLH function.
class MinimizerBase : public JsonBaseClass {

public:
  explicit MinimizerBase(FitterEngine* owner_);

  /// Local RTTI
  [[nodiscard]] virtual std::string getMinimizerTypeName() const { return "MinimizerBase"; };

  /// A pure virtual method that is called by the FitterEngine to find the
  /// minimum of the likelihood, or, in the case of a Bayesian integration find
  /// the posterior distribution.
  virtual void minimize() = 0;

  /// A pure virtual method that is called by the FiterEngine to calculate the
  /// covariance at best fit point.  In the case of a Bayesian integration, it
  /// should either be skipped, or the covariance can be filled using
  /// information from the posterior.
  virtual void calcErrors() = 0;

  /// A pure virtual method that returns true if the fit has converted.
  [[nodiscard]] virtual bool isFitHasConverged() const = 0;

  /// A virtual method that should scan the parameters used by the minimizer.
  /// This provides a view of the parameters seen by the minimizer, which may
  /// be different from the parameters used for the likelihood.  Most
  /// MinimizerBase derived classes should override this method.  If it is not
  /// provided then it will be a no-op.
  virtual void scanParameters(TDirectory* saveDir_);

  /// Set if the calcErrors method should be called by the FitterEngine.
  void setEnablePostFitErrorEval(bool enablePostFitErrorEval_) {_enablePostFitErrorEval_ = enablePostFitErrorEval_;}
  [[nodiscard]] bool isEnablePostFitErrorEval() const {return _enablePostFitErrorEval_;}

protected:
  /// Get a reference to the FitterEngine that owns this minimizer.
  inline FitterEngine& owner() { return *_owner_; }
  [[nodiscard]] inline const FitterEngine& owner() const { return *_owner_; }

  /// Implement the methods required by JsonBaseClass.  These MinimizerBase
  /// methods may be overridden by the derived class, but if overriden, the
  /// derived class must run these instantiations (i.e. call
  /// MinimizerBase::readConfigImpl() and MinimizerBase::initializeImpl in the
  /// respective methods).
  void readConfigImpl() override;
  void initializeImpl() override;

  // Get the propagator being used to calculate the likelihood.  This is a
  // local convenience function to get the propagator from the owner.
  Propagator& getPropagator();
  [[nodiscard]] const Propagator& getPropagator() const;

  // Get the likelihood that should be used by the minimization.  This is a
  // local convenience function to get the likelihood from the owner.
  LikelihoodInterface& getLikelihood();
  [[nodiscard]] const LikelihoodInterface& getLikelihood() const;

  // Get the convergence monitor that is maintained by the likelihood
  // interface.  A local convenience function to get the convergence monitor.
  // The monitor actually lives in the likelihood).
  GenericToolbox::VariablesMonitor &getConvergenceMonitor();

  // Get the vector of parameters being fitted.  This is a local convenience
  // function to get the vector of fit parameter pointers.  The actual vector
  // lives in the likelihood.
  std::vector<FitParameter *> &getMinimizerFitParameterPtr();

  // Print a table of the fitting parameters.
  void printMinimizerFitParameters();

private:
  /// Save a copy of the address of the engine that owns this object.
  FitterEngine* _owner_{nullptr};

  bool _enablePostFitErrorEval_{true};

};

#endif //GUNDAM_MinimizerBase_h

//  A Lesser GNU Public License

//  Copyright (C) 2023 GUNDAM DEVELOPERS

//  This library is free software; you can redistribute it and/or
//  modify it under the terms of the GNU Lesser General Public
//  License as published by the Free Software Foundation; either
//  version 2.1 of the License, or (at your option) any later version.

//  This library is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  Lesser General Public License for more details.

//  You should have received a copy of the GNU Lesser General Public
//  License along with this library; if not, write to the
//
//  Free Software Foundation, Inc.
//  51 Franklin Street, Fifth Floor,
//  Boston, MA  02110-1301  USA

// Local Variables:
// mode:c++
// c-basic-offset:2
// compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
// End:
