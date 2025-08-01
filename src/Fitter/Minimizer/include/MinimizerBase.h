//
// Created by Clark McGrew on 25/01/2023.
//

#ifndef GUNDAM_MINIMIZER_BASE_H
#define GUNDAM_MINIMIZER_BASE_H

#include "Propagator.h"
#include "ParameterScanner.h"
#include "LikelihoodInterface.h"
#include "Parameter.h"


#include "GenericToolbox.Utils.h"

#include "TDirectory.h"

#include <vector>
#include <string>

/*
  The MinimizerBase is an abstract layer (purely virtual) that provides
  several methods called by the FitterEngine to perform the fit.
  It includes a few user parameters as well as a collection of monitoring
  tools.

  Classes derived from MinimizerBase are used to run different types of
  fits (primarily a MINUIT based maximization of the likelihood the fits).
  Classes need to implement two worker methods.  The minimize() method is
  expected to find the minimim of the LLH function (or Chi-Squared), and the
  calcErrors() is expected to calculate the covariance of the LLH function.
*/

class FitterEngine; // owner

class MinimizerBase : public JsonBaseClass {

protected:
  /// Implement the methods required by JsonBaseClass.  These MinimizerBase
  /// methods may be overridden by the derived class, but if overriden, the
  /// derived class must run these instantiations (i.e. call
  /// MinimizerBase::configureImpl() and MinimizerBase::initializeImpl in the
  /// respective methods).
  void configureImpl() override;
  void initializeImpl() override;

  // Internal struct that hold info about the minimizer state
  struct Monitor{
    bool isEnabled{false};
    bool showParameters{false};
    int maxNbParametersPerLine{15};
    int nbEvalLikelihoodCalls{0};

    std::string minimizerTitle{"unset"};
    std::string stateTitleMonitor{};

    GenericToolbox::Time::AveragedTimer<10> evalLlhTimer{};
    GenericToolbox::Time::AveragedTimer<10> externalTimer{};
    GenericToolbox::Time::AveragedTimer<1> iterationCounterClock{};

    GenericToolbox::VariablesMonitor convergenceMonitor;

    std::unique_ptr<TTree> historyTree{nullptr};
  };

public:
  /// A virtual method that is called by the FitterEngine to find the
  /// minimum of the likelihood, or, in the case of a Bayesian integration find
  /// the posterior distribution.
  virtual void minimize();

  /// A virtual method that is called by the FiterEngine to calculate the
  /// covariance at best fit point.  In the case of a Bayesian integration, it
  /// should either be skipped, or the covariance can be filled using
  /// information from the posterior.
  virtual void calcErrors();

  /// A virtual method that should scan the parameters used by the minimizer.
  /// This provides a view of the parameters seen by the minimizer, which may
  /// be different from the parameters used for the likelihood.  Most
  /// MinimizerBase derived classes should override this method.  If it is not
  /// provided then it will be a no-op.
  virtual void scanParameters( TDirectory* saveDir_ );

  /// The main access is through the evalFit method which takes an array of floating
  /// point values and returns the likelihood. The meaning of the parameters is
  /// defined by the vector of pointers to Parameter returned by the LikelihoodInterface.
  virtual double evalFit( const double* parArray_ );

  // default calcErrors() is not defined
  [[nodiscard]] virtual bool isErrorCalcEnabled() const { return false; }

  // c-tor
  MinimizerBase(){} // for the python interface
  explicit MinimizerBase(FitterEngine* owner_): _owner_(owner_){}

  /// Set if the calcErrors method should be called by the FitterEngine.
  void setDisableCalcError(bool disableCalcError_){ _isEnabledCalcError_ = not disableCalcError_; }

  // const getters
  [[nodiscard]] bool disableCalcError() const{ return not _isEnabledCalcError_; }

  // Get the return status for the fitter.  The return value is specific
  // to the instantiated fitter, but is mostly centered around MINUIT
  [[nodiscard]] int getMinimizerStatus() const { return _minimizerStatus_; }
  void setMinimizerStatus(int s) {_minimizerStatus_ = s;}

  /// Define the type of validity that needs to be required by
  /// hasValidParameterValues.  This accepts a string with the possible values
  /// being:
  ///
  ///  "range"           -- Between the parameter minimum and maximum values.
  ///  "norange"         -- Do not require parameters in the valid range
  ///  "mirror"          -- Between the mirrored values (if parameter has
  ///                       mirroring).
  ///  "nomirror"        -- Do not require parameters in the mirrored range
  ///  "physical"        -- Only physically meaningful values.
  ///  "nophysical"      -- Do not require parameters in the physical range.
  ///
  /// Example: setParameterValidity("range,mirror,physical")
  void setParameterValidity(const std::string& validity);

  // mutable getters
  Monitor& getMonitor(){ return _monitor_; }
  [[nodiscard]] auto& getMonitor() const { return _monitor_; }

  // core
  void printParameters();
  [[nodiscard]] int fetchNbDegreeOfFreedom() const { return getLikelihoodInterface().getNbSampleBins() - _nbFreeParameters_; }


protected:
  /// Get a reference to the FitterEngine that owns this minimizer.
  FitterEngine& getOwner() { return *_owner_; }
  [[nodiscard]] auto& getOwner() const { return *_owner_; }

  // Get the propagator being used to calculate the likelihood.  This is a
  // local convenience function to get the propagator from the owner.
  Propagator& getModelPropagator();
  [[nodiscard]] const Propagator& getModelPropagator() const;

  // Get the parameter scanner object owned by the LikelihoodInterface.
  ParameterScanner& getParameterScanner();
  [[nodiscard]] const ParameterScanner& getParameterScanner() const;

  // Get the likelihood that should be used by the minimization.  This is a
  // local convenience function to get the likelihood from the owner.
  LikelihoodInterface& getLikelihoodInterface();
  [[nodiscard]] const LikelihoodInterface& getLikelihoodInterface() const;

  // Get the vector of parameters being fitted.  This is a local convenience
  // function to get the vector of fit parameter pointers.  The actual vector
  // lives in the likelihood.
  std::vector<Parameter *> &getMinimizerFitParameterPtr(){ return _minimizerParameterPtrList_; }
  const std::vector<Parameter *> &getMinimizerFitParameterPtr() const{ return _minimizerParameterPtrList_; }

  // The minimizer evalFit function won't explicitly check the parameter
  // validity without setting this to true.
  void setCheckParameterValidity(bool c) {_checkParameterValidity_ = c;}

  // Query if a normalized fit space is being used.
  bool useNormalizedFitSpace() const {return _useNormalizedFitSpace_;}
  int* getNbFreeParametersPtr() {return &_nbFreeParameters_;}

protected:
  std::vector<Parameter*> _minimizerParameterPtrList_{};
  Monitor _monitor_{};

private:
  /// Save a copy of the address of the engine that owns this object.
  FitterEngine* _owner_{nullptr};

  int _minimizerStatus_{-1}; // -1: invalid, 0: success, >0: errors
  int _nbFreeParameters_{0};

  // config
  bool _throwOnBadLlh_{false};
  bool _useNormalizedFitSpace_{true};
  bool _checkParameterValidity_{false};
  bool _isEnabledCalcError_{true};

  /// Define what sort of validity the parameters have to have for a finite
  /// likelihood.  The "range" value means that the parameter needs to be
  /// between the allowed minimum and maximum values for the parameter.  The
  /// "mirror" value means that the parameter needs to be between the mirror
  /// bounds too.  The specific definitions (and the default) is defined by
  /// Parameter.
  std::string _likelihoodValidity_{""};

  // output
  bool _writeLlhHistory_{false};

};

#endif //GUNDAM_MINIMIZER_BASE_H

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
// End:
