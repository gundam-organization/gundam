//
// Created by Clark McGrew
//

#ifndef GUNDAM_MCMCInterface_h
#define GUNDAM_MCMCInterface_h

#include "FitParameterSet.h"
#include "MinimizerBase.h"
#include "JsonBaseClass.h"

#include "GenericToolbox.VariablesMonitor.h"
#include "GenericToolbox.CycleTimer.h"

#include "TDirectory.h"
#include "nlohmann/json.hpp"

#include "memory"
#include "vector"

// Override TSimpleMCMC.H for how much output to use and where to send it.
#define MCMC_DEBUG_LEVEL 3
#define MCMC_DEBUG(level) if (level <= (MCMC_DEBUG_LEVEL)) LogInfo
#define MCMC_ERROR (LogInfo << "ERROR: ")
#include "TSimpleMCMC.H"

class FitterEngine;

/// Run generate an MCMC doing a Bayesian integration of the likelihood. The
/// parameters are assumed to have a uniform metric, and the priors are
/// defined as part of the propagator (i.e. included in the
/// LikelihoodInterface).
class MCMCInterface : public MinimizerBase {

public:
  explicit MCMCInterface(FitterEngine* owner_);

  /// A boolean to flag indicating if the MCMC exited successfully.
  [[nodiscard]] virtual bool isFitHasConverged() const override;

  /// Generate a chain.
  void minimize() override;

  /// Don't do anything.  This could calculate the covariance of the chain,
  /// but the concept doesn't really match the ideas of a Bayesian analysis.
  void calcErrors() override;

  /// Scan the parameters.
  void scanParameters(TDirectory* saveDir_) override;

protected:
  void readConfigImpl() override;
  void initializeImpl() override;

private:
  std::string _algorithmName_{"metropolis"};
  std::string _proposalName_{"adaptive"};
  int _stepCount_{1000};
  std::string _outTreeName_{"MCMC"};

  // The number of burn-in cylces to use.
  int _burninCycles_{0};

  // The number of cycles to dump during burn-in
  int _burninResets_{1};

  // The length of each burn-in cycle
  int _burninLength_{0};

  // Save the burnin (true) or dump it (false)
  bool _saveBurnin_{true};

  // The number of run cycles to use (each cycle will have _runLength_ steps.
  int _cycles_{1};

  // The number of steps in each run cycle.
  int _steps_{10000};

  //////////////////////////////////////////
  // Parameters for the adaptive stepper.

  // An input file name that contains a chain.  This causes the previous state
  // to be restored.
  std::string _adaptiveRestore_{""};

  // Freeze the burn-in step after this many cycles.
  int _burninFreezeAfter_{1000000}; // Never freeze except by request

  // The window to calculate the covariance during burn-in
  int _burninCovWindow_{20000};

  // The acceptance window during burn-in.
  int _burninWindow_{3000};

  // Freeze the step after this many cycles.
  int _adaptiveFreezeAfter_{1000000}; // Never freeze except by request

  // The window to calculate the covariance during burn-in
  int _adaptiveCovWindow_{20000};

  // The window used to calculate the current acceptance value.
  int _adaptiveWindow_{10000};

  //////////////////////////////////////////
  // Parameters for the simple stepper

  // The "sigma" of the Gaussian along each axis for the simple step.
  double _simpleSigma_{0.01};

  //////////////////////////////////////////
  // Manage the running of the MCMC

  // The full set of parameter values that are associated with the accepted
  // point
  std::vector<float> _point_;

  // Fill the point that will be saved to the output tree with the current set
  // of parameters.
  void fillPoint();

  /// A local proxy so the likelihood uses a ROOT::Math::Functor provided by
  /// the Likelihood interface.  The functor field MUST by accessing the
  /// likelihood using the TSimpleMCMC<>::GetLogLikelihood() method.  For
  /// example:
  ///
  /// mcmc.GetLogLikelihood().functor = getLikelihood().evalFitFunctor()
  ///
  struct PrivateProxyLikelihood {
    ROOT::Math::Functor* functor;
    std::vector<double> x;
    double operator() (const Vector& point) {
      LogThrowIf(functor == nullptr, "Functor is not initialized");
      // Copy the point into a local vector since there is no guarrantee that
      // the MCMC will be running on a vector of doubles.  This is paranoia
      // coding.
      if (x.size() != point.size()) x.resize(point.size());
      std::copy(point.begin(), point.end(), x.begin());
      // GUNDAM doesn't use the log likelihood it uses negative two times the
      // log likelihood (e.g. approximately chi-squared).  TSimpleMCMC must
      // have the true Log(Likelihood), so make the change here.
      return -0.5*(*functor)(x.data());
    }
  };

  ///////////////////////////////////////////////////////////////////
  // The different MCMC proposals have different idiosyncrasies and need
  // slightly different handling to have the chain become (quickly) stable.
  // Rather than trying to over-generalize (and fight reality), these methods
  // handle the differences.  Notice that the actual "chain" code is very
  // similar.

  /// The implementation with the simple step is used.  This is mostly an
  /// example of how to setup an alternate stepping proposal.
  void setupAndRunSimpleStep(
    TSimpleMCMC<PrivateProxyLikelihood,TProposeSimpleStep>& mcmc);

  /// The implementation when the adaptive step is used.  This is the default
  /// proposal for TSimpleMCMC, but is also dangerous for "unpleasant"
  /// likelihoods that have a lot of correlations between parameters.
  void setupAndRunAdaptiveStep(
    TSimpleMCMC<PrivateProxyLikelihood,TProposeAdaptiveStep>& mcmc);

};
#endif // GUNDAM_MCMCInterface_h

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
