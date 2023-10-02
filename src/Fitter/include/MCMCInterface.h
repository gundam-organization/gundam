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

#include <memory>
#include <vector>

// Override TSimpleMCMC.H for how much output to use and where to send it.
#define MCMC_DEBUG_LEVEL 1
#define MCMC_DEBUG(level) if (level <= (MCMC_DEBUG_LEVEL)) LogInfo
#define MCMC_ERROR LogError
#include "TSimpleMCMC.H"

class FitterEngine;

/// Run generate an MCMC doing a Bayesian integration of the likelihood. The
/// parameters are assumed to have a uniform metric, and the priors are
/// defined as part of the propagator (i.e. included in the
/// LikelihoodInterface).
class MCMCInterface : public MinimizerBase {

protected:
  void readConfigImpl() override;
  void initializeImpl() override;

public:
  explicit MCMCInterface(FitterEngine* owner_): MinimizerBase(owner_){}

  /// Local RTTI
  [[nodiscard]] std::string getMinimizerTypeName() const override { return "MCMCInterface"; };

  /// An MCMC doesn't really converge in the sense meant here. This flags success.
  [[nodiscard]] virtual bool isFitHasConverged() const override  {return true;}

  /// Generate a chain.
  void minimize() override;

  /// Don't do anything.  This could calculate the covariance of the chain,
  /// but the concept doesn't really match the ideas of a Bayesian analysis.
  void calcErrors() override;

  /// Scan the parameters.
  void scanParameters(TDirectory* saveDir_) override;


private:
  std::string _algorithmName_{"metropolis"};
  std::string _proposalName_{"adaptive"};
  std::string _outTreeName_{"MCMC"};

  // Define what sort of validity the parameters have to have for a finite
  // likelihood.  The "range" value means that the parameter needs to be
  // between the allowed minimum and maximum values for the parameter.  The
  // "mirror" value means that the parameter needs to be between the mirror
  // bounds too.
  std::string _likelihoodValidity_{"range,mirror,physical"};

  // Save or dump the raw (fitter space) points.  This can save about half the
  // output file space.  About the only time these would ever need to be saved
  // is during a burn-in run when the proposal covariance is being tuned.
  bool _saveRawSteps_{false};

  // The number of burn-in cylces to use.
  int _burninCycles_{0};

  // The number of cycles to dump during burn-in
  int _burninResets_{0};

  // The length of each burn-in cycle
  int _burninLength_{10000};

  // Save the burnin (true) or dump it (false)
  bool _saveBurnin_{true};

  // The number of run cycles to use (each cycle will have _runLength_ steps.
  int _cycles_{1};

  // The number of steps in each run cycle.
  int _steps_{10000};

  // The model for the likelihood takes up quite a bit of space, so it should
  // NOT be saved most of the time.  The _modelStride_ sets the number of
  // steps between when the model is saved to the output file.  The model is a
  // copy of the predicted sample histogram and can be used to calculate the
  // posterior predictive p-value.  The stride should be large(ish) compared
  // to the autocorrelation lag, or zero (if not saving the model).
  int _modelStride_{5000};

  //////////////////////////////////////////
  // Parameters for the adaptive stepper.

  // An input file name that contains a chain.  This causes the previous state
  // to be restored.  If the state is restored, then the burn-in will be
  // skipped.
  std::string _adaptiveRestore_{"none"};

  // An input file name that contains a TH2D with the covariance matrix that
  // will be used by the default proposal distribution.  If it's provided, it
  // will usually be the result of a previous MINUIT asimov fit.
  std::string _adaptiveCovFile_{"none"};

  // The name of a TH2D with a covariance matrix describing the proposal
  // distribution.  The default value is where GUNDAM puts the covariance for
  // from MINUIT.  If decomposition was used during the fit, this will be in
  // the decomposed space.
  std::string _adaptiveCovName_{"FitterEngine/postFit/Hesse/hessian/postfitCovariance_TH2D"};

  // The number of effective trials that the input covariance will count for.
  // This should typically be about 0.5*N^2 where N is the dimension of the
  // covariance.  That works out to the approximate number of function
  // calculations that were used to estimate the covariance.  The default
  // value of zero triggers the interface to make it's own estimate.
  double _adaptiveCovTrials_{500000.0};

  // Freeze the burn-in step after this many cycles.
  int _burninFreezeAfter_{1000000000}; // Never freeze except by request

  // The window to calculate the covariance during burn-in
  int _burninCovWindow_{1000000};

  // The amount of deweighting during burning updates.
  double _burninCovDeweighting_{0.0};

  // The acceptance window during burn-in.
  int _burninWindow_{1000};

  // Freeze the step after this many cycles by fixing the `sigma` parameter.
  // the proposal will be update when the state is restore, the sigma should
  // be adjusted when the state is restore, or the acceptance will generally
  // increase.  The default is one greater than _adaptiveFreezeCorrelations_
  int _adaptiveFreezeAfter_{0};

  // Stop updating the running covariance after this many cycles.
  int _adaptiveFreezeCorrelations_{100000000}; // Default: Never freeze

  // The window to calculate the covariance during normal chains.
  int _adaptiveCovWindow_{1000000};

  // The covariance deweighting while the chain is running.  This should
  // usually be left at zero so the entire chain history is used after an
  // update and more recent points don't get a heavier weight (within the
  // covariance window).
  double _adaptiveCovDeweighting_{0.0};

  // The window used to calculate the current acceptance value.
  int _adaptiveWindow_{1000};

  //////////////////////////////////////////
  // Parameters for the simple stepper

  // The "sigma" of the Gaussian along each axis for the simple step.
  double _simpleSigma_{0.01};

  //////////////////////////////////////////
  // Manage the running of the MCMC

  // The full set of parameter values that are associated with the accepted
  // point
  std::vector<float> _point_;

  // The predicted values from the reweighted MC (histogram) for the last
  // accepted step.
  std::vector<float> _model_;

  // The predicted values from the reweighted MC (histogram) to be saved to
  // the output file. This will often be empty to reduce the size of the
  // output file.
  std::vector<float> _saveModel_;

  // The uncertainty for the predicted values from the reweighted MC
  // (histogram) for the last accepted step.
  std::vector<float> _uncertainty_;

  // The uncertainty for the predicted values from the MC (histogram) to be
  // saved to the output file. This will often be empty to reduce the size of
  // the output file.
  std::vector<float> _saveUncertainty_;

  // The statistical part of the likelihood
  float _llhStatistical_{0.0};

  // The penalty part of the likelihood
  float _llhPenalty_{0.0};

  // Fill the point that will be saved to the output tree with the current set
  // of parameters.  If fillModel is true, this will also fill the model of
  // the expected data for this set of parametrs.
  void fillPoint(bool fillModel = true);

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
  typedef TSimpleMCMC<PrivateProxyLikelihood,TProposeSimpleStep> SimpleStepMCMC;
  void setupAndRunSimpleStep(
    SimpleStepMCMC& mcmc);

  /// The implementation when the adaptive step is used.  This is the default
  /// proposal for TSimpleMCMC, but is also dangerous for "unpleasant"
  /// likelihoods that have a lot of correlations between parameters.
  typedef TSimpleMCMC<PrivateProxyLikelihood,TProposeAdaptiveStep> AdaptiveStepMCMC;
  void setupAndRunAdaptiveStep(AdaptiveStepMCMC& mcmc);

  /////////////////////////////////////////////////////////////////
  // Support routines for the adaptive step.

  /// Restore the state from an input file (and the tree in the file).  This
  /// returns true if the state was restored and the chain is being continued.
  bool adaptiveRestoreState(AdaptiveStepMCMC& mcmc,
                            const std::string& fileName,
                            const std::string& treeName);

  /// Set the covariance of the proposal based on a TH2D histogram in a file.
  /// This returns true if the parameter correlations have been set.
  bool adaptiveLoadProposalCovariance(AdaptiveStepMCMC& mcmc,
                                      Vector& prior,
                                      const std::string& fileName,
                                      const std::string& histName);

  /// Set the default proposal based on the FitParameter values and steps.
  bool adaptiveDefaultProposalCovariance(AdaptiveStepMCMC& mcmc,Vector& prior);
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
