//
// Created by Clark McGrew
//

#ifndef GUNDAM_SIMPLE_MCMC_H
#define GUNDAM_SIMPLE_MCMC_H

#include "ParameterSet.h"
#include "MinimizerBase.h"
#include "FitterEngine.h"

#include "ConfigurationValue.h"

#include "GenericToolbox.Utils.h"
#include "GenericToolbox.Time.h"

#include "Math/Functor.h"
#include "TDirectory.h"

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
class SimpleMcmc : public MinimizerBase {
  friend class SimpleMcmcSequencer;

protected:
  void configureImpl() override;
  void initializeImpl() override;

public:
  // overrides
  void minimize() override; /// Generate a chain.

  // c-tor
  explicit SimpleMcmc(FitterEngine* owner_);

  // core

  /// Same as `evalFit` but also check that all the parameters are within
  /// the allowed ranges.  If a parameter is out of range, then return an
  /// "infinite" likelihood.
  double evalFitValid( const double* parArray_ );

  /// Check that the parameters for the last time the propagator was used are
  /// all within the allowed ranges.
  [[nodiscard]] bool hasValidParameterValues() const;

private:

  /// The basic algorithm used.  The only implemented algorithm is the
  /// metropolis step, but this may be extended to support Gibbs and other
  /// algorithms (Gibbs needs to know the conditional probabilities, which we
  /// will probably never know).
  std::string _algorithmName_{"metropolis"};

  /// The type of proposal used by the MCMC algorithm. The currently available
  /// proposals are "adaptive", and "fixed".
  std::string _proposalName_{"not-set"};

  /// The name of the output tree in the file.  While it can be changed, you
  /// probably shouldn't touch this.
  std::string _outTreeName_{"MCMC"};

  /// Choose the start point of MCMC is a random point (true) or the prior point
  /// (false).
  ConfigurationValue<bool> _randomStart_;

  /// Save or dump the raw (fitter space) points.  This can save about half the
  /// output file space.  About the only time these would ever need to be saved
  /// is during a burn-in run when the proposal covariance is being tuned.
  bool _saveRawSteps_{false};

  /// The model for the likelihood takes up quite a bit of space, so it should
  /// NOT be saved most of the time.  The _modelStride_ sets the number of
  /// steps between when the model is saved to the output file.  The model is a
  /// copy of the predicted sample histogram and can be used to calculate the
  /// posterior predictive p-value.  The stride should be large(ish) compared
  /// to the autocorrelation lag, or zero (if not saving the model).
  ConfigurationValue<int> _modelStride_{5000};

  /// The number of run cycles to use (each cycle will have _steps_ steps.
  int _cycles_{1};

  /// The sequence used for the MCMC.
  std::string _sequence_;

  /// The number of burn-in cycles to use.
  int _burninCycles_{0};

  /// The sequence used for the burn-in stage of the MCMC.  This is only used
  /// when the state has not been restored from a previous chain, and the
  /// number of burn-in cycles has been set to more than zero.
  std::string _burninSequence_;

  /// The number of steps in each run cycle.
  ConfigurationValue<int> _steps_{10000};

  /// Flag for if the steps in the cycle should be saved.
  ConfigurationValue<bool> _saveSteps_{true};

  /// The length of each burn-in cycle
  int _burninSteps_{10000};

  /// Save the burnin (true) or dump it (false)
  bool _saveBurnin_{true};

  /// The number of cycles to dump during burn-in
  int _burninResets_{0};

  //////////////////////////////////////////
  // Parameters for the adaptive stepper.

  /// An input file name that contains a chain.  This causes the previous state
  /// to be restored.  If the state is restored, then the burn-in will be
  /// skipped.
  std::string _adaptiveRestore_{"none"};

  /// An input file name that contains a TH2D with the covariance matrix that
  /// will be used by the default proposal distribution.  If it's provided, it
  /// will usually be the result of a previous MINUIT asimov fit.
  std::string _adaptiveCovFile_{"none"};

  /// The name of a TH2D with a covariance matrix describing the proposal
  /// distribution.  The default value is where GUNDAM puts the covariance for
  /// from MINUIT.  If decomposition was used during the fit, this will be in
  /// the decomposed space.
  std::string _adaptiveCovName_{
    "FitterEngine/postFit/Hesse/hessian/postfitCovariance_TH2D"
  };

  /// The number of effective trials that the input covariance will count for.
  /// This should typically be about 0.5*N^2 where N is the dimension of the
  /// covariance.  That works out to the approximate number of function
  /// calculations that were used to estimate the covariance.  The default
  /// value of zero triggers the interface to make it's own estimate.
  double _adaptiveCovTrials_{500000.0};

  /// Freeze the burn-in step after this many cycles.
  int _burninFreezeAfter_{1000000000}; // Never freeze except by request

  /// The window to calculate the covariance during burn-in
  int _burninCovWindow_{1000000};

  /// The amount of deweighting during burning updates.
  double _burninCovDeweighting_{0.0};

  /// The acceptance window during burn-in.
  int _burninWindow_{1000};

  /// Freeze the step after this many cycles by fixing the `sigma` parameter.
  /// the proposal will be update when the state is restore, the sigma should
  /// be adjusted when the state is restore, or the acceptance will generally
  /// increase.  The default is one greater than _adaptiveFreezeCorrelations_
  int _adaptiveFreezeAfter_{0};

  /// Stop updating the running covariance after this many cycles.
  int _adaptiveFreezeCorrelationsAfter_{100000000}; // Default: Never freeze

  /// Control whether the step is updated during an adaptive cycle.
  ConfigurationValue<bool> _adaptiveFreezeStep_{false};

  /// Control whether the covariance is updated during an adaptive cycle.
  ConfigurationValue<bool> _adaptiveFreezeCov_{false};

  /// Control whether the covariance should be reset to the value set during
  /// configuration.
  ConfigurationValue<bool> _adaptiveResetCov_{false};

  /// The window to calculate the covariance during normal chains.
  ConfigurationValue<int> _adaptiveCovWindow_{1000000};

  /// The covariance deweighting while the chain is running.  This should
  /// usually be left at zero so the entire chain history is used after an
  /// update and more recent points don't get a heavier weight (within the
  /// covariance window).
  ConfigurationValue<double> _adaptiveCovDeweighting_{0.0};

  /// The window used to calculate the current acceptance value.
  ConfigurationValue<int> _adaptiveWindow_{1000};

  /// The acceptance for steps to be used.  The valid values are 0)
  /// metropolis, 1) downhill-only, and 2) accept every step.
  ConfigurationValue<int> _adaptiveAcceptanceAlgorithm_{0};

  //////////////////////////////////////////
  // Parameters for the simple stepper

  /// The "sigma" of the Gaussian along each axis for the simple step.
  double _simpleSigma_{0.01};

  //////////////////////////////////////////
  // Manage the running of the MCMC

  /// The full set of parameter values that are associated with the accepted
  /// point
  std::vector<float> _point_;

  /// The predicted values from the reweighted MC (histogram) for the last
  /// accepted step.
  std::vector<float> _model_;

  /// The predicted values from the reweighted MC (histogram) to be saved to
  /// the output file. This will often be empty to reduce the size of the
  /// output file.
  std::vector<float> _saveModel_;

  /// The uncertainty for the predicted values from the reweighted MC
  /// (histogram) for the last accepted step.
  std::vector<float> _uncertainty_;

  /// The uncertainty for the predicted values from the MC (histogram) to be
  /// saved to the output file. This will often be empty to reduce the size of
  /// the output file.
  std::vector<float> _saveUncertainty_;

  /// The statistical part of the likelihood
  float _llhStatistical_{0.0};

  /// The penalty part of the likelihood
  float _llhPenalty_{0.0};

  /// Fill the point that will be saved to the output tree with the current set
  /// of parameters.  If fillModel is true, this will also fill the model of
  /// the expected data for this set of parametrs.
  void fillPoint(bool fillModel = true);

  /// A local proxy so the likelihood uses a ROOT::Math::Functor provided by
  /// the Likelihood interface.  The functor field MUST by accessing the
  /// likelihood using the TSimpleMCMC<>::GetLogLikelihood() method.  For
  /// example:
  ///
  /// mcmc.GetLogLikelihood().functor = getLikelihoodInterface().getFunctor()
  ///
  struct PrivateProxyLikelihood {
    /// A functor that can be called by Minuit or anybody else.  This usually
    /// wraps evalFit.
    std::unique_ptr<ROOT::Math::Functor> functor{};
    /// An internal variable to copy the TSimpleMCMC point into.  This matches
    /// what the functor is expecting.
    std::vector<double> x;
    /// This is what TSimpleMCMC will see.
    double operator() (const sMCMC::Vector& point) {
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

  /// Restore the configuration to what was setup in the YAML file
  void restoreConfiguration();

  ///////////////////////////////////////////////////////////////////
  // Handle the different stepper special cases.

  ///////////////////////////////////////////////////////////////////
  // Fixed Step Support
  ///////////////////////////////////////////////////////////////////

  /// TSimpleMCMC class for the FixedStepMCMC.
  typedef sMCMC::TSimpleMCMC<PrivateProxyLikelihood,
                             sMCMC::TProposeSimpleStep> FixedStepMCMC;

  /// The implementation with a fixed step is used.  This is mostly an example
  /// of how to setup an alternate stepping proposal. The different MCMC
  /// proposals have different idiosyncrasies and need slightly different
  /// handling to have the chain become (quickly) stable.  Rather than
  /// trying to over-generalize (and fight reality), these methods handle
  /// the differences.  Notice that the actual "chain" code is very
  /// similar.
  void fixedSetupAndRun(FixedStepMCMC& mcmc);

  ///////////////////////////////////////////////////////////////////
  // Adaptive Step Support
  ///////////////////////////////////////////////////////////////////

  /// TSimpleMCMC class for the AdaptiveStepMCMC.
  typedef sMCMC::TSimpleMCMC<PrivateProxyLikelihood,
                             sMCMC::TProposeAdaptiveStep> AdaptiveStepMCMC;
  AdaptiveStepMCMC* _adaptiveMCMC_{nullptr};

  /// The implementation when the adaptive step is used.  This is the default
  /// proposal for TSimpleMCMC, but is also dangerous for "unpleasant"
  /// likelihoods that have a lot of correlations between parameters. The
  /// different MCMC proposals have different idiosyncrasies and need slightly
  /// different handling to have the chain become (quickly) stable.  Rather
  /// than trying to over-generalize (and fight reality), these methods handle
  /// the differences.  Notice that the actual "chain" code is very similar.
  void adaptiveSetupAndRun(AdaptiveStepMCMC& mcmc);

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
                                      sMCMC::Vector& prior,
                                      const std::string& fileName,
                                      const std::string& histName);

  /// Set the default proposal based on the FitParameter values and steps.
  bool adaptiveDefaultProposalCovariance(AdaptiveStepMCMC& mcmc,
                                         sMCMC::Vector& prior);

  /// Run an adaptive cycle with the current configuration.
  bool adaptiveRunCycle(AdaptiveStepMCMC& mcmc,
                        std::string chainName,
                        int chainId);

  /// Start the MCMC and setup the prior.  This creates the prior using
  /// adaptiveMakePrior() which will randomize the prior if requested.
  void adaptiveStart(AdaptiveStepMCMC& mcmc,
                     sMCMC::Vector& prior,
                     bool randomize);

  /// Start the MCMC and setup the prior.  This creates the prior using
  /// adaptiveMakePrior() which will randomize the prior if requested.
  void adaptiveRestart(AdaptiveStepMCMC& mcmc,
                       bool randomize);

  /// Create a new point for the prior and then randomizes the starting
  /// point if `randomize` is true.
  void adaptiveMakePrior(AdaptiveStepMCMC& mcmc,
                         sMCMC::Vector& prior,
                         bool randomize);

};

/// A class to implement the MCMC sequencer used to define the order of
/// operations.  This implements the functions that are available for the
/// sequence: and burninSequence: directives.  None of these methods should be
/// called from C++ code.
class SimpleMcmcSequencer {
  friend SimpleMcmc;

private:
  FitterEngine* _engine_{nullptr};
  bool _validState_{false};

  /// Make the engine available to the sequencer.
  void setEngine(FitterEngine* engine);

  /// Get the MCMC class.  This will throw if there is a problem
  SimpleMcmc& Owner();

  /// Used by SimpleMcmc to flag that it is (or isn't) in a state where the
  /// SimpleMcmcSequencer commands can be used.
  void SetSequencerState(bool s);

public:
  SimpleMcmcSequencer();
  virtual ~SimpleMcmcSequencer();

  /// Start the MCMC from a random position.
  void RandomStart(bool v);

  /// Override the number of steps in the next cycle.  This is reset to the
  /// default value after the cycle is run
  void Steps(int v);

  /// Set to false to prevent the steps from being saved during the next
  /// cycle.  This is reset to true after the cycle is run
  void SaveSteps(bool v);

  /// Override the number of steps between saving the model in the next cycle.
  /// This is reset to the default value after the cycle is run.
  void ModelStride(int v);

  /// Freeze the step size for the next cycle.
  void FreezeStep(bool v);

  /// Freeze the covariance for the next cycle.
  void FreezeCovariance(bool v);

  /// Reset the covariance before the next cycle
  void ResetCovariance(bool v);

  /// Set the covariance averaging window for the next cycle.
  void CovarianceWindow(int v);

  /// Set the covariance deweighting for the next cycle.
  void CovarianceDeweighting(double v);

  /// Set the window used to calculate the acceptance during the next cycle.
  void AcceptanceWindow(int v);

  /// Low level control of how the step proposal works for the next cycle.
  void AcceptanceAlgorithm(int v);

  /// The number of burn-in cycles
  int Burnin();

  /// The number of cycles.
  int Cycles();

  /// The number of steps in a cycle.
  int Steps();

  /// The total number of trials that have been run in the chain.  This is the
  /// total number of steps that have been taken over the history of the
  /// chain, and may include steps taken before the current job was run.
  int Trials();

  /// Run a cycle with the current parameters.
  void RunCycle(std::string name, int id);

  /// Choose a new random starting point for the next chain.  The random
  /// starting point will be chose based on the prior.  This is roughly like
  /// starting a new chain and is useful during burn-in to build a better
  /// starting covariance.  This should usually be followed by a cycle with
  /// `AcceptanceAlgorithm(1)` (a downhill only step to get a point in the
  /// bulk of the posterior distribution) and `FreezeCovariance(true)` to
  /// return to a point in the bulk of the posterior distribution before
  /// updating the covariance.  An example of the usage can be seen in the
  /// `200HorrifyingMCMCCov-config.yaml` used during testing.
  void Restart();

  /// Force the step size in standard deviations.  This can help during
  /// burnin, but should be used with extreme caution.
  void SetSigma(double s);
};

#endif // GUNDAM_SIMPLE_MCMC_H

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
// compile-command:"$(git rev-parse --show-toplevel)/cmake/scripts/gundam-build.sh"
// End:
