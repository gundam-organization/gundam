//
// Created by Clark McGrew on 26/01/2023.
//

#include "SimpleMcmc.h"
#include "LikelihoodInterface.h"
#include "FitterEngine.h"
#include "GundamGlobals.h"
#include "GundamUtils.h"

#include "GenericToolbox.Root.h"
#include "Logger.h"

#include "TROOT.h"
#include "TPython.h"

#include <locale>
#include <string>

// Add a mostly hidden global declaration of the SimpleMcmcmSequencer class
// so that it can be used to define the sequence used for the MCMC.
SimpleMcmcSequencer gMCMC;

SimpleMcmc::SimpleMcmc(FitterEngine* owner_): MinimizerBase(owner_) {
  gMCMC.setEngine(owner_);
}

void SimpleMcmc::configureImpl(){
  setCheckParameterValidity(true);
  this->MinimizerBase::configureImpl();
  LogInfo << "Configure MCMC: " << _config_ << std::endl;

  // The type of algorithm to be using.  It should be left at the default
  // value (metropolis is the only supported MCMC algorithm right now).
  GenericToolbox::Json::fillValue(_config_, _algorithmName_, "algorithm");

  // The step proposal algorithm.
  GenericToolbox::Json::fillValue(_config_, _proposalName_, "proposal");

  // The name of the MCMC result tree in the output file.  This doesn't need
  // to be changed.  Generally, leave it alone.
  GenericToolbox::Json::fillValue(_config_, _outTreeName_, "mcmcOutputTree");

  // Define what sort of validity the parameters have to have for a finite
  // likelihood.  The "range" value means that the parameter needs to be
  // between the allowed minimum and maximum values for the parameter.  The
  // "mirror" value means that the parameter needs to be between the mirror
  // bounds too.  The "physical" value means that the parameter has to be in
  // the physically allowed range.
  GenericToolbox::Json::fillValue(_config_, _likelihoodValidity_, "likelihoodValidity");

  //Set whether MCMC chain start from a random point or the prior point.
  {
    bool tmp{_randomStart_};
    GenericToolbox::Json::fillValue(_config_, tmp, "randomStart");
    _randomStart_.set(tmp);
  }

  // Set whether the raw step should be saved, or only the step translated
  // into the likelihood space.
  GenericToolbox::Json::fillValue(_config_, _saveRawSteps_, "saveRawSteps");

  // The number of steps between when the predicted sample histograms should
  // be saved into the output file.  The sample histograms can then be used
  // with the parameterSampleData vector to calculate the PPP for the chain.
  // Note that the model is what the MC is predicting, so for a PPP
  // calculation, the data will need to be fluctuated around the prediction.
  {
    int tmp{_modelStride_};
    GenericToolbox::Json::fillValue(_config_, tmp, "modelSaveStride");
    _modelStride_.set(tmp);
  }

  // Get the MCMC chain parameters to be used during burn-in.  The burn-in will
  // be skipped if the state has been restored from a file.  The burn-in can be
  // skipped in favor of discarding the initial parts of the MCMC chain
  // (usually a better option).  A run is broken into "mini-Chains" called a
  // "cycle" where the posterior covariance information is updated after each
  // mini-chain.  Each cycle will have "steps" steps.
  GenericToolbox::Json::fillValue(_config_, _burninCycles_, "burninCycles");

  // The number of steps to run in each burn in cycle
  GenericToolbox::Json::fillValue(_config_, _burninSteps_, "burninSteps");

  // If this is set to false, the burn-in steps will not be saved to disk.
  // This should usually be true since it lets you see the progress of the
  // burn-in.
  GenericToolbox::Json::fillValue(_config_, _saveBurnin_, "saveBurnin");

  // Get the sequence for the burn-in.
  _burninSequence_
    = R"cxx(for (int chain = 0; chain < gMCMC.Burning(); ++chain) {
      gMCMC.RunChain("Burn-in chain", chain);})cxx";
  GenericToolbox::Json::fillValue(_config_,_burninSequence_,"burninSequence");

  // Get the MCMC chain parameters.  A run is broken into "mini-Chains"
  // called a "cycle" where the posterior covariance information is updated
  // after each mini-chain.
  GenericToolbox::Json::fillValue(_config_, _cycles_, "cycles");

  // Get the number of steps to be taken in each "mini-chain".
  {
    int tmp{_steps_};
    GenericToolbox::Json::fillValue(_config_, tmp, "steps");
    _steps_.set(tmp);
  }

  // Get the sequence for the chain.
  _sequence_
    = R"cxx(for (int chain = 0; chain < gMCMC.Cycles(); ++chain) {
      gMCMC.RunCycle("Chain", chain);})cxx";
  GenericToolbox::Json::fillValue(_config_,_sequence_,"sequence");

  ///////////////////////////////////////////////////////////////
  // Get parameters for the adaptive proposal.

  // Set the window to calculate the current acceptance value over during
  // burn-in.  If this is set to short, the step size will fluctuate.  If this
  // is set to long, the step size won't be adjusted to match the target
  // acceptance.  Make this very large to lock the step size.
  GenericToolbox::Json::fillValue(_config_, _burninCovWindow_, "burninCovWindow");

  // The covariance deweighting during burn-in.  This should usually be left
  // at the default value.  This sets how much extra influence new points
  // should have on the covariance.
  GenericToolbox::Json::fillValue(_config_, _burninCovDeweighting_, "burninCovDeweighting");

  // The number of times that the burn-in state will be reset.  If this is
  // zero, then there are no resets (one means reset after the first cycle,
  // &c).  Resets are sometimes needed if the initial conditions are far from
  // the main probability in the posterior and the "best fit" parameters need
  // to be found.
  GenericToolbox::Json::fillValue(_config_, _burninResets_, "burninResets");

  // Freeze the step size after this many burn-in chains.  This stops
  // adaptively adjusting the step size.
  GenericToolbox::Json::fillValue(_config_, _burninFreezeAfter_, "burninFreezeAfter");

  // Set the window to calculate the current acceptance value over during
  // burn-in.  If this is set to short, the step size will fluctuate.  If this
  // is set to long, the step size won't be adjusted to match the target
  // acceptance.  Make this very large to lock the step size.
  GenericToolbox::Json::fillValue(_config_, _burninWindow_, "burninWindow");

  // Set the name of a file containing an existing Markov chain to be
  // extended.  If this is set, then the burn-in will be skipped.
  //
  // The value is settable from the command line (setting from the command
  // line is the better option) using the override option
  //
  // "-O /fitterEngineConfig/minimizerConfig/adaptiveRestore=<filename>"
  //
  // If restore is going to be used, the adaptiveRestore value must exist in
  // the configuration file (with a NULL value)
  GenericToolbox::Json::fillValue(_config_, _adaptiveRestore_, "adaptiveRestore");
  _adaptiveRestore_ = GenericToolbox::expandEnvironmentVariables(_adaptiveRestore_);

  // Set the name of a file containing a TH2D that describes the covariance of
  // the proposal distribution.
  //
  // The value is settable from the command line (setting from the command
  // line is the better option) using the override option
  //
  // "-O /fitterEngineConfig/mcmcConfig/adaptiveCovFile=<filename>"
  //
  // For this to be used the adaptiveCovFile value must exist in the
  // configuration file (with a value of "none" to be ignored, or a default
  // value if a default should be loaded)
  GenericToolbox::Json::fillValue(_config_, _adaptiveCovFile_, "adaptiveCovFile");
  _adaptiveCovFile_ = GenericToolbox::expandEnvironmentVariables(_adaptiveCovFile_);

  // Set the name of a ROOT TH2D that will be used as the covariance of the
  // step proposal.  If adaptiveCovFile is not set, or has a value of "none",
  // this will be ignored.
  GenericToolbox::Json::fillValue(_config_, _adaptiveCovName_, "adaptiveCovName");

  // Get the effective number of trials for a proposal covariance that is
  // being read from a file. This should typically be about 0.5*N^2 where N is
  // the dimension of the covariance.  That works out to the approximate
  // number of function calculations that were used to estimate the
  // covariance.  The default value of zero triggers the interface to make
  // it's own estimate.
  GenericToolbox::Json::fillValue(_config_, _adaptiveCovTrials_, "adaptiveCovTrials");

  // Set the window to calculate the current covariance value over.  If this
  // is set to short, the covariance will not sample the entire posterior.
  // Generally, the window should be long compared to the number of steps
  // required to get to an uncorrelated point.
  {
    int tmp{_adaptiveCovWindow_};
    GenericToolbox::Json::fillValue(
      _config_, tmp, {{"covarianceWindow"}, {"adaptiveCovWindow"}});
    _adaptiveCovWindow_.set(tmp);
  }

  // The covariance deweighting while the chain is running.  This should
  // usually be left at zero so the entire chain history is used after an
  // update and more recent points don't get a heavier weight (within the
  // covariance window).
  {
    double tmp{_adaptiveCovDeweighting_};
    GenericToolbox::Json::fillValue(
      _config_, tmp, {{"covarianceDeweighting"}, {"adaptiveCovDeweighting"}});
    _adaptiveCovDeweighting_.set(tmp);
  }

  // Stop updating the correlations between the steps after this many cycles.
  // If this is negative, the step size is never updated.  This freeze the
  // running covariance calculation.
  GenericToolbox::Json::fillValue(_config_, _adaptiveFreezeCorrelationsAfter_, "adaptiveFreezeCorrelations");

  // Stop updating the step length after this many cycles.  If this is
  // negative, the step size is never updated.  Take the default from one more
  // than the number of steps to freeze the correlations.  An explicit value
  // in the config file will always override the default.
  _adaptiveFreezeAfter_ = _adaptiveFreezeCorrelationsAfter_+1;
  GenericToolbox::Json::fillValue(_config_, _adaptiveFreezeAfter_, "adaptiveFreezeLength");

  // Set the window to calculate the current acceptance value over.  If this
  // is set to short, the step size will fluctuate a lot.  If this is set to
  // long, the step size won't be adjusted to match the target acceptance.
  // Make this very large effectively locks the step size.
  {
    int tmp{_adaptiveWindow_};
    GenericToolbox::Json::fillValue(_config_, tmp,
                                    {{"acceptanceWindow"}, {"adaptiveWindow"}});
    _adaptiveWindow_.set(tmp);
  }

  ///////////////////////////////////////////////////////////////
  // Get parameters for the simple proposal.

  // Set the step size for the simple proposal.
  GenericToolbox::Json::fillValue(_config_, _simpleSigma_, "fixedSigma");
}

void SimpleMcmc::initializeImpl(){
  MinimizerBase::initializeImpl();
  LogInfo << "Initializing the MCMC Integration..." << std::endl;

  // Set how the parameter values are handled (outside of different validity ranges)
  this->setParameterValidity( _likelihoodValidity_ );
}

/// Restore the configuration to the default.
void SimpleMcmc::restoreConfiguration() {
  _randomStart_.restore();
  _steps_.restore();
  _modelStride_.restore();
  _adaptiveFreezeStep_.restore();
  _adaptiveFreezeCov_.restore();
  _adaptiveResetCov_.restore();
  _adaptiveCovWindow_.restore();
  _adaptiveCovDeweighting_.restore();
  _adaptiveWindow_.restore();
  _adaptiveAcceptanceAlgorithm_.restore();
}

/// Copy the current parameter values to the tree.
void SimpleMcmc::fillPoint( bool fillModel) {
  int count = 0;
  for (const ParameterSet& parSet: getModelPropagator().getParametersManager().getParameterSetsList()) {
    for (const Parameter& iPar : parSet.getParameterList()) {
      if (count >= _point_.size()) {
        LogWarning << "Point out of range " << _point_.size() << " " << count << std::endl;
        continue;
      }
      _point_[count++] = iPar.getParameterValue();
    }
  }
  _llhStatistical_ = getLikelihoodInterface().getLastStatLikelihood();
  _llhPenalty_ = getLikelihoodInterface().getLastPenaltyLikelihood();
  // Watch this next line for speed.  It DOES call the "float" destructor
  // which is trivial (i.e. a noop).  In gcc, all it does is reset the vector
  // "size".
  _model_.clear();
  _uncertainty_.clear();
  _saveModel_.clear();
  _saveUncertainty_.clear();
  if (not fillModel) return;
  for (const Sample& sample
      : getModelPropagator().getSampleSet().getSampleList()) {
    auto& hist = sample.getHistogram();
    /// Adrien: isn't it a bug?? i from 1 to nBins ? Should be from 0 ? or until nBins+1 ?
    for (int i = 1; i < hist.getNbBins(); ++i) {
      _model_.push_back( hist.getBinContentList()[i-1].sumWeights );
      _uncertainty_.push_back( hist.getBinContentList()[i-1].sqrtSumSqWeights );
    }
  }
}

bool SimpleMcmc::adaptiveRestoreState( AdaptiveStepMCMC& mcmc,
                                         const std::string& fileName,
                                         const std::string& treeName) {

  // No filename so, no restoration.
  if (fileName.empty()) return false;

  // Check if the filename is "none", and exit if it is.
  {
    std::string tmp{fileName};
    std::use_facet<std::ctype<std::string::value_type>>(std::locale())
      .tolower(&tmp[0],&tmp[0]+tmp.size());
    if (tmp == "none") return false;
  }

  // Open the file with the state.
  TFile* saveFile = gFile;
  std::unique_ptr<TFile> restoreFile
      (new TFile(fileName.c_str(), "old"));
  if (!restoreFile || !restoreFile->IsOpen()) {
    LogInfo << "File to restore was not found: "
            << fileName << std::endl;
    throw std::runtime_error("Old state file not open");
  }

  // Get the tree with the state.
  TTree* restoreTree = dynamic_cast<TTree*>(restoreFile->Get(treeName.c_str()));
  if (!restoreTree) {
    LogInfo << "Tree to restore state is not found in "
            << restoreFile->GetName()
            << treeName << std::endl;
    throw std::runtime_error("Old state tree not open");
  }

  LogInfo << "Restore the state of a previous MCMC and extend: "
          << restoreFile->GetName() << std::endl;

  // Load the old state from the tree.
  mcmc.Restore(restoreTree);
  LogInfo << "State Restored" << std::endl;

  // Set the deweighting to zero so set covariance is directly used.
  mcmc.GetProposeStep().SetCovarianceUpdateDeweighting(0.0);

  // Restore the original file status.
  gFile = saveFile;
  gFile->cd((std::string(getOwner().getSaveDir()->GetName())+"/fit").c_str());

  return true;
}
bool SimpleMcmc::adaptiveDefaultProposalCovariance( AdaptiveStepMCMC& mcmc,
                                                      sMCMC::Vector& prior) {

  /// Set the diagonal elements for the parameters.
  int count0 = 0;
  for (const Parameter* par : getMinimizerFitParameterPtr() ) {
    ++count0;
    if( useNormalizedFitSpace() ){
      // Changing the boundaries, change the value/step size?
      double step
          = ParameterSet::toNormalizedParRange(
              par->getStepSize(), *par);
      mcmc.GetProposeStep().SetGaussian(count0-1,step);
    }
    else if (par->getPriorType() == Parameter::PriorType::Flat) {
      // Gundam uses flat to mean "free", so this doesn't use a Uniform
      // step between bounds.
      double step = std::min(par->getStepSize(),par->getStdDevValue());
      if (step <= std::abs(1E-10*par->getPriorValue())) {
        step = std::max(par->getStepSize(),par->getStdDevValue());
      }
      step /= std::sqrt(getMinimizerFitParameterPtr().size());
      mcmc.GetProposeStep().SetGaussian(count0-1,step);
    }
    else {
      double step = par->getStdDevValue();
      mcmc.GetProposeStep().SetGaussian(count0-1,step);
    }
  }

  mcmc.GetProposeStep().ResetCorrelations();

  // Set up the correlations in the priors.
  int count1 = 0;
  for (const Parameter* par1 : getMinimizerFitParameterPtr() ) {
    ++count1;
    const ParameterSet* set1 = par1->getOwner();
    if (!set1) {
      LogInfo << "Parameter set reference is not defined for"
              << " " << par1->getName()
              << std::endl;
      continue;
    }
    if ( set1->isEnableEigenDecomp()) {
      continue;
    }

    int count2 = 0;
    for (const Parameter* par2 : getMinimizerFitParameterPtr()) {
      ++count2;
      const ParameterSet* set2 = par2->getOwner();
      if (!set2) {
        LogInfo << "Parameter set reference is not defined for"
                << " " << par1->getName()
                << std::endl;
        continue;
      }
      if ( set2->isEnableEigenDecomp()) {
        continue;
      }

      if (set1 != set2) continue;
      int in1 = par1->getParameterIndex();
      int in2 = par2->getParameterIndex();
      if (in2 <= in1) continue;
      const std::shared_ptr<TMatrixDSym>& corr
          = GenericToolbox::toCorrelationMatrix(set1->getPriorCovarianceMatrix());
      if (!corr) continue;
      double correlation = (*corr)(in1,in2);
      // Don't impose very small correlations, let them be discovered.
      if (std::abs(correlation) < 0.01) continue;
      // Complain about large correlations.  When a correlation is this
      // large, then the user should (but probably won't) rethink the
      // parameter definitions!
      if (std::abs(correlation) > 0.98) {
        LogInfo << "VERY LARGE CORRELATION (" << correlation
                << ") BETWEEN"
                << " " << set1->getName() << "/" << par1->getName()
                << " & " << set2->getName() << "/" << par2->getName()
                << std::endl;
      }
      mcmc.GetProposeStep().SetCorrelation(count1-1,count2-1,
                                           (*corr)(in1,in2));
    }
  }

  return true;
}
bool SimpleMcmc::adaptiveLoadProposalCovariance( AdaptiveStepMCMC& mcmc,
                                                   sMCMC::Vector& prior,
                                                   const std::string& fileName,
                                                   const std::string& histName) {
  // No filename so, no restoration.
  if (fileName.empty()) return false;

  // Check if the filename is "none", and exit if it is.
  {
    std::string tmp{fileName};
    std::use_facet<std::ctype<std::string::value_type>>(std::locale())
      .tolower(&tmp[0],&tmp[0]+tmp.size());
    if (tmp == "none") return false;
  }

  TFile* saveFile = gFile;
  std::unique_ptr<TFile> restoreFile
      (new TFile(fileName.c_str(), "old"));
  if (!restoreFile || !restoreFile->IsOpen()) {
    LogInfo << "File to restore was not found: "
            << fileName << std::endl;
    throw std::runtime_error("Old state file not open");
  }
  LogInfo << "Restore the covariance from external file: "
          << restoreFile->GetName() << std::endl;

  gFile = saveFile;
  gFile->cd((std::string(getOwner().getSaveDir()->GetName())+"/fit").c_str());

  LogInfo << "Set the value of the proposal covariance" << std::endl;
  TH2D* proposalCov = dynamic_cast<TH2D*>(
      restoreFile->Get(_adaptiveCovName_.c_str()));
  if (!proposalCov) {
    LogError << "Proposal TH2D not found "
             << _adaptiveCovName_ << std::endl;
    throw std::runtime_error("Covariance not in file");
  }

  // Check that the covariance that was read matchs the number of parameters.
  if (mcmc.GetProposeStep().GetDim() != proposalCov->GetNbinsX()
      or mcmc.GetProposeStep().GetDim() != proposalCov->GetNbinsY()) {
    LogError << "Loading proposal covariance with incorrect dimensions"
             << std::endl;
    LogError << "   Expected Dimensions: " << mcmc.GetProposeStep().GetDim()
             << std::endl;
    LogError << "   Proposal X Bins:     " << proposalCov->GetNbinsX()
             << std::endl;
    LogError << "   Proposal Y Bins:     " << proposalCov->GetNbinsY()
             << std::endl;
    LogThrow("Mismatched proposal covariance matrix");
  }

  // Dump all of the previous correlations.
  mcmc.GetProposeStep().ResetCorrelations();

  TAxis* covAxisLabels = dynamic_cast<TAxis*>(proposalCov->GetXaxis());
  int count1 = 0;
  for (const Parameter* par1 : getMinimizerFitParameterPtr() ) {
    ++count1;
    std::string parName(par1->getFullTitle());
    std::string covName(covAxisLabels->GetBinLabel(count1));
    if (parName != covName) {
      LogError << "Mismatch of parameter and covariance names" << std::endl;
      LogError << "Parameter:  " << parName << std::endl;
      LogError << "Covariance: " << covName << std::endl;
      LogThrow("Mismatched covariance histogram");
    }
    double sig1 = std::sqrt(proposalCov->GetBinContent(count1,count1));
    int count2 = 0;
    for (const Parameter* par2 : getMinimizerFitParameterPtr() ) {
      ++count2;
      double sig2 = std::sqrt(proposalCov->GetBinContent(count2,count2));
      if (count2 < count1) continue;
      else if (count2 == count1) {
        // Set the sigma for this variable
        double step = sig1;
#define COVARIANCE_NOT_IN_NORMALIZED_FIT_SPACE
#ifdef  COVARIANCE_NOT_IN_NORMALIZED_FIT_SPACE
        if (useNormalizedFitSpace()) {
          step = ParameterSet::toNormalizedParRange(sig1, *par1);
        }
#endif
        mcmc.GetProposeStep().SetGaussian(count1-1,step);
        continue;
      }
      double corr = proposalCov->GetBinContent(count1,count2)/sig1/sig2;
      mcmc.GetProposeStep().SetCorrelation(count1-1,count2-1,corr);
    }
  }

  // Set the effective number of trials for the covariance that was loaded.
  // The covariance is usually calculated by HESSE.  Empirically, HESSE calls
  // the function around constant plus half N^2 times.  Use that as the
  // initial number of trials.  This can also be set as a config file
  // parameter.
  double effectiveTrials = 100+0.5*count1*count1;
  LogInfo << "Setting effective number of trials "
          << effectiveTrials
          << "   User Request: " << _adaptiveCovTrials_
          << std::endl;
  if (_adaptiveCovTrials_ > 0) effectiveTrials = _adaptiveCovTrials_;
  _adaptiveCovTrials_ = effectiveTrials;
  mcmc.GetProposeStep().SetCovarianceTrials(_adaptiveCovTrials_);
  mcmc.GetProposeStep().SetEstimatedCenterTrials(_adaptiveCovTrials_);

  return true;
}

bool SimpleMcmc::adaptiveRunCycle(AdaptiveStepMCMC& mcmc,
                                  std::string chainName,
                                  int chainId) {
  mcmc.SetStepRMSWindow(_adaptiveWindow_);
  mcmc.GetProposeStep().SetAcceptanceWindow(_adaptiveWindow_);
  mcmc.GetProposeStep().SetCovarianceWindow(_adaptiveCovWindow_);

  LogInfo << "Run " << chainName << " " << chainId << std::endl;

  // Update the covariance with the steps from the last cycle.  This starts
  // a new "reversible-chain".  The update always happens at the start of a
  // cycle, even if the sigma and covariance are frozen.
  mcmc.GetProposeStep().UpdateProposal();

  // Reset the covariance to the initial state.  This forgets the path of
  // the previous cycle.  This only happens a few times to let it forget
  // about the very first burn-in cycles
  if (_adaptiveResetCov_) {
    sMCMC::Vector saveCenter{mcmc.GetProposeStep().GetEstimatedCenter()};
    mcmc.GetProposeStep().ResetProposal();
    // After the reset, set how many trials the prior covariance counts
    // for.  This needs to be done by hand since the extra weight lives
    // here, and not in TSimpleMCMC.h (TSimpleMCMC doesn't, and shouldn't,
    // know about this burn-in process).
    if (_adaptiveCovTrials_ > 0) {
      mcmc.GetProposeStep().SetCovarianceTrials(_adaptiveCovTrials_);
      // We have a good prior (usually based on a previous MINUIT fit), so
      // restore the current estimate of the centeral value.
      mcmc.GetProposeStep().SetEstimatedCenter(saveCenter);
      mcmc.GetProposeStep().SetEstimatedCenterTrials(_adaptiveCovTrials_);
    }
  }

  // Set whether the running covariance will be updated based on the new
  // steps.  This freezes right after the last update that will be called.
  if (_adaptiveFreezeCov_) {
    LogInfo << "Step correlations are frozen" << std::endl;
    mcmc.GetProposeStep().SetCovarianceFrozen(true);
  }
  else {
    LogInfo << "Step correlations are being updated" << std::endl;
    mcmc.GetProposeStep().SetCovarianceFrozen(false);
  }

  // Override default number of steps until the next automatic
  // UpdateProposal call.  This disables automatic updates during normal
  // chains.
  mcmc.GetProposeStep().SetNextUpdate(1E+32);

  // Set the adaptive covariance deweighting after the first update so that
  // the previous deweighting is used for the first update.  This is a very
  // cheap call so set it on every iteration.
  mcmc.GetProposeStep()
    .SetCovarianceUpdateDeweighting(_adaptiveCovDeweighting_);

  // Set whether the step size is going to be tuned to deliver the target
  // acceptance.
  if (_adaptiveFreezeStep_) {
    LogInfo << "Step length is frozen" << std::endl;
    mcmc.GetProposeStep().SetAcceptanceRigidity(-1);
  }
  else {
    LogWarning << "Step length will be updated" << std::endl;
    mcmc.GetProposeStep().SetAcceptanceRigidity(2.0);
  }

  ////////////////////////////////
  // Run the steps for this chain
  for (int i = 0; i < _steps_; ++i) {
    // Run step, but do not save the step.  The step isn't saved so the
    // accepted step can be copied into the points (which will have
    // any decomposition removed).
    if (mcmc.Step(false,_adaptiveAcceptanceAlgorithm_)) fillPoint();
    // Zero the size of the raw accepted points and only save the filled
    // points (which are in likelihood space).
    if (not _saveRawSteps_) mcmc.ClearSavedAccepted();
    // Limit the number of times that the model histogram data is saved to
    // the output file.  It won't be saved if the _modelStride_ is less than
    // one.  Otherwise it will be saved every _modelStride_ steps.
    if (_modelStride_ > 0
        and 0 == (mcmc.GetProposeStep().GetTrials()%_modelStride_)) {
      _saveModel_.resize(_model_.size());
      std::copy(_model_.begin(), _model_.end(), _saveModel_.begin());
      _saveUncertainty_.resize(_uncertainty_.size());
      std::copy(_uncertainty_.begin(), _uncertainty_.end(),
                _saveUncertainty_.begin());
    }
    // Now save the step.  This is going to write the points in the
    // "likelihood" space.  If "_saveRawSteps_" is true, then this also
    // saves the accepted point in the (possibly) decomposed state.
    if (_saveSteps_) mcmc.SaveStep(false);
    if(_steps_ > 100 && !(i%(_steps_/100))){
      LogInfo << chainName << ": " << chainId
              << " step: " << i << "/" << _steps_ << " "
              << i*100./_steps_ << "%"
              << " Trials: "
              << mcmc.GetProposeStep().GetSuccesses()
              << "/" << mcmc.GetProposeStep().GetTrials()
              << " (acc " << mcmc.GetProposeStep().GetAcceptance()
              << ", sig " << mcmc.GetProposeStep().GetSigma()
              << ", rms " << mcmc.GetStepRMS()
              << ")"
              << std::endl;
    }
  }
  // Make a final step and then save it with the covariance information.
  // These steps can be identified since the covariance is not empty, but
  // they should be real independent steps.
  if (mcmc.Step(false,_adaptiveAcceptanceAlgorithm_)) fillPoint();
  // This is not resetting the "SaveAccepted" size so the step is actually
  // saved.
  if (_saveSteps_) mcmc.SaveStep(true);
  LogInfo << chainName << ": " << chainId << " complete"
          << " Run Length: " << _steps_
          << " -- Saving state"
          << std::endl;

  return true;
}

void SimpleMcmc::adaptiveMakePrior(AdaptiveStepMCMC& mcmc,
                                   sMCMC::Vector& prior,
                                   bool randomize) {
  prior.clear();
  for (const Parameter* par : getMinimizerFitParameterPtr() ) {
    double val = par->getPriorValue();
    if (randomize) {
      double err = par->getStdDevValue();
      double r = gRandom->Uniform(0.0,1.0);
      double lowBound = val-1.0*err;
      double highBound = val+1.0*err;

      if( par->getParameterLimits().hasLowerBound() ) {
        lowBound = std::max(lowBound, par->getParameterLimits().min);
      }
      if( par->getMirrorRange().hasLowerBound() ){
        lowBound = std::max(lowBound, par->getMirrorRange().min);
      }
      if( par->getPhysicalLimits().hasLowerBound() ){
        lowBound = std::max(lowBound, par->getPhysicalLimits().min);
      }

      if( par->getParameterLimits().hasUpperBound() ) {
        highBound = std::min(highBound, par->getParameterLimits().max);
      }
      if( par->getMirrorRange().hasUpperBound() ){
        highBound = std::min(highBound, par->getMirrorRange().max);
      }
      if( par->getPhysicalLimits().hasUpperBound() ){
        highBound = std::min(highBound, par->getPhysicalLimits().max);
      }

      val = lowBound + r*(highBound-lowBound);
    }
    if (not useNormalizedFitSpace()) {
      prior.push_back(val);
    }
    else {
      prior.push_back(ParameterSet::toNormalizedParValue(val, *par));
    }
  }
}

void SimpleMcmc::adaptiveStart(AdaptiveStepMCMC& mcmc,
                               sMCMC::Vector& prior,
                               bool randomize) {
  int throttle{100};
  bool startStatus{false};
  do{
    adaptiveMakePrior(mcmc,prior,randomize);
    try {startStatus = mcmc.Start(prior, false);}
    catch (...) {startStatus = false;}
    if (not startStatus and not randomize) {
      LogExit("Invalid initial point for chain");
    }
    if(!startStatus) prior.clear();
  } while (not startStatus and 0 < throttle--);
}

void SimpleMcmc::adaptiveRestart(AdaptiveStepMCMC& mcmc,
                                 bool randomize) {
  sMCMC::Vector prior;
  int throttle{0};
  while (throttle++ < 1000) {
    LogInfo << "Restart at a new point (trial: " << throttle << ")"
            << std::endl;
    adaptiveMakePrior(mcmc, prior, true);
    mcmc.GetProposeStep().ForceStep(prior);
    mcmc.Step(false,2);
    // Stop if the proposed step has a non-zero (and finite) likelihood
    if (std::isfinite(mcmc.GetAcceptedLogLikelihood())
        and mcmc.GetAcceptedLogLikelihood() > -1.0E+10) break;
  };
}
void SimpleMcmc::adaptiveSetupAndRun(AdaptiveStepMCMC& mcmc) {

  _adaptiveMCMC_ = &mcmc;

  mcmc.GetProposeStep().SetDim(getMinimizerFitParameterPtr().size());
  mcmc.GetLogLikelihood().functor
    = std::make_unique<ROOT::Math::Functor>(
      this, &SimpleMcmc::evalFitValid,
      getMinimizerFitParameterPtr().size());
  mcmc.GetProposeStep().SetCovarianceUpdateDeweighting(0.0);
  mcmc.GetProposeStep().SetCovarianceFrozen(false);

  // Create a fitting parameter vector and initialize it.  No need to worry
  // about resizing it or it moving, so be lazy and just use push_back.
  sMCMC::Vector prior;

  adaptiveStart(mcmc,prior,_randomStart_);

  // Set the correlations in the default step proposal.
  if (not adaptiveLoadProposalCovariance(mcmc,
                                         prior,
                                         _adaptiveCovFile_,
                                         _adaptiveCovName_)) {;
    // The correlations were not loaded from a predefined covariance. Set up
    // a default set of correlations.
    adaptiveDefaultProposalCovariance(mcmc,prior);
  }

  // Fill the initial point.
  fillPoint();

  // Restore the chain if a file is provided.  The covariance is
  // updated during the restore, so the proposal is updated.  That means that
  // the step length should be tuned after a restore to maintain the correct
  // acceptance.
  std::string restorationTree = "FitterEngine/fit/" + _outTreeName_;
  bool restored = adaptiveRestoreState(mcmc,_adaptiveRestore_, restorationTree);

  // Check if there should be some burn-in cycles.
  if (not restored and _burninCycles_ > 0) {
    gMCMC.SetSequencerState(true);
#ifdef HARD_CODE_BURNIN_SEQUENCE
    for (int chain = 0; chain < gMCMC.Burnin(); ++chain) {
      gMCMC.Steps(_burninSteps_);
      gMCMC.AcceptanceWindow(_burninWindow_);
      gMCMC.CovarianceWindow(_burninCovWindow_);
      gMCMC.CovarianceDeweighting(_burninCovDeweighting_);
      gMCMC.FreezeStep((chain >= _burninFreezeAfter_));
      gMCMC.FreezeCovariance(false);
      gMCMC.ResetCovariance((chain < _burninResets_));
      gMCMC.RunCycle("Burn-in", chain);
    }
#else
    LogInfo << "Burn-in MCMC with sequence:" << std::endl << _burninSequence_ << std::endl;
    gROOT->ProcessLineSync(_burninSequence_.c_str());
    LogInfo << "Burn-in line processed" << std::endl;
#endif
    gMCMC.SetSequencerState(false);
    LogInfo << "Finished burn-in chains" << std::endl;
  }

  ////////////////////////////////////////////////////////////////
  // Run the main cycles.
  gMCMC.SetSequencerState(true);
#ifdef HARD_CODE_SEQUENCE
  for (int chain = 0; chain < gMCMC.Cycles(); ++chain) {
    gMCMC.FreezeStep((chain >= _adaptiveFreezeAfter_));
    gMCMC.FreezeCovariance((chain >= _adaptiveFreezeCorrelationsAfter_));
    gMCMC.RunCycle("Chain", chain);
  }
#else
  LogInfo << "Run MCMC with Sequence" << std::endl << _sequence_ << std::endl;
  gROOT->ProcessLineSync(_sequence_.c_str());
#endif
  gMCMC.SetSequencerState(false);
  LogInfo << "Finished running chains" << std::endl;

}
void SimpleMcmc::fixedSetupAndRun( FixedStepMCMC& mcmc) {

  mcmc.GetProposeStep().SetDim(getMinimizerFitParameterPtr().size());
  mcmc.GetLogLikelihood().functor
    = std::make_unique<ROOT::Math::Functor>(
      this,
      &SimpleMcmc::evalFitValid,
      getMinimizerFitParameterPtr().size());
  mcmc.GetProposeStep().fSigma = _simpleSigma_;

  sMCMC::Vector prior;
  for (const Parameter* par : getMinimizerFitParameterPtr() ) {
    prior.push_back(par->getPriorValue());
  }

  // Fill the initial point.
  fillPoint();

  // Initializing the mcmc sampler
  mcmc.Start(prior, _saveBurnin_);

  // Burn-In cycles
  for (int chain = 0; chain < _burninCycles_; ++chain){
    LogInfo << "Start Burn-In Cycle " << chain << std::endl;
    // Burn-In chain in each cycle
    for (int i = 0; i < _burninSteps_; ++i) {
      // Run step
      if (mcmc.Step(false)) fillPoint();
      // Save the step if burn-in steps are being saved..  Check to see if this
      // is the last step of the run, and if it is, then save the full state.
      if (_saveBurnin_) mcmc.SaveStep(_burninSteps_ <= (i+1));
      if(_burninSteps_ > 100 && !(i%(_burninSteps_/100))){
        LogInfo << "Burn-in: " << chain
                << " step: " << i << "/" << _burninSteps_ << " "
                << i*100./_burninSteps_ << "%"
                << std::endl;
      }
    }
  }

  // Run cycles
  for (int chain = 0; chain < _cycles_; ++chain){
    LogInfo << "Start Main Cycle " << chain << std::endl;
    // Update the covariance with the steps from the last cycle.  This
    // starts a new "reversible-chain".
    mcmc.GetProposeStep().UpdateProposal();
    // Run chain in each cycle
    for (int i = 0; i < _steps_; ++i) {
      // Run step, but do not save the step.  The step isn't saved so the
      // accepted step can be copied into the points (which will have
      // any decomposition removed).
      if (mcmc.Step(false)) fillPoint();
      // Now save the step.  If this is the last step in the cycle, do a full
      // save.
      if (i < (_steps_-1)) mcmc.SaveStep(false);
      else mcmc.SaveStep(true);
      if(_steps_ > 100 && !(i%(_steps_/100))){
        LogInfo << "Chain: " << chain
                << " step: " << i << "/" << _steps_ << " "
                << i*100./_steps_ << "%"
                << std::endl;
      }
    }

    LogInfo << "Chain: " << chain << " complete"
            << " Run Length: " << _steps_
            << " -- Saving state"
            << std::endl;
  }
  LogInfo << "Finished Running chains" << std::endl;

}

void SimpleMcmc::minimize() {
  this->MinimizerBase::minimize();

  GenericToolbox::mkdirTFile( getOwner().getSaveDir(), "fit" )->cd();

  // Create output tree in the existing file
  LogInfo << "Adding MCMC information to file " << gFile->GetName() << std::endl;

  // Store parameter names as a tree in the currentdirectory
  TTree *parameterSetsTree = new TTree("parameterSets",
                                       "Tree of MCMC Model Information");
  std::vector<std::string> parameterSetNames;
  std::vector<int> parameterSetOffsets;
  std::vector<int> parameterSetCounts;

  parameterSetsTree->Branch("parameterSetNames", &parameterSetNames);
  parameterSetsTree->Branch("parameterSetOffsets", &parameterSetOffsets);
  parameterSetsTree->Branch("parameterSetCounts", &parameterSetCounts);

  std::vector<int> parameterIndex;
  std::vector<bool> parameterFixed;
  std::vector<bool> parameterEnabled;
  std::vector<std::string> parameterName;
  std::vector<double> parameterPrior;
  std::vector<double> parameterSigma;
  std::vector<double> parameterMin;
  std::vector<double> parameterMax;

  parameterSetsTree->Branch("parameterIndex", &parameterIndex);
  parameterSetsTree->Branch("parameterFixed", &parameterFixed);
  parameterSetsTree->Branch("parameterEnabled", &parameterEnabled);
  parameterSetsTree->Branch("parameterName", &parameterName);
  parameterSetsTree->Branch("parameterPrior", &parameterPrior);
  parameterSetsTree->Branch("parameterSigma", &parameterSigma);
  parameterSetsTree->Branch("parameterMin", &parameterMin);
  parameterSetsTree->Branch("parameterMax", &parameterMax);

  std::vector<std::string> parameterSampleNames;
  std::vector<int> parameterSampleOffsets;
  std::vector<double> parameterSampleData;

  parameterSetsTree->Branch("parameterSampleName", &parameterSampleNames);
  parameterSetsTree->Branch("parameterSampleOffsets", &parameterSampleOffsets);
  parameterSetsTree->Branch("parameterSampleData", &parameterSampleData);

  // Pull all of the parameters out of the parameter sets and save the
  // names, index, priors, and sigmas to the output.  The order in these
  // vectors define the parameters that are saved in the Points vector in
  // the output file.  Note that these parameters do not define the meaning
  // of the parameters in the call to the likelihood function.  Those
  // parameters are defined by the _minimizerFitParameterPtr_ vector.

  for (const ParameterSet& parSet: getModelPropagator().getParametersManager().getParameterSetsList()) {
    // Save name of parameter set
    parameterSetNames.push_back(parSet.getName());
    parameterSetOffsets.push_back(parameterIndex.size());

    int countParameters = 0;
    for (const Parameter& iPar : parSet.getParameterList()) {
      ++countParameters;
      parameterIndex.push_back(iPar.getParameterIndex());
      parameterFixed.push_back(iPar.isFixed());
      parameterEnabled.push_back(iPar.isEnabled());
      parameterName.push_back(iPar.getTitle());
      parameterPrior.push_back(iPar.getPriorValue());
      parameterSigma.push_back(iPar.getStdDevValue());
      parameterMin.push_back(iPar.getParameterLimits().min);
      parameterMax.push_back(iPar.getParameterLimits().max);
    }
    parameterSetCounts.push_back(countParameters);
  }

  parameterSampleData.clear();
  for (const Sample& sample
      : getModelPropagator().getSampleSet().getSampleList()) {
    parameterSampleNames.push_back(sample.getName());
    parameterSampleOffsets.push_back(parameterSampleData.size());
    auto& hist = sample.getHistogram();
    // Adrien: same here... the last been has always been skipped
    for (int i = 1; i < hist.getNbBins(); ++i) {
      parameterSampleData.push_back(hist.getBinContentList()[i-1].sumWeights);
    }
    LogInfo << "Save data histogram for " << parameterSampleNames.back()
            << " @ " << parameterSampleOffsets.back()
            << " w/ " << hist.getNbBins() << " bins"
            << std::endl;
  }

  parameterSetsTree->Fill();
  parameterSetsTree->Write();

  // End of saving model information

  _point_.resize(parameterName.size());
  LogInfo << "Parameters in likelihood: " << _point_.size() << std::endl;

  // Create the output tree for the accepted points.
  auto *outputTree = new TTree(_outTreeName_.c_str(),
                               "Tree of accepted points");
  outputTree->Branch("Points",&_point_);
  outputTree->Branch("LLHPenalty",&_llhPenalty_);
  outputTree->Branch("LLHStatistical",&_llhStatistical_);
  outputTree->Branch("Models",&_saveModel_);
  outputTree->Branch("ModelUncertainty",&_saveUncertainty_);

  getMonitor().stateTitleMonitor = "Running MCMC chain";
  getMonitor().minimizerTitle = _algorithmName_ + "/" + _proposalName_;

  // Run a chain.
  int nbFitCallOffset = getMonitor().nbEvalLikelihoodCalls;
  LogInfo << "Fit call offset: " << nbFitCallOffset << std::endl;

  // Create the TSimpleMCMC object and call the specific runner.
  if (_proposalName_ == "adaptive") {
    sMCMC::TSimpleMCMC<PrivateProxyLikelihood,sMCMC::TProposeAdaptiveStep> mcmc(outputTree);
    adaptiveSetupAndRun(mcmc);
  }
  else if (_proposalName_ == "fixed") {
    sMCMC::TSimpleMCMC<PrivateProxyLikelihood,sMCMC::TProposeSimpleStep> mcmc(outputTree);
    fixedSetupAndRun(mcmc);
  }
  else {
    LogExit("Invalid proposal type for MCMC: " << _proposalName_)
  }

  int nbMCMCCalls = getMonitor().nbEvalLikelihoodCalls - nbFitCallOffset;

  // lasting printout
  LogInfo << getMonitor().convergenceMonitor.generateMonitorString();
  LogInfo << "MCMC ended after " << nbMCMCCalls << " calls." << std::endl;

  // Save the sampled points to the outputfile
  outputTree->Write();

  // success
  setMinimizerStatus(0);
}

double SimpleMcmc::evalFitValid(const double* parArray_) {

  double value = this->evalFit( parArray_ );
  if (hasValidParameterValues()) return value;
  /// A "Really Big Number".  This is nominally just infinity, but is done as
  /// a defined constant to make the code easier to understand.  This needs to
  /// be an appropriate value to safely represent an impossible chi-squared
  /// value "representing" -log(0.0)/2 and should should be larger than 5E+30.
  const double RBN = std::numeric_limits<double>::infinity();
  return RBN;
}
void SimpleMcmc::setParameterValidity(const std::string& validity) {


  LogWarning << "Set parameter validity to " << validity << std::endl;

  if      ( GenericToolbox::hasSubStr(validity, "noran") ){ _validFlags_ &= ~0b0001; }
  else if ( GenericToolbox::hasSubStr(validity, "ran")   ){ _validFlags_ |= 0b0001; }

  if (validity.find("nomir") != std::string::npos) _validFlags_ &= ~0b0010;
  else if (validity.find("mir") != std::string::npos) _validFlags_ |= 0b0010;

  if (validity.find("nophy") != std::string::npos) _validFlags_ &= ~0b0100;
  else if (validity.find("phy") != std::string::npos) _validFlags_ |= 0b0100;

  LogWarning << "Set parameter validity to " << validity << " (" << _validFlags_ << ")" << std::endl;
}
bool SimpleMcmc::hasValidParameterValues() const {


  int invalid = 0;
  for( auto& parSet: getModelPropagator().getParametersManager().getParameterSetsList() ){
    for( auto& par : parSet.getParameterList() ){
      if ( (_validFlags_ & 0b0001) != 0
           and std::isfinite(par.getParameterLimits().min)
           and par.getParameterValue() < par.getParameterLimits().min) GUNDAM_UNLIKELY_COMPILER_FLAG {
        ++invalid;
      }
      if ((_validFlags_ & 0b0001) != 0
          and std::isfinite(par.getParameterLimits().max)
          and par.getParameterValue() > par.getParameterLimits().max) GUNDAM_UNLIKELY_COMPILER_FLAG {
        ++invalid;
      }
      if ((_validFlags_ & 0b0010) != 0
          and std::isfinite(par.getMirrorRange().min)
          and par.getParameterValue() < par.getMirrorRange().min) GUNDAM_UNLIKELY_COMPILER_FLAG {
        ++invalid;
      }
      if ((_validFlags_ & 0b0010) != 0
          and std::isfinite(par.getMirrorRange().max)
          and par.getParameterValue() > par.getMirrorRange().max) GUNDAM_UNLIKELY_COMPILER_FLAG {
        ++invalid;
      }
      if ((_validFlags_ & 0b0100) != 0
          and std::isfinite(par.getPhysicalLimits().min)
          and par.getParameterValue() < par.getPhysicalLimits().min) GUNDAM_UNLIKELY_COMPILER_FLAG {
        ++invalid;
      }
      if ((_validFlags_ & 0b0100) != 0
          and std::isfinite(par.getPhysicalLimits().max)
          and par.getParameterValue() > par.getPhysicalLimits().max) GUNDAM_UNLIKELY_COMPILER_FLAG {
        ++invalid;
      }

    }
  }
  return (invalid == 0);
}

SimpleMcmcSequencer::SimpleMcmcSequencer() {};
SimpleMcmcSequencer::~SimpleMcmcSequencer() {};
void SimpleMcmcSequencer::setEngine(FitterEngine* engine) {_engine_ = engine;}
SimpleMcmc& SimpleMcmcSequencer::Owner() {
  SimpleMcmc* mcmc
    = dynamic_cast<SimpleMcmc*>(&(_engine_->getMinimizer()));
  LogThrowIf(not mcmc, "SimpleMcmcSequencer with wrong type of minimizer");
  LogThrowIf(not _validState_,
             "SimpleMcmcSequencer used outside of a valid SimpleMcmc state");
  return *mcmc;
}
void SimpleMcmcSequencer::SetSequencerState(bool s) {_validState_ = s;}
void SimpleMcmcSequencer::RandomStart(bool v) {Owner()._randomStart_ = v;}
void SimpleMcmcSequencer::Steps(int v) {Owner()._steps_ = v;}
void SimpleMcmcSequencer::SaveSteps(bool v) {Owner()._saveSteps_ = v;}
void SimpleMcmcSequencer::ModelStride(int v) {Owner()._modelStride_ = v;}
void SimpleMcmcSequencer::FreezeStep(bool v) {Owner()._adaptiveFreezeStep_ = v;}
void SimpleMcmcSequencer::FreezeCovariance(bool v) {Owner()._adaptiveFreezeCov_ = v;}
void SimpleMcmcSequencer::ResetCovariance(bool v) {Owner()._adaptiveResetCov_ = v;}
void SimpleMcmcSequencer::CovarianceWindow(int v) {Owner()._adaptiveCovWindow_ = v;}
void SimpleMcmcSequencer::CovarianceDeweighting(double v) {Owner()._adaptiveCovDeweighting_ = v;}
void SimpleMcmcSequencer::AcceptanceWindow(int v) {Owner()._adaptiveWindow_ = v;}
void SimpleMcmcSequencer::AcceptanceAlgorithm(int v) {Owner()._adaptiveAcceptanceAlgorithm_ = v;}
void SimpleMcmcSequencer::SetSigma(double s) {Owner()._adaptiveMCMC_->GetProposeStep().SetSigma(s);}
int SimpleMcmcSequencer::Burnin() {return Owner()._burninCycles_;}
int SimpleMcmcSequencer::Cycles() {return Owner()._cycles_;}
int SimpleMcmcSequencer::Steps() {return Owner()._steps_;}
void SimpleMcmcSequencer::RunCycle(std::string name, int id) {
  Owner().adaptiveRunCycle(*Owner()._adaptiveMCMC_, name, id);
  Owner().restoreConfiguration();
}
void SimpleMcmcSequencer::Restart() {
  sMCMC::Vector prior;
  Owner().adaptiveRestart(*Owner()._adaptiveMCMC_, true);
}

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
