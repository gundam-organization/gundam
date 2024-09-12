//
// Created by Clark McGrew 24/1/23
//

#include "LikelihoodInterface.h"
#include "FitterEngine.h"
#include "GundamGlobals.h"

#include "GenericToolbox.ScopedGuard.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.Json.h"
#include "GenericToolbox.h"
#include "Logger.h"

#include <limits>


LoggerInit([]{
  Logger::setUserHeaderStr("[Likelihood]");
});


LikelihoodInterface::LikelihoodInterface(FitterEngine* owner_): _owner_(owner_) {}
LikelihoodInterface::~LikelihoodInterface() = default;

void LikelihoodInterface::setOwner(FitterEngine* owner_) {
  _owner_ = owner_;
}
void LikelihoodInterface::setStateTitleMonitor(const std::string& stateTitleMonitor_){
  _stateTitleMonitor_ = stateTitleMonitor_;
}


void LikelihoodInterface::initialize() {

  if( not GundamGlobals::isLightOutputMode() ){
    _chi2HistoryTree_ = std::make_unique<TTree>("chi2History", "chi2History");
    _chi2HistoryTree_->SetDirectory(nullptr);
    _chi2HistoryTree_->Branch("nbFitCalls", &_nbFitCalls_);
    _chi2HistoryTree_->Branch("chi2Total", (double*) _owner_->getPropagator().getLlhBufferPtr());
    _chi2HistoryTree_->Branch("chi2Stat", (double*) _owner_->getPropagator().getLlhStatBufferPtr());
    _chi2HistoryTree_->Branch("chi2Pulls", (double*) _owner_->getPropagator().getLlhPenaltyBufferPtr());
    _chi2HistoryTree_->Branch("itSpeed", _itSpeedMon_.getCountSpeedPtr());
  }

  LogWarning << "Fetching the effective number of fit parameters..." << std::endl;
  _minimizerFitParameterPtr_.clear();
  _nbFreePars_ = 0;
  for( auto& parSet : _owner_->getPropagator().getParametersManager().getParameterSetsList() ){
    for( auto& par : parSet.getEffectiveParameterList() ){
      if( par.isEnabled() and not par.isFixed() ) {
        _minimizerFitParameterPtr_.emplace_back(&par);
        if( par.isFree() ) _nbFreePars_++;
      }
    }
  }
  _nbFitParameters_ = int(_minimizerFitParameterPtr_.size());

  LogInfo << "Building functor with " << _nbFitParameters_ << " parameters ..." << std::endl;
  _functor_ = std::make_unique<ROOT::Math::Functor>(this, &LikelihoodInterface::evalFit, _nbFitParameters_);
  _validFunctor_ = std::make_unique<ROOT::Math::Functor>(this, &LikelihoodInterface::evalFitValid, _nbFitParameters_);

  _nbFitBins_ = 0;
  for( auto& sample : _owner_->getPropagator().getFitSampleSet().getFitSampleList() ){
    _nbFitBins_ += int(sample.getBinning().getBinList().size());
  }

  _convergenceMonitor_.addDisplayedQuantity("VarName");
  _convergenceMonitor_.addDisplayedQuantity("LastAddedValue");
  _convergenceMonitor_.addDisplayedQuantity("SlopePerCall");

  _convergenceMonitor_.getQuantity("VarName").title = "Likelihood";
  _convergenceMonitor_.getQuantity("LastAddedValue").title = "Current Value";
  _convergenceMonitor_.getQuantity("SlopePerCall").title = "Avg. Slope /call";

  _convergenceMonitor_.addVariable("Total/dof");
  _convergenceMonitor_.addVariable("Total");
  _convergenceMonitor_.addVariable("Stat");
  _convergenceMonitor_.addVariable("Syst");

  _isInitialized_ = true;
}

void LikelihoodInterface::saveChi2History() {
  if( not GundamGlobals::isLightOutputMode() ){
    LogInfo << "Saving LLH history..." << std::endl;
    GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(_owner_->getSaveDir(), "fit"), _chi2HistoryTree_.get());
  }
  else{
    LogAlert << "Not saving LLH history as light output mode is fired." << std::endl;
  }
}
void LikelihoodInterface::saveGradientSteps(){

  if( GundamGlobals::isLightOutputMode() ){
    LogAlert << "Skipping saveGradientSteps as light output mode is fired." << std::endl;
    return;
  }

  LogInfo << "Saving " << _gradientMonitor_.size() << " gradient steps..." << std::endl;

  // make sure the parameter states get restored as we leave
  auto currentParState = _owner_->getPropagator().getParametersManager().exportParameterInjectorConfig();
  GenericToolbox::ScopedGuard g{
    [&](){
      ParametersManager::muteLogger();
      ParameterSet::muteLogger();
      ParScanner::muteLogger();
    },
    [&](){
      _owner_->getPropagator().getParametersManager().injectParameterValues( currentParState );
      ParametersManager::unmuteLogger();
      ParameterSet::unmuteLogger();
      ParScanner::unmuteLogger();
    }
  };

  // load starting point
  auto lastParStep{_owner_->getPreFitParState()};

  std::vector<GraphEntry> globalGraphList;
  for(size_t iGradStep = 0 ; iGradStep < _gradientMonitor_.size() ; iGradStep++ ){
    GenericToolbox::displayProgressBar(iGradStep, _gradientMonitor_.size(), LogInfo.getPrefixString() + "Saving gradient steps...");

    // why do we need to remute the logger at each loop??
    ParameterSet::muteLogger(); Propagator::muteLogger(); ParametersManager::muteLogger();
    _owner_->getPropagator().getParametersManager().injectParameterValues(_gradientMonitor_[iGradStep].parState );
    _owner_->getPropagator().updateLlhCache();

    if( not GundamGlobals::isLightOutputMode() ) {
      auto outDir = GenericToolbox::mkdirTFile(_owner_->getSaveDir(), Form("fit/gradient/step_%i", int(iGradStep)));
      GenericToolbox::writeInTFile(outDir, TNamed("parState", GenericToolbox::Json::toReadableString(_gradientMonitor_[iGradStep].parState).c_str()));
      GenericToolbox::writeInTFile(outDir, TNamed("llhState", _owner_->getPropagator().getLlhBufferSummary().c_str()));
    }

    // line scan from previous point
    _owner_->getPropagator().getParScanner().scanSegment( nullptr, _gradientMonitor_[iGradStep].parState, lastParStep, 8 );
    lastParStep = _gradientMonitor_[iGradStep].parState;

    if( globalGraphList.empty() ){
      // copy
      globalGraphList = _owner_->getPropagator().getParScanner().getGraphEntriesBuf();
    }
    else{
      // current
      auto& grEntries = _owner_->getPropagator().getParScanner().getGraphEntriesBuf();

      for( size_t iEntry = 0 ; iEntry < globalGraphList.size() ; iEntry++ ){
        for(int iPt = 0 ; iPt < grEntries[iEntry].graph.GetN() ; iPt++ ){
          globalGraphList[iEntry].graph.AddPoint( grEntries[iEntry].graph.GetX()[iPt], grEntries[iEntry].graph.GetY()[iPt] );
        }
      }

    }
  }

  if( not globalGraphList.empty() ){
    auto outDir = GenericToolbox::mkdirTFile(_owner_->getSaveDir(), "fit/gradient/global");
    for( auto& gEntry : globalGraphList ){
      gEntry.scanDataPtr->title = "Minimizer path to minimum";
      ParScanner::writeGraphEntry(gEntry, outDir);
    }
    GenericToolbox::triggerTFileWrite(outDir);

    outDir = GenericToolbox::mkdirTFile(_owner_->getSaveDir(), "fit/gradient/globalRelative");
    for( auto& gEntry : globalGraphList ){
      if( gEntry.graph.GetN() == 0 ){ continue; }

      double minY{gEntry.graph.GetY()[gEntry.graph.GetN()-1]};
      double maxY{gEntry.graph.GetY()[0]};
      double delta{1E-6*std::abs( maxY - minY )};
      // allow log scale
      minY += delta;

      for( int iPt = 0 ; iPt < gEntry.graph.GetN() ; iPt++ ){
        gEntry.graph.GetY()[iPt] -= minY;
      }
      gEntry.scanDataPtr->title = "Minimizer path to minimum (difference)";
      ParScanner::writeGraphEntry(gEntry, outDir);
    }
    GenericToolbox::triggerTFileWrite(outDir);
  }

}

double LikelihoodInterface::evalFit(const double* parArray_){
  LogThrowIf(not _isInitialized_, "not initialized");

  if(_nbFitCalls_ != 0){
    _outEvalFitAvgTimer_.counts++ ; _outEvalFitAvgTimer_.cumulated += GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds("out_evalFit");
  }
  ++_nbFitCalls_;

  static bool lastCallOK{true};
  if (not lastCallOK) {
    // Provide a marker in the log file after there was an evaluation problem.
    LogError << "Call: " << _nbFitCalls_
             << " -- Previous evalFit call was not OK" << std::endl;
  }
  lastCallOK = true;

  // Update fit parameter values:
  const double *v = parArray_;
  for( auto* par : _minimizerFitParameterPtr_ ){
    double p = *(v++);
    if (getUseNormalizedFitSpace()) p = ParameterSet::toRealParValue(p, *par);
    par->setParameterValue(p,true);
  }

  GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);

  // Compute the Chi2
  _owner_->getPropagator().updateLlhCache();

  if( std::isnan(_owner_->getPropagator().getLlhBuffer()) ){
    lastCallOK = false;
    LogError << "Call: " << _nbFitCalls_
             << " " << _owner_->getPropagator().getLlhBuffer() << ": invalid reported likelihood value." << std::endl;
    LogError << GET_VAR_NAME_VALUE( _owner_->getPropagator().getLlhStatBuffer() ) << std::endl;
    for( auto& sample : _owner_->getPropagator().getFitSampleSet().getFitSampleList() ){
      LogScopeIndent;
      LogError << sample.getName() << ": " << sample.getLlhStatBuffer() << std::endl;
    }
    LogError << GET_VAR_NAME_VALUE( _owner_->getPropagator().getLlhPenaltyBuffer() ) << std::endl;
    LogThrow("Invalid total llh.");
  }
  else if( _owner_->getPropagator().getLlhBuffer() > 0.05*std::numeric_limits<float>::max()
           or not std::isfinite(_owner_->getPropagator().getLlhBuffer())) {
    lastCallOK = false;
    LogError << "Call: " << _nbFitCalls_
             << " " << _owner_->getPropagator().getLlhBuffer() << ": very large likelihood value." << std::endl;
    LogError << GET_VAR_NAME_VALUE( _owner_->getPropagator().getLlhStatBuffer() ) << std::endl;
    for( auto& sample : _owner_->getPropagator().getFitSampleSet().getFitSampleList() ){
      LogScopeIndent;
      LogError << sample.getName() << ": " << sample.getLlhStatBuffer() << std::endl;
    }
    LogError << GET_VAR_NAME_VALUE( _owner_->getPropagator().getLlhPenaltyBuffer() ) << std::endl;
  }
  else if( _owner_->getPropagator().getLlhBuffer() < 0.0) {
    lastCallOK = false;
    LogError << "Call: " << _nbFitCalls_
             << " "  << _owner_->getPropagator().getLlhBuffer() << ": negative likelihood value." << std::endl;
    LogError << GET_VAR_NAME_VALUE( _owner_->getPropagator().getLlhStatBuffer() ) << std::endl;
    for( auto& sample : _owner_->getPropagator().getFitSampleSet().getFitSampleList() ){
      LogScopeIndent;
      LogError << sample.getName() << ": " << sample.getLlhStatBuffer() << std::endl;
    }
    LogError << GET_VAR_NAME_VALUE( _owner_->getPropagator().getLlhPenaltyBuffer() ) << std::endl;
  }

  _evalFitAvgTimer_.counts++; _evalFitAvgTimer_.cumulated += GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);

  // Minuit based algo might want this
  if( _monitorGradientDescent_ ){
    // check if point is moving toward the minimum
    bool isGradientDescentStep = std::all_of(_minimizerFitParameterPtr_.begin(), _minimizerFitParameterPtr_.end(), [](const Parameter* par_){
      return ( par_->gotUpdated() or par_->isFixed() or not par_->isEnabled() );
    } );
    if( isGradientDescentStep ){
      if( _lastGradientFall_ == _nbFitCalls_-1 ){
        LogWarning << "Overriding last gradient descent entry (minimizer adjusting step size...): ";
        LogWarning(_gradientMonitor_.size() >= 2) << _gradientMonitor_[_gradientMonitor_.size() - 2].llh << " -> ";
        LogWarning << _owner_->getPropagator().getLlhBuffer() << std::endl;
        _gradientMonitor_.back().parState = _owner_->getPropagator().getParametersManager().exportParameterInjectorConfig();
        _gradientMonitor_.back().llh = _owner_->getPropagator().getLlhBuffer();
        _lastGradientFall_ = _nbFitCalls_;
      }
      else{
        // saving each step of the gradient descent
        _gradientMonitor_.emplace_back();
        LogWarning << "Gradient step detected at iteration #" << _nbFitCalls_ << ": ";
        LogWarning(_gradientMonitor_.size() >= 2) << _gradientMonitor_[_gradientMonitor_.size() - 2].llh << " -> ";
        LogWarning << _owner_->getPropagator().getLlhBuffer() << std::endl;
        _gradientMonitor_.back().parState = _owner_->getPropagator().getParametersManager().exportParameterInjectorConfig();
        _gradientMonitor_.back().llh = _owner_->getPropagator().getLlhBuffer();
        _lastGradientFall_ = _nbFitCalls_;
      }
    }
  }

  if( _enableFitMonitor_ and _convergenceMonitor_.isGenerateMonitorStringOk() ){

    _itSpeedMon_.cycle( _nbFitCalls_ - _itSpeedMon_.getCounts() );

    if( _itSpeed_.counts != 0 ){
      _itSpeed_.counts = _nbFitCalls_ - _itSpeed_.counts; // how many cycles since last print
      _itSpeed_.cumulated = GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds("itSpeed"); // time since last print
    }
    else{
      _itSpeed_.counts = _nbFitCalls_;
      GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds("itSpeed");
    }

    std::stringstream ssHeader;
    ssHeader << std::endl << __METHOD_NAME__ << ": call #" << _nbFitCalls_;
    ssHeader << std::endl << _stateTitleMonitor_;
//    ssHeader << std::endl << "Target EDM: " << _owner_->getMinimizer().get;
    ssHeader << std::endl << "RAM: " << GenericToolbox::parseSizeUnits(double(GenericToolbox::getProcessMemoryUsage()));
    double cpuPercent = GenericToolbox::getCpuUsageByProcess();
    ssHeader << " / CPU: " << cpuPercent << "% (" << cpuPercent / GundamGlobals::getParallelWorker().getNbThreads() << "% efficiency)";
    ssHeader << std::endl << "Avg " << GUNDAM_CHI2 << " computation time: " << _evalFitAvgTimer_;
    ssHeader << std::endl;

    GenericToolbox::TablePrinter t;

    t << "" << GenericToolbox::TablePrinter::NextColumn;
    t << "Propagator" << GenericToolbox::TablePrinter::NextColumn;
    t << "Re-weight" << GenericToolbox::TablePrinter::NextColumn;
    t << "histograms fill" << GenericToolbox::TablePrinter::NextColumn;
    t << _minimizerType_ << "/" << _minimizerAlgo_ << GenericToolbox::TablePrinter::NextLine;

    t << "Speed" << GenericToolbox::TablePrinter::NextColumn;
    t << _itSpeedMon_ << GenericToolbox::TablePrinter::NextColumn;
//    t << (double)_itSpeed_.counts / (double)_itSpeed_.cumulated * 1E6 << " it/s" << GenericToolbox::TablePrinter::NextColumn;
    t << _owner_->getPropagator().weightProp << GenericToolbox::TablePrinter::NextColumn;
    t << _owner_->getPropagator().fillProp << GenericToolbox::TablePrinter::NextColumn;
    t << _outEvalFitAvgTimer_ << GenericToolbox::TablePrinter::NextLine;

    ssHeader << t.generateTableString();

    if( _showParametersOnFitMonitor_ ){
      std::string curParSet;
      ssHeader << std::endl << std::setprecision(1) << std::scientific << std::showpos;
      int nParPerLine{0};
      for( auto* fitPar : _minimizerFitParameterPtr_ ){
        if( fitPar->isFixed() ) continue;
        if( curParSet != fitPar->getOwner()->getName() ){
          if( not curParSet.empty() ) ssHeader << std::endl;
          curParSet = fitPar->getOwner()->getName();
          ssHeader << curParSet
                   << (fitPar->getOwner()->isUseEigenDecompInFit()? " (eigen)": "")
                   << ":" << std::endl;
          nParPerLine = 0;
        }
        else{
          ssHeader << ", ";
          if( nParPerLine >= _maxNbParametersPerLineOnMonitor_ ) {
            ssHeader << std::endl; nParPerLine = 0;
          }
        }
        if(fitPar->gotUpdated()) ssHeader << GenericToolbox::ColorCodes::blueBackground;
        if(getUseNormalizedFitSpace()) ssHeader << ParameterSet::toNormalizedParValue(fitPar->getParameterValue(), *fitPar);
        else ssHeader << fitPar->getParameterValue();
        if(fitPar->gotUpdated()) ssHeader << GenericToolbox::ColorCodes::resetColor;
        nParPerLine++;
      }
    }

    _convergenceMonitor_.setHeaderString(ssHeader.str());
    _convergenceMonitor_.getVariable("Total/dof").addQuantity(_owner_->getPropagator().getLlhBuffer() / double(_nbFitBins_ - _nbFreePars_));
    _convergenceMonitor_.getVariable("Total").addQuantity(_owner_->getPropagator().getLlhBuffer());
    _convergenceMonitor_.getVariable("Stat").addQuantity(_owner_->getPropagator().getLlhStatBuffer());
    _convergenceMonitor_.getVariable("Syst").addQuantity(_owner_->getPropagator().getLlhPenaltyBuffer());

    if( _nbFitCalls_ == 1 ){
      // don't erase these lines
      LogWarning << _convergenceMonitor_.generateMonitorString();
    }
    else{
      LogInfo << _convergenceMonitor_.generateMonitorString(
        GenericToolbox::getTerminalWidth() != 0, // trail back if not in batch mode
        true // force generate
      );
    }

    _itSpeed_.counts = _nbFitCalls_;
  }

  if( not GundamGlobals::isLightOutputMode() ){
    // Fill History
    _chi2HistoryTree_->Fill();
  }

  GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds("out_evalFit");

  // Protect the caller from an "infinite" value.  In principle, the caller
  // should correctly handle infinite values, but in practice they tend to
  // skimp on numeric safety in the pursuit of speed.  Truncate based on a
  // "float" so that it doesn't matter what type of real number the caller is
  // using.  (The maximum float is about 3.4E+38).
  double llh = _owner_->getPropagator().getLlhBuffer();
  if (llh > 0.1*std::numeric_limits<float>::max()) {
    lastCallOK = false;
    llh = 0.1*std::numeric_limits<float>::max();
    LogError << "Call: " << _nbFitCalls_
             << " "  << "Truncate LLH from "
             << _owner_->getPropagator().getLlhBuffer()
             << " to " << llh
             << std::endl;
  }

  return llh;
}

double LikelihoodInterface::evalFitValid(const double* parArray_) {
  // Check fit parameter values:
  const double *v = parArray_;
  for( auto* par : _minimizerFitParameterPtr_ ){
    if (not par->isValidValue(*(v++))) return std::numeric_limits<double>::infinity();
  }
  return evalFit(parArray_);
}

double LikelihoodInterface::getLastLikelihood() const {
  return _owner_->getPropagator().getLlhBuffer();
}

double LikelihoodInterface::getLastLikelihoodStat() const {
  return _owner_->getPropagator().getLlhStatBuffer();
}

double LikelihoodInterface::getLastLikelihoodPenalty() const {
  return _owner_->getPropagator().getLlhPenaltyBuffer();
}

void LikelihoodInterface::setParameterValidity(const std::string& validity) {
  return _owner_->getPropagator().getParametersManager().setParameterValidity(validity);
}

bool LikelihoodInterface::hasValidParameterValues() const {
  return _owner_->getPropagator().getParametersManager().hasValidParameterSets();
}

// An MIT Style License

// Copyright (c) 2022 GUNDAM DEVELOPERS

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Local Variables:
// mode:c++
// c-basic-offset:2
// End:
