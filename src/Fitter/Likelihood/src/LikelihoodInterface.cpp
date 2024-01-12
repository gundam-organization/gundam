//
// Created by Clark McGrew 24/1/23
//

#include "LikelihoodInterface.h"
#include "FitterEngine.h"
#include "GundamGlobals.h"

#include "GenericToolbox.Utils.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.Json.h"
#include "Logger.h"

#include <limits>


LoggerInit([]{
  Logger::setUserHeaderStr("[Likelihood]");
});


void LikelihoodInterface::initializeImpl() {

  if( not GundamGlobals::isLightOutputMode() ){
    _chi2HistoryTree_ = std::make_unique<TTree>("chi2History", "chi2History");
    _chi2HistoryTree_->SetDirectory(nullptr);
    _chi2HistoryTree_->Branch("nbFitCalls", &_nbLlhEvals_);
    _chi2HistoryTree_->Branch("chi2Total", (double*) _propagator_.getLlhBufferPtr());
    _chi2HistoryTree_->Branch("chi2Stat", (double*) _propagator_.getLlhStatBufferPtr());
    _chi2HistoryTree_->Branch("chi2Pulls", (double*) _propagator_.getLlhPenaltyBufferPtr());
    _chi2HistoryTree_->Branch("itSpeed", _itSpeedMon_.getCountSpeedPtr());
  }

  LogWarning << "Fetching the effective number of fit parameters..." << std::endl;
  _minimizerFitParameterPtr_.clear();
  _nbFreePars_ = 0;
  for( auto& parSet : _propagator_.getParametersManager().getParameterSetsList() ){
    for( auto& par : parSet.getEffectiveParameterList() ){
      if( par.isEnabled() and not par.isFixed() ) {
        _minimizerFitParameterPtr_.emplace_back(&par);
        if( par.isFree() ) _nbFreePars_++;
      }
    }
  }
  _nbFitParameters_ = int(_minimizerFitParameterPtr_.size());

  LogInfo << "Building functor with " << _nbFitParameters_ << " parameters ..." << std::endl;
  _functor_ = std::make_unique<ROOT::Math::Functor>(this, &LikelihoodInterface::evalLikelihood, _nbFitParameters_);
  _validFunctor_ = std::make_unique<ROOT::Math::Functor>(this, &LikelihoodInterface::evalFitValid, _nbFitParameters_);

  _nbFitBins_ = 0;
  for( auto& sample : _propagator_.getFitSampleSet().getSampleList() ){
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


}

void LikelihoodInterface::writeChi2History() {
  LogReturnIf( GundamGlobals::isLightOutputMode(), "Not saving LLH history as light output mode is fired." );

  LogInfo << "Saving LLH history..." << std::endl;
  GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(_owner_->getSaveDir(), "fit"), _chi2HistoryTree_.get());
}
void LikelihoodInterface::saveGradientSteps(){

  if( GundamGlobals::isLightOutputMode() ){
    LogAlert << "Skipping saveGradientSteps as light output mode is fired." << std::endl;
    return;
  }

  LogInfo << "Saving " << _gradientMonitor_.size() << " gradient steps..." << std::endl;

  // make sure the parameter states get restored as we leave
  auto currentParState = _propagator_.getParametersManager().exportParameterInjectorConfig();
  GenericToolbox::ScopedGuard g{
    [&](){
      ParametersManager::muteLogger();
      ParameterSet::muteLogger();
      ParScanner::muteLogger();
    },
    [&](){
      _propagator_.getParametersManager().injectParameterValues( currentParState );
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
    _propagator_.getParametersManager().injectParameterValues(_gradientMonitor_[iGradStep].parState );
    _propagator_.updateLlhCache();

    if( not GundamGlobals::isLightOutputMode() ) {
      auto outDir = GenericToolbox::mkdirTFile(_owner_->getSaveDir(), Form("fit/gradient/step_%i", int(iGradStep)));
      GenericToolbox::writeInTFile(outDir, TNamed("parState", GenericToolbox::Json::toReadableString(_gradientMonitor_[iGradStep].parState).c_str()));
      GenericToolbox::writeInTFile(outDir, TNamed("llhState", _propagator_.getLlhBufferSummary().c_str()));
    }

    // line scan from previous point
    _propagator_.getParScanner().scanSegment( nullptr, _gradientMonitor_[iGradStep].parState, lastParStep, 8 );
    lastParStep = _gradientMonitor_[iGradStep].parState;

    if( globalGraphList.empty() ){
      // copy
      globalGraphList = _propagator_.getParScanner().getGraphEntriesBuf();
    }
    else{
      // current
      auto& grEntries = _propagator_.getParScanner().getGraphEntriesBuf();

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

double LikelihoodInterface::evalLikelihood( const double* parArray_ ){
  GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);

  if( _nbLlhEvals_ != 0){
    _outEvalFitAvgTimer_.counts++ ; _outEvalFitAvgTimer_.cumulated += GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds("out_evalFit");
  }
  ++_nbLlhEvals_;

  // Update fit parameter values:
  int iFitPar{0};
  for( auto* par : _minimizerFitParameterPtr_ ){
    if( getUseNormalizedFitSpace() ) par->setParameterValue(ParameterSet::toRealParValue(parArray_[iFitPar++], *par));
    else par->setParameterValue(parArray_[iFitPar++]);
  }


  _propagator_.propagateParameters();

  this->evalStatLikelihood();

  _buffer_.penaltyLikelihood = 0;
  for( auto& parSet : _propagator_.getParametersManager().getParameterSetsList() ){
    _buffer_.penaltyLikelihood += parSet.getPenaltyChi2();
  }

  _buffer_.regulariseLikelihood = 0; // unused

  _buffer_.totalLikelihood = _buffer_.statLikelihood + _buffer_.penaltyLikelihood + _buffer_.regulariseLikelihood;

  _evalFitAvgTimer_.counts++; _evalFitAvgTimer_.cumulated += GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);

  // Minuit based algo might want this
  if( _monitorGradientDescent_ ){
    // check if minuit is moving toward the minimum
    bool isGradientDescentStep = std::all_of(_minimizerFitParameterPtr_.begin(), _minimizerFitParameterPtr_.end(), [](const Parameter* par_){
      return ( par_->gotUpdated() or par_->isFixed() or not par_->isEnabled() );
    } );
    if( isGradientDescentStep ){
      if( _lastGradientFall_ == _nbLlhEvals_ - 1 ){
        LogWarning << "Overriding last gradient descent entry (minimizer adjusting step size...): ";
        LogWarning(_gradientMonitor_.size() >= 2) << _gradientMonitor_[_gradientMonitor_.size() - 2].llh << " -> ";
        LogWarning << _propagator_.getLlhBuffer() << std::endl;
        _gradientMonitor_.back().parState = _propagator_.getParametersManager().exportParameterInjectorConfig();
        _gradientMonitor_.back().llh = _propagator_.getLlhBuffer();
        _lastGradientFall_ = _nbLlhEvals_;
      }
      else{
        // saving each step of the gradient descen
        _gradientMonitor_.emplace_back();
        LogWarning << "Gradient step detected at iteration #" << _nbLlhEvals_ << ": ";
        LogWarning(_gradientMonitor_.size() >= 2) << _gradientMonitor_[_gradientMonitor_.size() - 2].llh << " -> ";
        LogWarning << _propagator_.getLlhBuffer() << std::endl;
        _gradientMonitor_.back().parState = _propagator_.getParametersManager().exportParameterInjectorConfig();
        _gradientMonitor_.back().llh = _propagator_.getLlhBuffer();
        _lastGradientFall_ = _nbLlhEvals_;
      }
    }
  }

  if( _fitMonitor_.isEnabled and _convergenceMonitor_.isGenerateMonitorStringOk() ){

    _itSpeedMon_.cycle(_nbLlhEvals_ - _itSpeedMon_.getCounts() );

    if( _itSpeed_.counts != 0 ){
      _itSpeed_.counts = _nbLlhEvals_ - _itSpeed_.counts; // how many cycles since last print
      _itSpeed_.cumulated = GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds("itSpeed"); // time since last print
    }
    else{
      _itSpeed_.counts = _nbLlhEvals_;
      GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds("itSpeed");
    }

    std::stringstream ssHeader;
    ssHeader << std::endl << __METHOD_NAME__ << ": call #" << _nbLlhEvals_;
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
    t << _propagator_.weightProp << GenericToolbox::TablePrinter::NextColumn;
    t << _propagator_.fillProp << GenericToolbox::TablePrinter::NextColumn;
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
    _convergenceMonitor_.getVariable("Total/dof").addQuantity(_propagator_.getLlhBuffer() / double(_nbFitBins_ - _nbFreePars_));
    _convergenceMonitor_.getVariable("Total").addQuantity(_propagator_.getLlhBuffer());
    _convergenceMonitor_.getVariable("Stat").addQuantity(_propagator_.getLlhStatBuffer());
    _convergenceMonitor_.getVariable("Syst").addQuantity(_propagator_.getLlhPenaltyBuffer());

    if( _nbLlhEvals_ == 1 ){
      // don't erase these lines
      LogWarning << _convergenceMonitor_.generateMonitorString();
    }
    else{
      LogInfo << _convergenceMonitor_.generateMonitorString(
        GenericToolbox::getTerminalWidth() != 0, // trail back if not in batch mode
        true // force generate
      );
    }

    _itSpeed_.counts = _nbLlhEvals_;
  }

  if( not GundamGlobals::isLightOutputMode() ){
    // Fill History
    _chi2HistoryTree_->Fill();
  }

  if( std::isnan(_propagator_.getLlhBuffer()) and _throwOnBadLlh_ ){
    LogError << _propagator_.getLlhBuffer() << ": invalid reported likelihood value." << std::endl;
    LogError << GET_VAR_NAME_VALUE( _propagator_.getLlhStatBuffer() ) << std::endl;
    for( auto& sample : _propagator_.getFitSampleSet().getSampleList() ){
      LogScopeIndent;
      LogError << sample.getName() << ": " << sample.getLlhStatBuffer() << std::endl;
    }
    LogError << GET_VAR_NAME_VALUE( _propagator_.getLlhPenaltyBuffer() ) << std::endl;
    LogThrow("Invalid total llh.");
  }

  GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds("out_evalFit");
  return _propagator_.getLlhBuffer();
}
double LikelihoodInterface::evalStatLikelihood(){

  _buffer_.statLikelihood = _propagator_.getFitSampleSet().evalLikelihood();

  _buffer_.statLikelihood = 0.;
  for( auto& sample : _propagator_.getFitSampleSet().getSampleList() ){
    // TODO: keep buffer per sample
    _buffer_.statLikelihood += _jointProbabilityPtr_->eval( sample );
  }
  return _buffer_.statLikelihood;

}

double LikelihoodInterface::evalFitValid(const double* parArray_) {
  double value = evalLikelihood(parArray_);
  if (hasValidParameterValues()) return value;
  /// A "Really Big Number".  This is nominally just infinity, but is done as
  /// a defined constant to make the code easier to understand.  This needs to
  /// be an appropriate value to safely represent an impossible chi-squared
  /// value "representing" -log(0.0)/2 and should should be larger than 5E+30.
  const double RBN = std::numeric_limits<double>::infinity();
  return RBN;
}

void LikelihoodInterface::setParameterValidity(const std::string& validity) {
  /// Define the type of validity that needs to be required by
  /// hasValidParameterValues.  This accepts a string with the possible values
  /// being:
  ///
  ///  "range" (default) -- Between the parameter minimum and maximum values.
  ///  "norange"         -- Do not require parameters in the valid range
  ///  "mirror"          -- Between the mirrored values (if parameter has
  ///                       mirroring).
  ///  "nomirror"        -- Do not require parameters in the mirrored range
  ///  "physical"        -- Only physically meaningful values.
  ///  "nophysical"      -- Do not require parameters in the physical range.
  ///
  /// Example: setParameterValidity("range,mirror,physical")

  LogWarning << "Set parameter validity to " << validity << std::endl;
  if (validity.find("noran") != std::string::npos) _validFlags_ &= ~0b0001;
  else if (validity.find("ran") != std::string::npos) _validFlags_ |= 0b0001;
  if (validity.find("nomir") != std::string::npos) _validFlags_ &= ~0b0010;
  else if (validity.find("mir") != std::string::npos) _validFlags_ |= 0b0010;
  if (validity.find("nophy") != std::string::npos) _validFlags_ &= ~0b0100;
  else if (validity.find("phy") != std::string::npos) _validFlags_ |= 0b0100;
  LogWarning << "Set parameter validity to " << validity
             << " (" << _validFlags_ << ")" << std::endl;
}

bool LikelihoodInterface::hasValidParameterValues() const {
  int invalid = 0;
  for (const ParameterSet& parSet:
         _propagator_.getParametersManager().getParameterSetsList()) {
    for (const Parameter& par : parSet.getParameterList()) {
      if ( (_validFlags_ & 0b0001) != 0
          and std::isfinite(par.getMinValue())
          and par.getParameterValue() < par.getMinValue()) [[unlikely]] {
        ++invalid;
      }
      if ((_validFlags_ & 0b0001) != 0
          and std::isfinite(par.getMaxValue())
          and par.getParameterValue() > par.getMaxValue()) [[unlikely]] {
        ++invalid;
      }
      if ((_validFlags_ & 0b0010) != 0
          and std::isfinite(par.getMinMirror())
          and par.getParameterValue() < par.getMinMirror()) [[unlikely]] {
        ++invalid;
      }
      if ((_validFlags_ & 0b0010) != 0
          and std::isfinite(par.getMaxMirror())
          and par.getParameterValue() > par.getMaxMirror()) [[unlikely]] {
        ++invalid;
      }
      if ((_validFlags_ & 0b0100) != 0
          and std::isfinite(par.getMinPhysical())
          and par.getParameterValue() < par.getMinPhysical()) [[unlikely]] {
        ++invalid;
      }
      if ((_validFlags_ & 0b0100) != 0
          and std::isfinite(par.getMaxPhysical())
          and par.getParameterValue() > par.getMaxPhysical()) [[unlikely]] {
        ++invalid;
      }

    }
  }
  return (invalid == 0);
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
// compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
// End:
