//
// Created by Clark McGrew 24/1/23
//

#include "LikelihoodInterface.h"
#include "FitterEngine.h"
#include "GlobalVariables.h"

#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"
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
  _chi2HistoryTree_ = std::make_unique<TTree>("chi2History", "chi2History");
  _chi2HistoryTree_->SetDirectory(nullptr);
  _chi2HistoryTree_->Branch("nbFitCalls", &_nbFitCalls_);
  _chi2HistoryTree_->Branch("chi2Total", (double*) _owner_->getPropagator().getLlhBufferPtr());
  _chi2HistoryTree_->Branch("chi2Stat", (double*) _owner_->getPropagator().getLlhStatBufferPtr());
  _chi2HistoryTree_->Branch("chi2Pulls", (double*) _owner_->getPropagator().getLlhPenaltyBufferPtr());

  LogWarning << "Fetching the effective number of fit parameters..." << std::endl;
  _minimizerFitParameterPtr_.clear();
  _nbFreePars_ = 0;
  for( auto& parSet : _owner_->getPropagator().getParameterSetsList() ){
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
    _nbFitBins_ += int(sample.getBinning().getBinsList().size());
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
  GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(_owner_->getSaveDir(), "fit"), _chi2HistoryTree_.get());
}

double LikelihoodInterface::evalFit(const double* parArray_){
  LogThrowIf(not _isInitialized_, "not initialized");
  GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);

  if(_nbFitCalls_ != 0){
    _outEvalFitAvgTimer_.counts++ ; _outEvalFitAvgTimer_.cumulated += GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds("out_evalFit");
  }
  ++_nbFitCalls_;

  // Update fit parameter values:
  int iFitPar{0};
  for( auto* par : _minimizerFitParameterPtr_ ){
    if( getUseNormalizedFitSpace() ) par->setParameterValue(FitParameterSet::toRealParValue(parArray_[iFitPar++], *par));
    else par->setParameterValue(parArray_[iFitPar++]);
  }

  // Compute the Chi2
  _owner_->getPropagator().updateLlhCache();

  _evalFitAvgTimer_.counts++; _evalFitAvgTimer_.cumulated += GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);

  if(_enableFitMonitor_ && _convergenceMonitor_.isGenerateMonitorStringOk()){
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
    ssHeader << " / CPU: " << cpuPercent << "% (" << cpuPercent / GlobalVariables::getNbThreads() << "% efficiency)";
    ssHeader << std::endl << "Avg " << GUNDAM_CHI2 << " computation time: " << _evalFitAvgTimer_;
    ssHeader << std::endl;

    GenericToolbox::TablePrinter t;

    t << "" << GenericToolbox::TablePrinter::NextColumn;
    t << "Propagator" << GenericToolbox::TablePrinter::NextColumn;
    t << "Re-weight" << GenericToolbox::TablePrinter::NextColumn;
    t << "histograms fill" << GenericToolbox::TablePrinter::NextColumn;
    t << _minimizerType_ << "/" << _minimizerAlgo_ << GenericToolbox::TablePrinter::NextLine;

    t << "Speed" << GenericToolbox::TablePrinter::NextColumn;
    t << (double)_itSpeed_.counts / (double)_itSpeed_.cumulated * 1E6 << " it/s" << GenericToolbox::TablePrinter::NextColumn;
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
          if( nParPerLine >= _maxNbParametersPerLineOnMonitor_ ) { ssHeader << std::endl; nParPerLine = 0; }
        }
        if(fitPar->gotUpdated()) ssHeader << GenericToolbox::ColorCodes::blueBackground;
        if(getUseNormalizedFitSpace()) ssHeader << FitParameterSet::toNormalizedParValue(fitPar->getParameterValue(), *fitPar);
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

  // Fill History
  _chi2HistoryTree_->Fill();

  GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds("out_evalFit");
  return _owner_->getPropagator().getLlhBuffer();
}

double LikelihoodInterface::evalFitValid(const double* parArray_) {
  double value = evalFit(parArray_);
  if (hasValidParameterValues()) return value;
  /// A "Really Big Number".  This is nominally just infinity, but is done as
  /// a defined constant to make the code easier to understand.  This needs to
  /// be an appropriate value to safely represent an impossible chi-squared
  /// value "representing" -log(0.0)/2 and should should be larger than 5E+30.
  const double RBN = std::numeric_limits<double>::infinity();
  return RBN;
}

bool LikelihoodInterface::hasValidParameterValues() const {
  for (const FitParameterSet& parSet:
         _owner_->getPropagator().getParameterSetsList()) {
    for (const FitParameter& par : parSet.getParameterList()) {
      if (std::isfinite(par.getMinValue()) && par.getParameterValue() < par.getMinValue()) [[unlikely]] return false;
      if (std::isfinite(par.getMaxValue()) && par.getParameterValue() > par.getMaxValue()) [[unlikely]] return false;
    }
  }
  return true;
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
