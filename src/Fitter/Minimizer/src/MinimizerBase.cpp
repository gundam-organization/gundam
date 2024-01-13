//
// Created by Clark McGrew on 25/01/2023.
//

#include "MinimizerBase.h"
#include "FitterEngine.h"

#include "GenericToolbox.Json.h"
#include "Logger.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[MinimizerBase]");
});


void MinimizerBase::readConfigImpl(){
  LogInfo << "Reading MinimizerBase config..." << std::endl;

  _monitor_.showParameters = GenericToolbox::Json::fetchValue(_config_, "showParametersOnFitMonitor", _monitor_.showParameters);
  _monitor_.maxNbParametersPerLine = GenericToolbox::Json::fetchValue(_config_, "maxNbParametersPerLineOnMonitor", _monitor_.maxNbParametersPerLine);
  _monitor_.convergenceMonitor.setMaxRefreshRateInMs(
      GenericToolbox::Json::fetchValue( _config_, "monitorRefreshRateInMs", int(5000) )
      * ( GenericToolbox::getTerminalWidth() != 0 ? 1 : 5 ) // slow down the refresh rate if in batch mode
  );

  _useNormalizedFitSpace_ = GenericToolbox::Json::fetchValue(_config_, "useNormalizedFitSpace", _useNormalizedFitSpace_);

}
void MinimizerBase::initializeImpl(){
  LogInfo << "Initializing the minimizer..." << std::endl;
  LogThrowIf(_owner_ == nullptr, "FitterEngine owner not set.");

  _nbFreeParameters_ = 0;
  _minimizerParameterPtrList_.clear();
  _minimizerParameterPtrList_.reserve( _owner_->getLikelihoodInterface().getNbParameters() );
  for( auto& parSet : _owner_->getPropagator().getParametersManager().getParameterSetsList() ){
    for( auto& par : parSet.getEffectiveParameterList() ){
      if( par.isEnabled() and not par.isFixed() ) {
        _minimizerParameterPtrList_.emplace_back( &par );
        if( par.isFree() ){ _nbFreeParameters_++; }
      }
    }
  }

  if( not GundamGlobals::isLightOutputMode() ){
    _monitor_.historyTree = std::make_unique<TTree>( "chi2History", "chi2History");
    _monitor_.historyTree->SetDirectory( nullptr ); // will be saved later
    _monitor_.historyTree->Branch("nbEvalLikelihoodCalls", &_monitor_.nbEvalLikelihoodCalls);
    _monitor_.historyTree->Branch("totalLikelihood", &_owner_->getLikelihoodInterface().getBuffer().totalLikelihood);
    _monitor_.historyTree->Branch("statLikelihood", &_owner_->getLikelihoodInterface().getBuffer().statLikelihood);
    _monitor_.historyTree->Branch("penaltyLikelihood", &_owner_->getLikelihoodInterface().getBuffer().penaltyLikelihood);
    _monitor_.historyTree->Branch("iterationSpeed", _monitor_.itSpeedMon.getCountSpeedPtr());
  }

  _monitor_.convergenceMonitor.addDisplayedQuantity("VarName");
  _monitor_.convergenceMonitor.addDisplayedQuantity("LastAddedValue");
  _monitor_.convergenceMonitor.addDisplayedQuantity("SlopePerCall");

  _monitor_.convergenceMonitor.getQuantity("VarName").title = "Likelihood";
  _monitor_.convergenceMonitor.getQuantity("LastAddedValue").title = "Current Value";
  _monitor_.convergenceMonitor.getQuantity("SlopePerCall").title = "Avg. Slope /call";

  _monitor_.convergenceMonitor.addVariable("Total/dof");
  _monitor_.convergenceMonitor.addVariable("Total");
  _monitor_.convergenceMonitor.addVariable("Stat");
  _monitor_.convergenceMonitor.addVariable("Syst");

}

// const getters
const FitterEngine& MinimizerBase::getOwner() const { return *_owner_; }
const Propagator& MinimizerBase::getPropagator() const{ return _owner_->getPropagator(); }
const ParameterScanner& MinimizerBase::getParameterScanner() const { return _owner_->getParameterScanner(); }
const LikelihoodInterface& MinimizerBase::getLikelihoodInterface() const{ return _owner_->getLikelihoodInterface(); }

// mutable getters
FitterEngine& MinimizerBase::getOwner(){ return *_owner_; }
Propagator& MinimizerBase::getPropagator(){ return _owner_->getPropagator(); }
ParameterScanner& MinimizerBase::getParameterScanner(){ return _owner_->getParameterScanner(); }
LikelihoodInterface& MinimizerBase::getLikelihoodInterface(){ return _owner_->getLikelihoodInterface(); }

void MinimizerBase::minimize(){
  LogThrowIf(not isInitialized(), "not initialized");

  this->printParameters();

  getLikelihoodInterface().propagateAndEvalLikelihood();
  LogInfo << "Initial likelihood state:" << getLikelihoodInterface().getSummary();

  LogInfo << "Number of defined parameters: " << getLikelihoodInterface().getNbParameters() << std::endl
  << "Number of fit parameters: " << _minimizerParameterPtrList_.size() << std::endl
  << "Number of fixed parameters: " << getLikelihoodInterface().getNbParameters() - _minimizerParameterPtrList_.size() << std::endl
  << "Number of free parameters: " << _nbFreeParameters_ << std::endl
  << "Number of fit bins: " << getLikelihoodInterface().getNbSampleBins() << std::endl
  << "Number of degree of freedom: " << getNbDegreeOfFreedom()
  << std::endl;

  LogWarning << std::endl << GenericToolbox::addUpDownBars("Calling minimize...") << std::endl;
}
void MinimizerBase::scanParameters(TDirectory* saveDir_){
  LogInfo << "Performing scans of fit parameters..." << std::endl;
  LogThrowIf( not isInitialized() );
  for( auto& parPtr : _minimizerParameterPtrList_ ) { getParameterScanner().scanParameter( *parPtr, saveDir_ ); }
}
double MinimizerBase::evalFit( const double* parArray_ ){
  _monitor_.externalTimer.stop();

  // Update fit parameter values:
  int iFitPar{0};
  for( auto* parPtr : _minimizerParameterPtrList_ ){
    parPtr->setParameterValue(
        _useNormalizedFitSpace_ ?
        ParameterSet::toRealParValue(parArray_[iFitPar++], *parPtr) :
        parArray_[iFitPar++]
    );
  }

  // Propagate the parameters
  getLikelihoodInterface().propagateAndEvalLikelihood();

  // Monitor if enabled
  if( _monitor_.isEnabled ){
    if( _monitor_.historyTree != nullptr ){ _monitor_.historyTree->Fill(); }
    if( _monitor_.gradientDescentMonitor.isEnabled ){

      auto& gradient = _monitor_.gradientDescentMonitor;

      // When gradient descent base minimizer probe a point toward the minimum, every parameter get updated
      bool isGradientDescentStep =
          std::all_of(
              _minimizerParameterPtrList_.begin(), _minimizerParameterPtrList_.end(),
              [](const Parameter* par_){
                return ( par_->gotUpdated() or par_->isFixed() or not par_->isEnabled() );
              } );
      if( isGradientDescentStep ){

        if( gradient.lastGradientFall == _monitor_.nbEvalLikelihoodCalls - 1 ){
          LogWarning << "Overriding last gradient descent entry (minimizer adjusting step size...): ";
        }
        else{
          gradient.stepPointList.emplace_back();
          LogWarning << "Gradient step detected at iteration #" << _monitor_.nbEvalLikelihoodCalls << ": ";
        }
        LogWarningIf(gradient.stepPointList.size() >= 2) << gradient.stepPointList[gradient.stepPointList.size() - 2].llh << " -> ";
        LogWarning << getLikelihoodInterface().getBuffer().totalLikelihood << std::endl;
        gradient.stepPointList.back().parState = getPropagator().getParametersManager().exportParameterInjectorConfig();
        gradient.stepPointList.back().llh = getLikelihoodInterface().getBuffer().totalLikelihood;
        gradient.lastGradientFall = _monitor_.nbEvalLikelihoodCalls;
      }
    }
    if( _monitor_.convergenceMonitor.isGenerateMonitorStringOk() ){

      _monitor_.itSpeedMon.cycle( _monitor_.nbEvalLikelihoodCalls - _monitor_.itSpeedMon.getCounts() );

      if( _monitor_.itSpeed.counts != 0 ){
        _monitor_.itSpeed.counts = _monitor_.nbEvalLikelihoodCalls - _monitor_.itSpeed.counts; // how many cycles since last print
        _monitor_.itSpeed.cumulated = GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds("itSpeed"); // time since last print
      }
      else{
        _monitor_.itSpeed.counts = _monitor_.nbEvalLikelihoodCalls;
        GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds("itSpeed");
      }

      std::stringstream ssHeader;
      ssHeader << std::endl << __METHOD_NAME__ << ": call #" << _monitor_.nbEvalLikelihoodCalls;
      ssHeader << std::endl << _monitor_.stateTitleMonitor;
//    ssHeader << std::endl << "Target EDM: " << getMinimizer().get;
      ssHeader << std::endl << "RAM: " << GenericToolbox::parseSizeUnits(double(GenericToolbox::getProcessMemoryUsage()));
      double cpuPercent = GenericToolbox::getCpuUsageByProcess();
      ssHeader << " / CPU: " << cpuPercent << "% (" << cpuPercent / GundamGlobals::getParallelWorker().getNbThreads() << "% efficiency)";
      ssHeader << std::endl << "Avg " << GUNDAM_CHI2 << " computation time: " << _monitor_.evalLlhTimer;
      ssHeader << std::endl;

      GenericToolbox::TablePrinter t;

      t << "" << GenericToolbox::TablePrinter::NextColumn;
      t << "Propagator" << GenericToolbox::TablePrinter::NextColumn;
      t << "Re-weight" << GenericToolbox::TablePrinter::NextColumn;
      t << "histograms fill" << GenericToolbox::TablePrinter::NextColumn;
      t << _monitor_.minimizerTitle << GenericToolbox::TablePrinter::NextLine;

      t << "Speed" << GenericToolbox::TablePrinter::NextColumn;
      t << _monitor_.itSpeedMon << GenericToolbox::TablePrinter::NextColumn;
//    t << (double)_monitor_.itSpeed.counts / (double)_monitor_.itSpeed.cumulated * 1E6 << " it/s" << GenericToolbox::TablePrinter::NextColumn;
      t << getPropagator().weightProp << GenericToolbox::TablePrinter::NextColumn;
      t << getPropagator().fillProp << GenericToolbox::TablePrinter::NextColumn;
      t << _monitor_.externalTimer << GenericToolbox::TablePrinter::NextLine;

      ssHeader << t.generateTableString();

      if( _monitor_.showParameters ){
        std::string curParSet;
        ssHeader << std::endl << std::setprecision(1) << std::scientific << std::showpos;
        int nParPerLine{0};
        for( auto* fitPar : _minimizerParameterPtrList_ ){
          if( fitPar->isFixed() ) continue;
          if( curParSet != fitPar->getOwner()->getName() ){
            if( not curParSet.empty() ) ssHeader << std::endl;
            curParSet = fitPar->getOwner()->getName();
            ssHeader << curParSet
                     << (fitPar->getOwner()->useEigenDecomposition() ? " (eigen)" : "")
                     << ":" << std::endl;
            nParPerLine = 0;
          }
          else{
            ssHeader << ", ";
            if( nParPerLine >= _monitor_.maxNbParametersPerLine ) {
              ssHeader << std::endl; nParPerLine = 0;
            }
          }
          if(fitPar->gotUpdated()) ssHeader << GenericToolbox::ColorCodes::blueBackground;
          if(_useNormalizedFitSpace_){
            ssHeader << ParameterSet::toNormalizedParValue(fitPar->getParameterValue(), *fitPar);
          }
          else{ ssHeader << fitPar->getParameterValue(); }
          if(fitPar->gotUpdated()) ssHeader << GenericToolbox::ColorCodes::resetColor;
          nParPerLine++;
        }
      }

      _monitor_.convergenceMonitor.setHeaderString(ssHeader.str());
      _monitor_.convergenceMonitor.getVariable("Total/dof").addQuantity(
          getLikelihoodInterface().getBuffer().totalLikelihood / getNbDegreeOfFreedom()
      );
      _monitor_.convergenceMonitor.getVariable("Total").addQuantity( getLikelihoodInterface().getBuffer().totalLikelihood );
      _monitor_.convergenceMonitor.getVariable("Stat").addQuantity( getLikelihoodInterface().getBuffer().statLikelihood );
      _monitor_.convergenceMonitor.getVariable("Syst").addQuantity( getLikelihoodInterface().getBuffer().penaltyLikelihood );

      if( _monitor_.nbEvalLikelihoodCalls == 1 ){
        // don't erase these lines
        LogWarning << _monitor_.convergenceMonitor.generateMonitorString();
      }
      else{
        LogInfo << _monitor_.convergenceMonitor.generateMonitorString(
            GenericToolbox::getTerminalWidth() != 0, // trail back if not in batch mode
            true // force generate
        );
      }

      _monitor_.itSpeed.counts = _monitor_.nbEvalLikelihoodCalls;
    }
  }

  if( _throwOnBadLlh_ and not getLikelihoodInterface().getBuffer().isValid() ){
    LogError << getLikelihoodInterface().getSummary() << std::endl;
    LogThrow( "Invalid total likelihood value." );
  }

  _monitor_.externalTimer.start();
  return getLikelihoodInterface().getBuffer().totalLikelihood;
}


void MinimizerBase::printParameters(){
  // This prints the same set of parameters as are in the vector returned by
  // getMinimizerFitParameterPtr(), but does it by parameter set so that the
  // output is a little more clear.

  LogWarning << std::endl << GenericToolbox::addUpDownBars("Summary of the fit parameters:") << std::endl;
  for( const auto& parSet : _owner_->getPropagator().getParametersManager().getParameterSetsList() ){

    GenericToolbox::TablePrinter t;
    t.setColTitles({ {"Title"}, {"Starting"}, {"Prior"}, {"StdDev"}, {"Min"}, {"Max"}, {"Status"} });

    auto& parList = parSet.getEffectiveParameterList();
    LogWarning << parSet.getName() << ": " << parList.size() << " parameters" << std::endl;
    if( parList.empty() ) continue;

    for( const auto& par : parList ){
      std::string colorStr;
      std::string statusStr;

      if( not par.isEnabled() ) { statusStr = "Disabled"; colorStr = GenericToolbox::ColorCodes::yellowBackground; }
      else if( par.isFixed() )  { statusStr = "Fixed";    colorStr = GenericToolbox::ColorCodes::redBackground; }
      else                      {
        statusStr = PriorType::toString(par.getPriorType()) + " Prior";
        if(par.getPriorType()==PriorType::Flat) colorStr = GenericToolbox::ColorCodes::blueBackground;
      }

#ifdef NOCOLOR
      colorStr = "";
#endif

      t.addTableLine({
                         par.getTitle(),
                         std::to_string( par.getParameterValue() ),
                         std::to_string( par.getPriorValue() ),
                         std::to_string( par.getStdDevValue() ),
                         std::to_string( par.getMinValue() ),
                         std::to_string( par.getMaxValue() ),
                         statusStr
                     }, colorStr);
    }

    t.printTable();
  }
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
// compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
// End:
