#include "MinimizerBase.h"
#include "FitterEngine.h"

#ifdef GUNDAM_USING_CACHE_MANAGER
#include "CacheManager.h"
#endif


#include "Logger.h"


void MinimizerBase::configureImpl(){

  // Do not use checkFields here.  All of the fields should be checked in the
  // derived classes.

  // nested objects first
  int monitorRefreshRateInMs(5000);
  _config_.fillValue(monitorRefreshRateInMs, "monitorRefreshRateInMs");
  // slow down the refresh rate if in batch mode
  monitorRefreshRateInMs *= ( GenericToolbox::getTerminalWidth() != 0 ? 1 : 10 );
  _monitor_.convergenceMonitor.setMaxRefreshRateInMs( monitorRefreshRateInMs );

  // members
  _config_.fillValue(_monitor_.showParameters, "showParametersOnFitMonitor");
  _config_.fillValue(_monitor_.maxNbParametersPerLine, "maxNbParametersPerLineOnMonitor");
  _config_.fillValue(_isEnabledCalcError_, "enablePostFitErrorFit");
  _config_.fillValue(_useNormalizedFitSpace_, "useNormalizedFitSpace");
  _config_.fillValue(_writeLlhHistory_, "writeLlhHistory");
  _config_.fillValue(_checkParameterValidity_, "checkParameterValidity");
}
void MinimizerBase::initializeImpl(){

  _config_.printUnusedKeys();

  LogWarning << "Initializing MinimizerBase..." << std::endl;

  LogThrowIf(_owner_ == nullptr, "FitterEngine owner not set.");

  _nbFreeParameters_ = 0;
  _minimizerParameterPtrList_.clear();
  _minimizerParameterPtrList_.reserve( getLikelihoodInterface().getNbParameters() );
  for( auto& parSet : getModelPropagator().getParametersManager().getParameterSetsList() ){
    for( auto& par : parSet.getEffectiveParameterList() ){
      if( par.isEnabled() and not par.isFixed() ) {
        _minimizerParameterPtrList_.emplace_back( &par );
        if( par.isFree() ){ _nbFreeParameters_++; }
      }
    }
  }
  LogInfo << "Nb minimizer parameters: " << _minimizerParameterPtrList_.size() << std::endl;

  if( not GundamGlobals::isLightOutputMode() and _writeLlhHistory_ ){
    _monitor_.historyTree = std::make_unique<TTree>( "llhHistory", "llhHistory");
    _monitor_.historyTree->SetDirectory( nullptr ); // will be saved later
    _monitor_.historyTree->Branch("totalLikelihood", &getLikelihoodInterface().getBuffer().totalLikelihood);
    _monitor_.historyTree->Branch("statLikelihood", &getLikelihoodInterface().getBuffer().statLikelihood);
    _monitor_.historyTree->Branch("penaltyLikelihood", &getLikelihoodInterface().getBuffer().penaltyLikelihood);
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

  LogWarning << "MinimizerBase initialized." << std::endl;
}

void MinimizerBase::minimize(){
  /// An almost pure virtual method that is called by the FitterEngine to find the
  /// minimum of the likelihood, or, in the case of a Bayesian integration find
  /// the posterior distribution.
  ///
  /// This base implementation can be called in derived class in order to print
  /// the initial state of the fit.

  LogThrowIf(not isInitialized(), "not initialized");

  this->printParameters();

  getLikelihoodInterface().propagateAndEvalLikelihood();
  LogInfo << "Initial likelihood state:" << std::endl;
  LogInfo << getLikelihoodInterface().getSummary() << std::endl;

  LogInfo << "Number of defined parameters: " << getLikelihoodInterface().getNbParameters() << std::endl
          << "Number of fit parameters: " << _minimizerParameterPtrList_.size() << std::endl
          << "Number of fixed parameters: " << getLikelihoodInterface().getNbParameters() - _minimizerParameterPtrList_.size() << std::endl
          << "Number of free parameters: " << _nbFreeParameters_ << std::endl
          << "Number of fit bins: " << getLikelihoodInterface().getNbSampleBins() << std::endl
          << "Number of degree of freedom: " << fetchNbDegreeOfFreedom()
          << std::endl;

  LogWarning << std::endl << GenericToolbox::addUpDownBars("Calling minimize()...") << std::endl;
}
void MinimizerBase::calcErrors(){
  /// A virtual method that is called by the FiterEngine to calculate the
  /// errors at best fit point. By default it does nothing.
}
void MinimizerBase::scanParameters(TDirectory* saveDir_){
  /// A virtual method that by default scans the parameters used by the minimizer.
  /// This provides a view of the parameters seen by the minimizer, which may
  /// be different from the parameters used for the likelihood.

  LogInfo << "Performing scans of fit parameters..." << std::endl;
  LogThrowIf( not isInitialized() );
  for( auto& parPtr : _minimizerParameterPtrList_ ) { getParameterScanner().scanParameter( *parPtr, saveDir_ ); }
}
double MinimizerBase::evalFit( const double* parArray_ ){
/// The main access is through the evalFit method which takes an array of
/// floating point values and returns the likelihood. The meaning of the
/// parameters is defined by the vector of pointers to Parameter returned by
/// the LikelihoodInterface.

  _monitor_.externalTimer.stop();
  _monitor_.evalLlhTimer.start();

  // Check the fit parameter values.  Do this first so that the parameters
  // don't change when a bad set of values is tried with evalFit.  This will
  // only be enabled if the derived class has requested it.
  if (_checkParameterValidity_) {
    const double* v = parArray_;
    for( auto* par : _minimizerParameterPtrList_ ){
      double val = *(v++);
      if (_useNormalizedFitSpace_) val = ParameterSet::toRealParValue(val,*par);
      if (par->isValidValue(val)) continue;
      _monitor_.evalLlhTimer.stop();
      return std::numeric_limits<double>::infinity();
    }
  }

  // Looks OK, so update the parameter values.  The check for the
  // normalization outside of the loop so it runs a tiny bit faster.
  if (_useNormalizedFitSpace_) {
    const double* v = parArray_;
    for( auto* par : _minimizerParameterPtrList_ ){
      par->setParameterValue(ParameterSet::toRealParValue(*(v++),*par), true);
    }
  }
  else {
    const double* v = parArray_;
    for( auto* par : _minimizerParameterPtrList_ ){
      par->setParameterValue(*(v++), true);
    }
  }

  // Propagate the parameters
  getLikelihoodInterface().propagateAndEvalLikelihood();
  _monitor_.evalLlhTimer.stop();

  // Monitor if enabled
  if( _monitor_.isEnabled ){
    _monitor_.nbEvalLikelihoodCalls++;

    if( _monitor_.historyTree != nullptr ){ _monitor_.historyTree->Fill(); }
    if( _monitor_.convergenceMonitor.isGenerateMonitorStringOk() ){

      _monitor_.iterationCounterClock.count( _monitor_.nbEvalLikelihoodCalls );

      size_t nbValidPars = std::count_if(
              _minimizerParameterPtrList_.begin(), _minimizerParameterPtrList_.end(),
              [](const Parameter* par_){ return not ( par_->isFixed() or not par_->isEnabled() ); } );

      std::stringstream ssHeader;
      ssHeader << std::endl << GenericToolbox::getNowDateString("%Y.%m.%d %H:%M:%S");
      ssHeader << std::endl << __METHOD_NAME__ << ": call #" << _monitor_.nbEvalLikelihoodCalls;
      ssHeader << std::endl << _monitor_.stateTitleMonitor << " / Nb of parameters to fit: " << nbValidPars;
      ssHeader << std::endl << "RAM: " << GenericToolbox::parseSizeUnits(double(GenericToolbox::getProcessMemoryUsage()));
      double cpuPercent = GenericToolbox::getCpuUsageByProcess();
      ssHeader << " / CPU: " << cpuPercent << "% / " << cpuPercent / GundamGlobals::getNbCpuThreads() << "% efficiency";
      ssHeader << std::endl << "Avg log-likelihood computation time: " << _monitor_.evalLlhTimer;
      ssHeader << std::endl;

      GenericToolbox::TablePrinter t;

      t << "" << GenericToolbox::TablePrinter::NextColumn;
      t << "Propagator" << GenericToolbox::TablePrinter::NextColumn;

#ifdef GUNDAM_USING_CACHE_MANAGER
      if( Cache::Manager::Get() != nullptr ){
        t << "Cache::Fill" << GenericToolbox::TablePrinter::NextColumn;
        t << "Pull from device" << GenericToolbox::TablePrinter::NextColumn;
      }
      else{
#endif
        t << "Re-weight" << GenericToolbox::TablePrinter::NextColumn;
        t << "histograms fill" << GenericToolbox::TablePrinter::NextColumn;
#ifdef GUNDAM_USING_CACHE_MANAGER
      }
#endif
      t << _monitor_.minimizerTitle << GenericToolbox::TablePrinter::NextLine;

      t << "Speed" << GenericToolbox::TablePrinter::NextColumn;
      t << _monitor_.iterationCounterClock.evalTickSpeed() << " it/s" << GenericToolbox::TablePrinter::NextColumn;


#ifdef GUNDAM_USING_CACHE_MANAGER
      if( Cache::Manager::Get() != nullptr ){
        t << Cache::Manager::GetCacheFillTimer() << GenericToolbox::TablePrinter::NextColumn;
        t << Cache::Manager::GetPullFromDeviceTimer() << GenericToolbox::TablePrinter::NextColumn;
      }
      else{
#endif
        t << getModelPropagator().reweightTimer << GenericToolbox::TablePrinter::NextColumn;
        t << getModelPropagator().refillHistogramTimer << GenericToolbox::TablePrinter::NextColumn;
#ifdef GUNDAM_USING_CACHE_MANAGER
      }
#endif
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
                     << (fitPar->getOwner()->isEnableEigenDecomp() ? " (eigen)" : "")
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
          getLikelihoodInterface().getLastLikelihood() / fetchNbDegreeOfFreedom()
      );
      _monitor_.convergenceMonitor.getVariable("Total").addQuantity( getLikelihoodInterface().getLastLikelihood() );
      _monitor_.convergenceMonitor.getVariable("Stat").addQuantity( getLikelihoodInterface().getLastStatLikelihood() );
      _monitor_.convergenceMonitor.getVariable("Syst").addQuantity( getLikelihoodInterface().getLastPenaltyLikelihood() );

      if( _monitor_.nbEvalLikelihoodCalls == 1 ){
        // don't erase these lines
        LogWarning << _monitor_.convergenceMonitor.generateMonitorString(false, true);
      }
      else{
        std::cout << _monitor_.convergenceMonitor.generateMonitorString(
            GenericToolbox::getTerminalWidth() != 0, // trail back if not in batch mode
            true // force generate
        );
      }

      // in some ROOT versions the trail back does not work properly. This seems to fix the issue
      Logger::moveTerminalCursorBack(1);
      std::cout << std::endl;

    }
  }

  if( _throwOnBadLlh_ and not getLikelihoodInterface().getBuffer().isValid() ){
    LogError << getLikelihoodInterface().getSummary() << std::endl;
    LogThrow( "Invalid total likelihood value." );
  }

  _monitor_.externalTimer.start();
  return getLikelihoodInterface().getLastLikelihood();
}

void MinimizerBase::printParameters(){
  // This prints the same set of parameters as are in the vector returned by
  // getMinimizerFitParameterPtr(), but does it by parameter set so that the
  // output is a little more clear.

  LogWarning << std::endl << GenericToolbox::addUpDownBars("Summary of the fit parameters:") << std::endl;
  for( const auto& parSet : getModelPropagator().getParametersManager().getParameterSetsList() ){

    GenericToolbox::TablePrinter t;
    t.setColTitles({ {"Title"}, {"Starting"}, {"Prior"}, {"StdDev"}, {"Limits"}, {"Status"} });

    auto& parList = parSet.getEffectiveParameterList();
    LogWarning << parSet.getName() << ": " << parList.size() << " parameters" << std::endl;
    if( parList.empty() ) continue;

    for( const auto& par : parList ){
      std::string colorStr;
      std::string statusStr;

      if( not par.isEnabled() ) { continue; }
      else if( par.isFixed() )  { statusStr = "Fixed (prior applied)";    colorStr = GenericToolbox::ColorCodes::yellowLightText; }
      else                      {
        statusStr = Parameter::PriorType::toString(par.getPriorType()) + " Prior";
        if(par.getPriorType()==Parameter::PriorType::Flat) colorStr = GenericToolbox::ColorCodes::blueLightText;
      }

#ifdef NOCOLOR
      colorStr = "";
#endif

      t.addTableLine({
                         par.getTitle(),
                         std::to_string( par.getParameterValue() ),
                         std::to_string( par.getPriorValue() ),
                         std::to_string( par.getStdDevValue() ),
                         par.getParameterLimits().toString(),
                         statusStr
                     }, colorStr);
    }

    t.printTable();
  }
}


Propagator& MinimizerBase::getModelPropagator(){ return _owner_->getLikelihoodInterface().getModelPropagator(); }
[[nodiscard]] const Propagator& MinimizerBase::getModelPropagator() const { return _owner_->getLikelihoodInterface().getModelPropagator(); }
ParameterScanner& MinimizerBase::getParameterScanner(){ return _owner_->getParameterScanner(); }
[[nodiscard]] const ParameterScanner& MinimizerBase::getParameterScanner() const { return _owner_->getParameterScanner(); }
LikelihoodInterface& MinimizerBase::getLikelihoodInterface(){ return _owner_->getLikelihoodInterface(); }
[[nodiscard]] const LikelihoodInterface& MinimizerBase::getLikelihoodInterface() const{ return _owner_->getLikelihoodInterface(); }


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
