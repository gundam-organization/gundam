//
// Created by Nadrino on 07/04/2022.
//

#include "ParameterScanner.h"
#include "Propagator.h"
#include "Parameter.h"

#include "GenericToolbox.Utils.h"
#include "GenericToolbox.Json.h"
#include "Logger.h"

#include "TGraph.h"
#include <TDirectory.h>

#include <utility>


#ifndef DISABLE_USER_HEADER
LoggerInit([]{ Logger::setUserHeaderStr("[ParameterScanner]"); });
#endif

void ParameterScanner::muteLogger(){ Logger::setIsMuted(true); }
void ParameterScanner::unmuteLogger(){ Logger::setIsMuted(false); }

void ParameterScanner::readConfigImpl() {
  if( _config_.empty() ) return;
  LogWarning << "Configuring ParameterScanner..." << std::endl;

  _useParameterLimits_ = GenericToolbox::Json::fetchValue(_config_, "useParameterLimits", _useParameterLimits_);
  _nbPoints_ = GenericToolbox::Json::fetchValue(_config_, "nbPoints", _nbPoints_);
  _nbPointsLineScan_ = GenericToolbox::Json::fetchValue(_config_, "nbPointsLineScan", _nbPoints_);
  _parameterSigmaRange_ = GenericToolbox::Json::fetchValue(_config_, "parameterSigmaRange", _parameterSigmaRange_);

  _varsConfig_ = GenericToolbox::Json::fetchValue(_config_, "varsConfig", JsonType());

  LogWarning << "ParameterScanner configured." << std::endl;
}
void ParameterScanner::initializeImpl() {
  LogWarning << "Initializing ParameterScanner..." << std::endl;

  LogThrowIf(_likelihoodInterfacePtr_ == nullptr, "_likelihoodInterfacePtr_ not set.");

  _scanDataDict_.clear();
  if( GenericToolbox::Json::fetchValue(_varsConfig_, "llh", true) ){
    _scanDataDict_.emplace_back();
    auto& scanEntry = _scanDataDict_.back();
    scanEntry.yPoints = std::vector<double>(_nbPoints_+1,0);
    scanEntry.folder = "llh";
    scanEntry.title = "Total Likelihood Scan";
    scanEntry.yTitle = "LLH value";
    scanEntry.evalY = [this](){ return _likelihoodInterfacePtr_->getLastLikelihood(); };
  }
  if( GenericToolbox::Json::fetchValue(_varsConfig_, "llhPenalty", true) ){
    _scanDataDict_.emplace_back();
    auto& scanEntry = _scanDataDict_.back();
    scanEntry.yPoints = std::vector<double>(_nbPoints_+1,0);
    scanEntry.folder = "llhPenalty";
    scanEntry.yPoints = std::vector<double>(_nbPoints_+1,0);
    scanEntry.title = "Penalty Likelihood Scan";
    scanEntry.yTitle = "Penalty LLH value";
    scanEntry.evalY = [this](){ return _likelihoodInterfacePtr_->getLastPenaltyLikelihood(); };
  }
  if( GenericToolbox::Json::fetchValue(_varsConfig_, "llhStat", true) ){
    _scanDataDict_.emplace_back();
    auto& scanEntry = _scanDataDict_.back();
    scanEntry.yPoints = std::vector<double>(_nbPoints_+1,0);
    scanEntry.folder = "llhStat";
    scanEntry.title = "Stat Likelihood Scan";
    scanEntry.yTitle = "Stat LLH value";
    scanEntry.evalY = [this](){ return _likelihoodInterfacePtr_->getLastStatLikelihood(); };
  }
  if( GenericToolbox::Json::fetchValue(_varsConfig_, "llhStatPerSample", false) ){
    _scanDataDict_.reserve( _likelihoodInterfacePtr_->getDataSetManager().getPropagator().getSampleSet().getSampleList().size() );
    for( auto& sample : _likelihoodInterfacePtr_->getDataSetManager().getPropagator().getSampleSet().getSampleList() ){
      _scanDataDict_.emplace_back();
      auto& scanEntry = _scanDataDict_.back();
      scanEntry.yPoints = std::vector<double>(_nbPoints_+1,0);
      scanEntry.folder = "llhStat/" + sample.getName() + "/";
      scanEntry.title = Form("Stat Likelihood Scan of sample \"%s\"", sample.getName().c_str());
      scanEntry.yTitle = "Stat LLH value";
      auto* samplePtr = &sample;
      scanEntry.evalY = [this, samplePtr](){ return _likelihoodInterfacePtr_->evalStatLikelihood( *samplePtr ); };
    }
  }
  if( GenericToolbox::Json::fetchValue(_varsConfig_, "llhStatPerSamplePerBin", false) ){
    for( auto& sample : _likelihoodInterfacePtr_->getDataSetManager().getPropagator().getSampleSet().getSampleList() ){
      for( auto& bin : sample.getMcContainer().getHistogram().binList ){
        _scanDataDict_.emplace_back();
        auto& scanEntry = _scanDataDict_.back();
        scanEntry.yPoints = std::vector<double>(_nbPoints_+1,0);
        scanEntry.folder = "llhStat/" + sample.getName() + "/bin_" + std::to_string(bin.index);
        scanEntry.title = Form(R"(Stat LLH Scan of sample "%s", bin #%d "%s")",
                               sample.getName().c_str(), bin.index,
                               sample.getBinning().getBinList()[bin.index].getSummary().c_str());
        scanEntry.yTitle = "Stat LLH value";
        auto* samplePtr = &sample;
        int iBin = bin.index;
        scanEntry.evalY = [this, samplePtr, iBin](){ return _likelihoodInterfacePtr_->getJointProbabilityPtr()->eval( *samplePtr, iBin ); };
      }
    }
  }
  if( GenericToolbox::Json::fetchValue(_varsConfig_, "weightPerSample", false) ){
    for( auto& sample : _likelihoodInterfacePtr_->getDataSetManager().getPropagator().getSampleSet().getSampleList() ){
      _scanDataDict_.emplace_back();
      auto& scanEntry = _scanDataDict_.back();
      scanEntry.yPoints = std::vector<double>(_nbPoints_+1,0);
      scanEntry.folder = "weight/" + sample.getName();
      scanEntry.title = Form("MC event weight scan of sample \"%s\"", sample.getName().c_str());
      scanEntry.yTitle = "Total MC event weight";
      auto* samplePtr = &sample;
      scanEntry.evalY = [samplePtr](){ return samplePtr->getMcContainer().getSumWeights(); };
    }
  }
  if( GenericToolbox::Json::fetchValue(_varsConfig_, "weightPerSamplePerBin", false) ){
    for( auto& sample : _likelihoodInterfacePtr_->getDataSetManager().getPropagator().getSampleSet().getSampleList() ){
      for( auto& bin : sample.getMcContainer().getHistogram().binList ){
        _scanDataDict_.emplace_back();
        auto& scanEntry = _scanDataDict_.back();
        scanEntry.yPoints = std::vector<double>(_nbPoints_+1,0);
        scanEntry.folder = "weight/" + sample.getName() + "/bin_" + std::to_string(bin.index);
        scanEntry.title = Form(R"(MC event weight scan of sample "%s", bin #%d "%s")",
                               sample.getName().c_str(),
                               bin.index,
                               sample.getBinning().getBinList()[bin.index].getSummary().c_str());
        scanEntry.yTitle = "Total MC event weight";
        auto* samplePtr = &sample;
        scanEntry.evalY = [bin](){ return bin.content; };
      }
    }
  }

  LogWarning << "ParameterScanner initialized." << std::endl;
}

void ParameterScanner::scanParameterList( std::vector<Parameter>& par_, TDirectory* saveDir_){
  LogThrowIf(not isInitialized());
  LogThrowIf(saveDir_ == nullptr);
  for( auto& par : par_ ){ this->scanParameter(par, saveDir_); }
}
void ParameterScanner::scanParameter(Parameter& par_, TDirectory* saveDir_) {
  LogThrowIf(not isInitialized());
  LogThrowIf(saveDir_ == nullptr);
  std::vector<double> parPoints(_nbPoints_+1,0);

  LogInfo << "Scanning: " << par_.getFullTitle() << " / " << _nbPoints_ << " steps..." << std::endl;

  if( par_.getOwner()->isEnableEigenDecomp() and not par_.isEigen() ){
    // temporarily disable the automatic conversion Eigen -> Original
    _likelihoodInterfacePtr_->getDataSetManager().getPropagator().setEnableEigenToOrigInPropagate( false );
  }

  double origVal = par_.getParameterValue();
  double lowBound = origVal + _parameterSigmaRange_.first * par_.getStdDevValue();
  double highBound = origVal + _parameterSigmaRange_.second * par_.getStdDevValue();

  if( _useParameterLimits_ ){
    lowBound = std::max(lowBound, par_.getMinValue());
    highBound = std::min(highBound, par_.getMaxValue());
  }

  int offSet{0}; // offset help make sure the first point
  for( int iPt = 0 ; iPt < _nbPoints_+1 ; iPt++ ){
    double newVal = lowBound + double(iPt-offSet)/(_nbPoints_-1)*( highBound - lowBound );
    if( offSet == 0 and newVal > origVal ){
      newVal = origVal;
      offSet = 1;
    }

    LogThrowIf(
        std::isnan(newVal),
        "Scanning point is nan. Current values are: "
        << std::endl
        << GET_VAR_NAME_VALUE(iPt) << std::endl
        << GET_VAR_NAME_VALUE(lowBound) << std::endl
        << GET_VAR_NAME_VALUE(highBound) << std::endl
        << GET_VAR_NAME_VALUE(_nbPoints_) << std::endl
        << GET_VAR_NAME_VALUE(offSet) << std::endl
        << GET_VAR_NAME_VALUE(origVal) << std::endl
        << GET_VAR_NAME_VALUE(_parameterSigmaRange_.first) << std::endl
        << GET_VAR_NAME_VALUE(_parameterSigmaRange_.second) << std::endl
        << GET_VAR_NAME_VALUE(par_.getStdDevValue()) << std::endl
        );

    par_.setParameterValue(newVal);

    _likelihoodInterfacePtr_->propagateAndEvalLikelihood();
    parPoints[iPt] = par_.getParameterValue();

    for( auto& scanEntry : _scanDataDict_ ){
        double y = scanEntry.evalY();
        if (std::isnan(y)) y = -2.0;
        if (not std::isfinite(y)) y = -1.0;
        scanEntry.yPoints[iPt] = y;
    }
  }

  // sorting points in increasing order
  auto p = GenericToolbox::getSortPermutation(parPoints, [](double a_, double b_){
    if( a_ < b_ ) return true;
    return false;
  });
  GenericToolbox::applyPermutation(parPoints, p);
  for( auto& scanEntry : _scanDataDict_ ){
    GenericToolbox::applyPermutation(scanEntry.yPoints, p);
  }


  par_.setParameterValue(origVal);
  _likelihoodInterfacePtr_->propagateAndEvalLikelihood();

  // Disable the auto conversion from Eigen to Original if the fit is set to use eigen decomp
  if( par_.getOwner()->isEnableEigenDecomp() and not par_.isEigen() ){
    _likelihoodInterfacePtr_->getDataSetManager().getPropagator().setEnableEigenToOrigInPropagate( true );
  }

  std::stringstream ss;
  ss << GenericToolbox::replaceSubstringInString(par_.getFullTitle(), "/", "_");
  ss << "_TGraph";

  for( auto& scanEntry : _scanDataDict_ ){
    TGraph scanGraph(int(parPoints.size()), &parPoints[0], &scanEntry.yPoints[0]);
    scanGraph.SetTitle(scanEntry.title.c_str());
    scanGraph.GetYaxis()->SetTitle(scanEntry.yTitle.c_str());
    scanGraph.GetXaxis()->SetTitle(par_.getFullTitle().c_str());
    scanGraph.SetDrawOption("AP");
    scanGraph.SetMarkerStyle(kFullDotLarge);
    GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile( saveDir_, scanEntry.folder ), &scanGraph, ss.str());
  }

  std::stringstream ssVal;
  ssVal << GenericToolbox::replaceSubstringInString(par_.getFullTitle(), "/", "_");
  ssVal << "_CurrentPar";

  // current parameter value / center of the scan:
  TVectorD currentParValue(1);
  currentParValue[0] = par_.getParameterValue();
  GenericToolbox::writeInTFile(saveDir_, &currentParValue, ssVal.str());
}
void ParameterScanner::scanSegment(TDirectory *saveDir_, const JsonType &end_, const JsonType &start_, int nSteps_) {
  if( nSteps_ == -1 ){ nSteps_ = _nbPointsLineScan_; }
  LogWarning << "Scanning along a segment with " << nSteps_ << " steps." << std::endl;

  // don't shout while re-injecting parameters
  GenericToolbox::ScopedGuard s(
      []{ ParameterSet::muteLogger(); ParametersManager::muteLogger(); },
      []{ ParameterSet::unmuteLogger(); ParametersManager::unmuteLogger(); }
  );

  LogThrowIf(end_.empty(), "Ending injector config is empty()");
  LogThrowIf(nSteps_ < 0, "Invalid nSteps");

  // nSteps_+2 as we also want the first and last points
  int nTotalSteps = nSteps_+2;

  LogInfo << "Backup current position of the propagator..." << std::endl;
  auto currentParState = _likelihoodInterfacePtr_->getDataSetManager().getPropagator().getParametersManager().exportParameterInjectorConfig();

  LogInfo << "Reading start point parameter state..." << std::endl;
  std::vector<std::pair<Parameter*, double>> startPointParValList;
  if( not start_.empty() ){ _likelihoodInterfacePtr_->getDataSetManager().getPropagator().getParametersManager().injectParameterValues(start_); }
  else{ _likelihoodInterfacePtr_->getDataSetManager().getPropagator().getParametersManager().injectParameterValues(currentParState); }
  for( auto& parSet : _likelihoodInterfacePtr_->getDataSetManager().getPropagator().getParametersManager().getParameterSetsList() ){
    if( not parSet.isEnabled() ){ continue; }
    for( auto& par : parSet.getParameterList() ){
      if( not par.isEnabled() ){ continue; }
      startPointParValList.emplace_back(&par, par.getParameterValue());
    }
  }

  LogInfo << "Reading end point parameter state..." << std::endl;
  std::vector<std::pair<Parameter*, double>> endPointParValList;
  _likelihoodInterfacePtr_->getDataSetManager().getPropagator().getParametersManager().injectParameterValues(end_);
  endPointParValList.reserve(startPointParValList.size());
  for( auto& parPair : startPointParValList ){
    endPointParValList.emplace_back(parPair.first, parPair.first->getParameterValue());
  }

  LogInfo << "Creating graph holders..." << std::endl;

  _graphEntriesBuf_.clear();
  for( auto& scanData : _scanDataDict_ ){
    for( auto& parPair : startPointParValList ){
      _graphEntriesBuf_.emplace_back(GraphEntry{&scanData, parPair.first, TGraph(nTotalSteps)});
    }
  }

  std::stringstream ss;
  ss << LogWarning.getPrefixString() << "Scanning...";

  LogInfo << "Scanning along the line..." << std::endl;
  for( int iStep = 0 ; iStep < nTotalSteps ; iStep++ ){
    if( not Logger::isMuted() ){ GenericToolbox::displayProgressBar(iStep, nTotalSteps-1, ss.str()); }

    for( size_t iPar = 0 ; iPar < startPointParValList.size() ; iPar++ ){
      auto* par = startPointParValList[iPar].first;
      par->setParameterValue(
          startPointParValList[iPar].second
          + ( endPointParValList[iPar].second - startPointParValList[iPar].second ) * double(iStep) / double(nTotalSteps-1)
          );
    }

    for( auto& parSet : _likelihoodInterfacePtr_->getDataSetManager().getPropagator().getParametersManager().getParameterSetsList() ){
      if( not parSet.isEnabled() ){ continue; }
      if( parSet.isEnableEigenDecomp() ){
        // make sure the parameters don't get overwritten
        parSet.propagateOriginalToEigen();
      }
    }

    _likelihoodInterfacePtr_->propagateAndEvalLikelihood();

    for( auto& graphEntry : _graphEntriesBuf_ ){
      graphEntry.graph.SetPointX(iStep, graphEntry.fitParPtr->getParameterValue());
      graphEntry.graph.SetPointY(iStep, graphEntry.scanDataPtr->evalY());
    }

  }

  LogInfo << "Writing scan line graph in file..." << std::endl;
  for( auto& graphEntry : _graphEntriesBuf_ ){
    ParameterScanner::writeGraphEntry( graphEntry, saveDir_ );
  }

  LogInfo << "Restore position of the propagator..." << std::endl;
  _likelihoodInterfacePtr_->getDataSetManager().getPropagator().getParametersManager().injectParameterValues( currentParState );
  _likelihoodInterfacePtr_->propagateAndEvalLikelihood();
}
void ParameterScanner::generateOneSigmaPlots(TDirectory* saveDir_){
  LogThrowIf(not isInitialized());
  LogThrowIf(saveDir_ == nullptr, "saveDir_ not set.");

  // Build the histograms with the current parameters
  _likelihoodInterfacePtr_->getDataSetManager().getPropagator().propagateParameters();
  _likelihoodInterfacePtr_->getDataSetManager().getPropagator().getPlotGenerator().generateSamplePlots();
  auto refHistList = _likelihoodInterfacePtr_->getDataSetManager().getPropagator().getPlotGenerator().getHistHolderList();

  auto makeOneSigmaPlotFct = [&](Parameter& par_, TDirectory* parSavePath_){
    LogInfo << "Generating one sigma plots for \"" << par_.getFullTitle() << "\" -> " << par_.getParameterValue() << " + " << par_.getStdDevValue() << std::endl;
    double currentParValue = par_.getParameterValue();

    // Push the selected parameter to 1 sigma
    par_.setParameterValue( currentParValue + par_.getStdDevValue() );

    // Propagate the parameters
    _likelihoodInterfacePtr_->getDataSetManager().getPropagator().propagateParameters();

    // put the saved histograms in slot 1 of the buffer
    _likelihoodInterfacePtr_->getDataSetManager().getPropagator().getPlotGenerator().generateSampleHistograms(nullptr, 1);

    // Compare with the
    _likelihoodInterfacePtr_->getDataSetManager().getPropagator().getPlotGenerator().generateComparisonPlots(
        _likelihoodInterfacePtr_->getDataSetManager().getPropagator().getPlotGenerator().getHistHolderList(1),
        refHistList, parSavePath_
    );

    // Come back to the original place
    par_.setParameterValue( currentParValue );
    _likelihoodInterfacePtr_->getDataSetManager().getPropagator().propagateParameters();
  };

  // +1 sigma
  for( auto& parSet : _likelihoodInterfacePtr_->getDataSetManager().getPropagator().getParametersManager().getParameterSetsList() ){

    if( not parSet.isEnabled() ) continue;

    if( GenericToolbox::Json::fetchValue(parSet.getConfig(), "disableOneSigmaPlots", false) ){
      LogInfo << "+1σ plots disabled for \"" << parSet.getName() << "\"" << std::endl;
      continue;
    }

    for( auto& effPar : parSet.getEffectiveParameterList() ){
      if( not effPar.isEnabled() ) continue;
      std::string tag;
      if( effPar.isFixed() ){ tag += "_FIXED"; }
      makeOneSigmaPlotFct(
          effPar,
          GenericToolbox::mkdirTFile(saveDir_, parSet.getName() + "/" + effPar.getTitle() + tag)
      );
    }

  }

  // Since those were not saved, delete manually
//  for( auto& refHist : refHistList ){ delete refHist.histPtr; }
  refHistList.clear();

  GenericToolbox::triggerTFileWrite(saveDir_);

}
void ParameterScanner::varyEvenRates(const std::vector<double>& paramVariationList_, TDirectory* saveDir_){
  LogThrowIf(not isInitialized());
  saveDir_->cd();

  LogInfo << __METHOD_NAME__ << std::endl;
  LogScopeIndent;

  // make sure the parameters are rolled back to their original value
  std::map<Parameter*, double> parStateList{};
  GenericToolbox::ScopedGuard g(
      [&]{
        LogWarning << "Temporarily pulling back parameters at their prior before performing the event rate..." << std::endl;
        for( auto& parSet : _likelihoodInterfacePtr_->getDataSetManager().getPropagator().getParametersManager().getParameterSetsList() ){
          if( not parSet.isEnabled() ) { continue; }
          for( auto& par : parSet.getParameterList() ){
            if( not par.isEnabled() ) { continue; }
            parStateList[&par] = par.getParameterValue();
            par.setParameterValue( par.getPriorValue() );
          }
        }
        _likelihoodInterfacePtr_->getDataSetManager().getPropagator().propagateParameters();
      },
      [&]{
        LogWarning << "Restoring parameters to their original values..." << std::endl;
        for( auto& parSet : _likelihoodInterfacePtr_->getDataSetManager().getPropagator().getParametersManager().getParameterSetsList() ){
          if( not parSet.isEnabled() ) { continue; }
          for( auto& par : parSet.getParameterList() ){
            if( not par.isEnabled() ){ continue; }
            par.setParameterValue( parStateList[&par] );
          }
        }
        _likelihoodInterfacePtr_->getDataSetManager().getPropagator().propagateParameters();
      }
  );

  auto makeVariedEventRatesFct = [&](Parameter& par_, std::vector<double> variationList_, TDirectory* saveSubDir_){
    LogInfo << "Making varied event rates for " << par_.getFullTitle() << std::endl;

    // First make sure all params are at their prior <- is it necessary?
    for( auto& parSet : _likelihoodInterfacePtr_->getDataSetManager().getPropagator().getParametersManager().getParameterSetsList() ){
      if( not parSet.isEnabled() ) continue;
      for( auto& par : parSet.getParameterList() ){
        if( not par.isEnabled() ) continue;
        par.setParameterValue( par.getPriorValue() );
      }
    }
    _likelihoodInterfacePtr_->getDataSetManager().getPropagator().propagateParameters();

    saveSubDir_->cd();

    std::vector<std::vector<double>> buffEvtRatesMap; //[iVar][iSample]
    /*std::vector<double> variationList;
    if (par_.isFree()){
      // Preliminary implementation
      if(par_.getMinValue() == par_.getMinValue()) variationList.push_back(par_.getMinValue());
      variationList.push_back(par_.getPriorValue());
      if(par_.getMaxValue() == par_.getMaxValue()) variationList.push_back(par_.getMaxValue());
    }
    else{
      variationList = variationList_;
    }*/

    for ( size_t iVar = 0 ; iVar < variationList_.size() ; iVar++ ){

      buffEvtRatesMap.emplace_back();

      double cappedParValue{par_.getPriorValue() + variationList_[iVar] * par_.getStdDevValue()};
      cappedParValue = std::min(cappedParValue, par_.getMaxValue());
      cappedParValue = std::max(cappedParValue, par_.getMinValue());

      par_.setParameterValue( cappedParValue );
      _likelihoodInterfacePtr_->getDataSetManager().getPropagator().propagateParameters();

      for(auto & sample : _likelihoodInterfacePtr_->getDataSetManager().getPropagator().getSampleSet().getSampleList()){
        buffEvtRatesMap[iVar].emplace_back( sample.getMcContainer().getSumWeights() );
      }

      // back to the prior
      par_.setParameterValue( par_.getPriorValue() );
      _likelihoodInterfacePtr_->getDataSetManager().getPropagator().propagateParameters();
    }


    // Write in the output

    auto* variationList_TVectorD = new TVectorD(int(variationList_.size()));

    for ( int iVar = 0 ; iVar < variationList_TVectorD->GetNrows() ; iVar++ ){
      /*if (par_.isFree()) (*variationList_TVectorD)(iVar) = variationList_[iVar];
      else
      */
      (*variationList_TVectorD)(iVar) = par_.getPriorValue() + variationList_[iVar] * par_.getStdDevValue();
    }
    GenericToolbox::writeInTFile(saveSubDir_, variationList_TVectorD, "paramValues");

    TVectorD* buffVariedEvtRates_TVectorD{nullptr};

    for( size_t iSample = 0 ; iSample < _likelihoodInterfacePtr_->getDataSetManager().getPropagator().getSampleSet().getSampleList().size() ; iSample++ ){

      buffVariedEvtRates_TVectorD = new TVectorD(int(variationList_.size()));

      for( int iVar = 0 ; iVar < buffVariedEvtRates_TVectorD->GetNrows() ; iVar++ ){
        (*buffVariedEvtRates_TVectorD)(iVar) = buffEvtRatesMap[iVar][iSample];
      }

      GenericToolbox::writeInTFile(
          saveSubDir_, buffVariedEvtRates_TVectorD,
          _likelihoodInterfacePtr_->getDataSetManager().getPropagator().getSampleSet().getSampleList()[iSample].getName()
      );

    }


  };

  // vary parameters

  for( auto& parSet : _likelihoodInterfacePtr_->getDataSetManager().getPropagator().getParametersManager().getParameterSetsList() ){

    if( not parSet.isEnabled() ) continue;
    if( GenericToolbox::Json::fetchValue(parSet.getConfig(), "skipVariedEventRates", false) ){
      LogInfo << "Event rate variation skipped for \"" << parSet.getName() << "\"" << std::endl;
      continue;
    }

    if( parSet.isEnableEigenDecomp() ){
      // not implemented
      continue;
    }
    else{
      for( auto& par : parSet.getParameterList() ){

        if( not par.isEnabled() ) continue;

        std::string tag;
        if( par.isFixed() ){ tag += "_FIXED"; }
        if( par.isFree() ){ tag += "_FREE"; }

        TDirectory* subDir = GenericToolbox::mkdirTFile(saveDir_, parSet.getName() + "/" + par.getTitle() + tag);
        makeVariedEventRatesFct(par, paramVariationList_, subDir);

      }
    }

  }
}

// statics
void ParameterScanner::writeGraphEntry(GraphEntry& entry_, TDirectory* saveDir_){
  entry_.graph.SetTitle(entry_.scanDataPtr->title.c_str());
  entry_.graph.GetYaxis()->SetTitle(entry_.scanDataPtr->yTitle.c_str());
  entry_.graph.GetXaxis()->SetTitle(entry_.fitParPtr->getFullTitle().c_str());
  entry_.graph.SetDrawOption("AP");

  // marker indicates the direction
  if( entry_.graph.GetY()[0] == entry_.graph.GetY()[entry_.graph.GetN()-1] ){
    // Did not move
    entry_.graph.SetMarkerStyle( kFullDotMedium );
  }
  else if( entry_.graph.GetY()[0] > entry_.graph.GetY()[entry_.graph.GetN()-1] ){
    // Did go down
    entry_.graph.SetMarkerStyle( kFullTriangleDown );
  }
  else{
    // Did go up
    entry_.graph.SetMarkerStyle( kFullTriangleUp );
  }

  GenericToolbox::writeInTFile(
      GenericToolbox::mkdirTFile( saveDir_, entry_.scanDataPtr->folder ),
      entry_.graph,
      GenericToolbox::generateCleanBranchName(entry_.fitParPtr->getFullTitle())
  );
}
