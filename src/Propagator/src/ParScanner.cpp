//
// Created by Adrien BLANCHET on 07/04/2022.
//

#include "ParScanner.h"
#include "JsonUtils.h"
#include "Propagator.h"
#include "FitParameter.h"

#include "Logger.h"

#include "TGraph.h"
#include <TDirectory.h>

#include <utility>


LoggerInit([]{
  Logger::setUserHeaderStr("[ParScanner]");
});


ParScanner::ParScanner(Propagator* owner_): _owner_(owner_) {}

void ParScanner::readConfigImpl() {
  if( _config_.empty() ) return;

  _useParameterLimits_ = JsonUtils::fetchValue(_config_, "useParameterLimits", _useParameterLimits_);
  _nbPoints_ = JsonUtils::fetchValue(_config_, "nbPoints", _nbPoints_);
  _parameterSigmaRange_ = JsonUtils::fetchValue(_config_, "parameterSigmaRange", _parameterSigmaRange_);

  _varsConfig_ = JsonUtils::fetchValue(_config_, "varsConfig", nlohmann::json());
}
void ParScanner::initializeImpl() {
  LogInfo << "Initializing ParScanner..." << std::endl;
  LogThrowIf(_owner_== nullptr, "_owner_ is not set");
}

void ParScanner::setOwner(Propagator *owner){
  _owner_ = owner;
}
void ParScanner::setNbPoints(int nbPoints) {
  _nbPoints_ = nbPoints;
}

int ParScanner::getNbPoints() const {
  return _nbPoints_;
}
const std::pair<double, double> &ParScanner::getParameterSigmaRange() const {
  return _parameterSigmaRange_;
}
bool ParScanner::isUseParameterLimits() const {
  return _useParameterLimits_;
}

void ParScanner::scanFitParameters(std::vector<FitParameter>& parList_, TDirectory* saveDir_){
  LogThrowIf(not isInitialized());
  LogThrowIf(saveDir_ == nullptr);
  for( auto& par : parList_ ){ this->scanFitParameter(par, saveDir_); }
}
void ParScanner::scanFitParameter(FitParameter& par_, TDirectory* saveDir_) {
  LogThrowIf(not isInitialized());
  LogThrowIf(saveDir_ == nullptr);
  std::vector<double> parPoints(_nbPoints_+1,0);

  std::stringstream ssPbar;
  ssPbar << LogInfo.getPrefixString() << "Scanning: " << par_.getFullTitle() << " / " << _nbPoints_ << " steps...";
  GenericToolbox::displayProgressBar(0, _nbPoints_, ssPbar.str());

  scanDataDict.clear();
  if( JsonUtils::fetchValue(_varsConfig_, "llh", true) ){
    scanDataDict.emplace_back();
    auto& scanEntry = scanDataDict.back();
    scanEntry.yPoints = std::vector<double>(_nbPoints_+1,0);
    scanEntry.folder = "llh";
    scanEntry.title = "Total Likelihood Scan";
    scanEntry.yTitle = "LLH value";
    scanEntry.evalY = [this](){ return _owner_->getLlhBuffer(); };
  }
  if( JsonUtils::fetchValue(_varsConfig_, "llhPenalty", true) ){
    scanDataDict.emplace_back();
    auto& scanEntry = scanDataDict.back();
    scanEntry.yPoints = std::vector<double>(_nbPoints_+1,0);
    scanEntry.folder = "llhPenalty";
    scanEntry.yPoints = std::vector<double>(_nbPoints_+1,0);
    scanEntry.title = "Penalty Likelihood Scan";
    scanEntry.yTitle = "Penalty LLH value";
    scanEntry.evalY = [this](){ return _owner_->getLlhPenaltyBuffer(); };
  }
  if( JsonUtils::fetchValue(_varsConfig_, "llhStat", true) ){
    scanDataDict.emplace_back();
    auto& scanEntry = scanDataDict.back();
    scanEntry.yPoints = std::vector<double>(_nbPoints_+1,0);
    scanEntry.folder = "llhStat";
    scanEntry.title = "Stat Likelihood Scan";
    scanEntry.yTitle = "Stat LLH value";
    scanEntry.evalY = [this](){ return _owner_->getLlhStatBuffer(); };
  }
  if( JsonUtils::fetchValue(_varsConfig_, "llhStatPerSample", false) ){
    for( auto& sample : _owner_->getFitSampleSet().getFitSampleList() ){
      scanDataDict.emplace_back();
      auto& scanEntry = scanDataDict.back();
      scanEntry.yPoints = std::vector<double>(_nbPoints_+1,0);
      scanEntry.folder = "llhStat/" + sample.getName() + "/";
      scanEntry.title = Form("Stat Likelihood Scan of sample \"%s\"", sample.getName().c_str());
      scanEntry.yTitle = "Stat LLH value";
      auto* samplePtr = &sample;
      scanEntry.evalY = [this, samplePtr](){ return _owner_->getFitSampleSet().evalLikelihood(*samplePtr); };
    }
  }
  if( JsonUtils::fetchValue(_varsConfig_, "llhStatPerSamplePerBin", false) ){
    for( auto& sample : _owner_->getFitSampleSet().getFitSampleList() ){
      for( int iBin = 1 ; iBin <= sample.getMcContainer().histogram->GetNbinsX() ; iBin++ ){
        scanDataDict.emplace_back();
        auto& scanEntry = scanDataDict.back();
        scanEntry.yPoints = std::vector<double>(_nbPoints_+1,0);
        scanEntry.folder = "llhStat/" + sample.getName() + "/bin_" + std::to_string(iBin);
        scanEntry.title = Form(R"(Stat LLH Scan of sample "%s", bin #%d "%s")",
                               sample.getName().c_str(),
                               iBin,
                               sample.getBinning().getBinsList()[iBin-1].getSummary().c_str());
        scanEntry.yTitle = "Stat LLH value";
        auto* samplePtr = &sample;
        scanEntry.evalY = [this, samplePtr, iBin](){ return _owner_->getFitSampleSet().getJointProbabilityFct()->eval(*samplePtr, iBin); };
      }
    }
  }
  if( JsonUtils::fetchValue(_varsConfig_, "weightPerSample", false) ){
    for( auto& sample : _owner_->getFitSampleSet().getFitSampleList() ){
      scanDataDict.emplace_back();
      auto& scanEntry = scanDataDict.back();
      scanEntry.yPoints = std::vector<double>(_nbPoints_+1,0);
      scanEntry.folder = "weight/" + sample.getName();
      scanEntry.title = Form("MC event weight scan of sample \"%s\"", sample.getName().c_str());
      scanEntry.yTitle = "Total MC event weight";
      auto* samplePtr = &sample;
      scanEntry.evalY = [samplePtr](){ return samplePtr->getMcContainer().getSumWeights(); };
    }
  }
  if( JsonUtils::fetchValue(_varsConfig_, "weightPerSamplePerBin", false) ){
    for( auto& sample : _owner_->getFitSampleSet().getFitSampleList() ){
      for( int iBin = 1 ; iBin <= sample.getMcContainer().histogram->GetNbinsX() ; iBin++ ){
        scanDataDict.emplace_back();
        auto& scanEntry = scanDataDict.back();
        scanEntry.yPoints = std::vector<double>(_nbPoints_+1,0);
        scanEntry.folder = "weight/" + sample.getName() + "/bin_" + std::to_string(iBin);
        scanEntry.title = Form(R"(MC event weight scan of sample "%s", bin #%d "%s")",
                               sample.getName().c_str(),
                               iBin,
                               sample.getBinning().getBinsList()[iBin-1].getSummary().c_str());
        scanEntry.yTitle = "Total MC event weight";
        auto* samplePtr = &sample;
        scanEntry.evalY = [samplePtr, iBin](){ return samplePtr->getMcContainer().histogram->GetBinContent(iBin); };
      }
    }
  }

  double origVal = par_.getParameterValue();
  double lowBound = origVal + _parameterSigmaRange_.first * par_.getStdDevValue();
  double highBound = origVal + _parameterSigmaRange_.second * par_.getStdDevValue();

  if( _useParameterLimits_ ){
    lowBound = std::max(lowBound, par_.getMinValue());
    highBound = std::min(highBound, par_.getMaxValue());
  }

  int offSet{0};
  for( int iPt = 0 ; iPt < _nbPoints_+1 ; iPt++ ){
    GenericToolbox::displayProgressBar(iPt, _nbPoints_, ssPbar.str());

    double newVal = lowBound + double(iPt-offSet)/(_nbPoints_-1)*( highBound - lowBound );
    if( offSet == 0 and newVal > origVal ){
      newVal = origVal;
      offSet = 1;
    }

    par_.setParameterValue(newVal);
    _owner_->updateLlhCache();
    parPoints[iPt] = par_.getParameterValue();

    for( auto& scanEntry : scanDataDict ){ scanEntry.yPoints[iPt] = scanEntry.evalY(); }
  }

  par_.setParameterValue(origVal);
  _owner_->updateLlhCache();

  std::stringstream ss;
  ss << GenericToolbox::replaceSubstringInString(par_.getFullTitle(), "/", "_");
  ss << "_TGraph";

  for( auto& scanEntry : scanDataDict ){
    TGraph scanGraph(int(parPoints.size()), &parPoints[0], &scanEntry.yPoints[0]);
    scanGraph.SetTitle(scanEntry.title.c_str());
    scanGraph.GetYaxis()->SetTitle(scanEntry.yTitle.c_str());
    scanGraph.GetXaxis()->SetTitle(par_.getFullTitle().c_str());
    scanGraph.SetDrawOption("AP");
    scanGraph.SetMarkerStyle(kFullDotLarge);
    GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile( saveDir_, scanEntry.folder ), &scanGraph, ss.str());
  }
}
void ParScanner::generateOneSigmaPlots(TDirectory* saveDir_){
  LogThrowIf(not isInitialized());
  LogThrowIf(saveDir_ == nullptr, "saveDir_ not set.");

  // Build the histograms with the current parameters
  _owner_->propagateParametersOnSamples();
  _owner_->getPlotGenerator().generateSamplePlots();
  auto refHistList = _owner_->getPlotGenerator().getHistHolderList();

  auto makeOneSigmaPlotFct = [&](FitParameter& par_, TDirectory* parSavePath_){
    LogInfo << "Generating one sigma plots for \"" << par_.getFullTitle() << "\" -> " << par_.getParameterValue() << " + " << par_.getStdDevValue() << std::endl;
    double currentParValue = par_.getParameterValue();

    // Push the selected parameter to 1 sigma
    par_.setParameterValue( currentParValue + par_.getStdDevValue() );

    // Propagate the parameters
    _owner_->propagateParametersOnSamples();

    // put the saved histograms in slot 1 of the buffer
    _owner_->getPlotGenerator().generateSampleHistograms(nullptr, 1);

    // Compare with the
    _owner_->getPlotGenerator().generateComparisonPlots(
        _owner_->getPlotGenerator().getHistHolderList(1),
        refHistList, parSavePath_
    );

    // Come back to the original place
    par_.setParameterValue( currentParValue );
    _owner_->propagateParametersOnSamples();
  };

  // +1 sigma
  for( auto& parSet : _owner_->getParameterSetsList() ){

    if( not parSet.isEnabled() ) continue;

    if( JsonUtils::fetchValue(parSet.getConfig(), "disableOneSigmaPlots", false) ){
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
void ParScanner::varyEvenRates(const std::vector<double>& paramVariationList_, TDirectory* saveDir_){
  LogThrowIf(not isInitialized());
  saveDir_->cd();

  auto makeVariedEventRatesFct = [&](FitParameter& par_, std::vector<double> variationList_, TDirectory* saveSubDir_){

    LogInfo << "Making varied event rates for " << par_.getFullTitle() << std::endl;

    // First make sure all params are at their prior <- is it necessary?
    for( auto& parSet : _owner_->getParameterSetsList() ){
      if( not parSet.isEnabled() ) continue;
      for( auto& par : parSet.getParameterList() ){
        par.setParameterValue(par.getPriorValue());
      }
    }
    _owner_->propagateParametersOnSamples();

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

      if(par_.getPriorValue() + variationList_[iVar] * par_.getStdDevValue() > par_.getMaxValue())
        par_.setParameterValue(par_.getMaxValue());
      else if (par_.getPriorValue() + variationList_[iVar] * par_.getStdDevValue() < par_.getMinValue())
        par_.setParameterValue(par_.getMinValue());
      else
        par_.setParameterValue(par_.getPriorValue() + variationList_[iVar] * par_.getStdDevValue());

      _owner_->propagateParametersOnSamples();

      for(auto & sample : _owner_->getFitSampleSet().getFitSampleList()){
        buffEvtRatesMap[iVar].push_back(sample.getMcContainer().getSumWeights() );
      }
      par_.setParameterValue(par_.getPriorValue());
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

    for( size_t iSample = 0 ; iSample < _owner_->getFitSampleSet().getFitSampleList().size() ; iSample++ ){

      buffVariedEvtRates_TVectorD = new TVectorD(int(variationList_.size()));

      for ( int iVar = 0 ; iVar < buffVariedEvtRates_TVectorD->GetNrows() ; iVar++ ){
        (*buffVariedEvtRates_TVectorD)(iVar) = buffEvtRatesMap[iVar][iSample];
      }

      GenericToolbox::writeInTFile(saveSubDir_, buffVariedEvtRates_TVectorD,
                                   _owner_->getFitSampleSet().getFitSampleList()[iSample].getName());

    }


  };

  // vary parameters

  for( auto& parSet : _owner_->getParameterSetsList() ){

    if( not parSet.isEnabled() ) continue;
    if( JsonUtils::fetchValue(parSet.getConfig(), "skipVariedEventRates", false) ){
      LogInfo << "Event rate variation skipped for \"" << parSet.getName() << "\"" << std::endl;
      continue;
    }

    if( parSet.isUseEigenDecompInFit() ){
      // TODO ?
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



