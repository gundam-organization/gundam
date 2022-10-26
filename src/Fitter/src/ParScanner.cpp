//
// Created by Adrien BLANCHET on 07/04/2022.
//

#include "ParScanner.h"
#include "JsonUtils.h"
#include "FitterEngine.h"
#include "FitParameter.h"

#include "Logger.h"

#include "TGraph.h"
#include <TDirectory.h>

#include <utility>


LoggerInit([]{
  Logger::setUserHeaderStr("[ParScanner]");
});


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


void ParScanner::setOwner(FitterEngine *owner){
  _owner_ = owner;
}
void ParScanner::setSaveDir(TDirectory *saveDir) {
  _saveDir_ = saveDir;
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

void ParScanner::scanMinimizerParameters(const std::string& saveSubdir_){
  LogThrowIf(not isInitialized());
  LogInfo << "Performing scans of fit parameters..." << std::endl;
  for( int iPar = 0 ; iPar < _owner_->getMinimizer().getMinimizer()->NDim() ; iPar++ ){
    if( _owner_->getMinimizer().getMinimizer()->IsFixedVariable(iPar) ){
      LogWarning << _owner_->getMinimizer().getMinimizer()->VariableName(iPar)
      << " is fixed. Skipping..." << std::endl;
      continue;
    }
    this->scanFitParameter(*_owner_->getMinimizer().getMinimizerFitParameterPtr()[iPar], saveSubdir_);
  } // iPar
}
void ParScanner::scanFitParameters(std::vector<FitParameter>& parList_, const std::string& saveSubdir_){
  LogThrowIf(not isInitialized());
  for( auto& par : parList_ ){ this->scanFitParameter(par, saveSubdir_); }
}
void ParScanner::scanFitParameter(FitParameter& par_, const std::string &saveSubdir_) {
  LogThrowIf(not isInitialized());
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
    scanEntry.evalY = [this](){ return _owner_->getChi2Buffer(); };
  }
  if( JsonUtils::fetchValue(_varsConfig_, "llhPenalty", true) ){
    scanDataDict.emplace_back();
    auto& scanEntry = scanDataDict.back();
    scanEntry.yPoints = std::vector<double>(_nbPoints_+1,0);
    scanEntry.folder = "llhPenalty";
    scanEntry.yPoints = std::vector<double>(_nbPoints_+1,0);
    scanEntry.title = "Penalty Likelihood Scan";
    scanEntry.yTitle = "Penalty LLH value";
    scanEntry.evalY = [this](){ return _owner_->getChi2PullsBuffer(); };
  }
  if( JsonUtils::fetchValue(_varsConfig_, "llhStat", true) ){
    scanDataDict.emplace_back();
    auto& scanEntry = scanDataDict.back();
    scanEntry.yPoints = std::vector<double>(_nbPoints_+1,0);
    scanEntry.folder = "llhStat";
    scanEntry.title = "Stat Likelihood Scan";
    scanEntry.yTitle = "Stat LLH value";
    scanEntry.evalY = [this](){ return _owner_->getChi2StatBuffer(); };
  }
  if( JsonUtils::fetchValue(_varsConfig_, "llhStatPerSample", false) ){
    for( auto& sample : _owner_->getPropagator().getFitSampleSet().getFitSampleList() ){
      scanDataDict.emplace_back();
      auto& scanEntry = scanDataDict.back();
      scanEntry.yPoints = std::vector<double>(_nbPoints_+1,0);
      scanEntry.folder = "llhStat/" + sample.getName() + "/";
      scanEntry.title = Form("Stat Likelihood Scan of sample \"%s\"", sample.getName().c_str());
      scanEntry.yTitle = "Stat LLH value";
      auto* samplePtr = &sample;
      scanEntry.evalY = [this, samplePtr](){ return _owner_->getPropagator().getFitSampleSet().evalLikelihood(*samplePtr); };
    }
  }
  if( JsonUtils::fetchValue(_varsConfig_, "llhStatPerSamplePerBin", false) ){
    for( auto& sample : _owner_->getPropagator().getFitSampleSet().getFitSampleList() ){
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
        scanEntry.evalY = [this, samplePtr, iBin](){ return _owner_->getPropagator().getFitSampleSet().getJointProbabilityFct()->eval(*samplePtr, iBin); };
      }
    }
  }
  if( JsonUtils::fetchValue(_varsConfig_, "weightPerSample", false) ){
    for( auto& sample : _owner_->getPropagator().getFitSampleSet().getFitSampleList() ){
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
    for( auto& sample : _owner_->getPropagator().getFitSampleSet().getFitSampleList() ){
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
    _owner_->updateChi2Cache();
    parPoints[iPt] = par_.getParameterValue();

    for( auto& scanEntry : scanDataDict ){ scanEntry.yPoints[iPt] = scanEntry.evalY(); }
  }

  par_.setParameterValue(origVal);
  _owner_->updateChi2Cache();

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
    if( _saveDir_ != nullptr ){
      GenericToolbox::mkdirTFile(_saveDir_, saveSubdir_ + "/" + scanEntry.folder )->cd();
      scanGraph.Write( ss.str().c_str() );
    }
  }
}
void ParScanner::generateOneSigmaPlots(const std::string& savePath_){

  _owner_->getPropagator().propagateParametersOnSamples();
  _owner_->getPropagator().getPlotGenerator().generateSamplePlots();

  GenericToolbox::mkdirTFile(_saveDir_, savePath_)->cd();
  auto refHistList = _owner_->getPropagator().getPlotGenerator().getHistHolderList(); // current buffer


  auto makeOneSigmaPlotFct = [&](FitParameter& par_, const std::string& parSavePath_){
    double currentParValue = par_.getParameterValue();
    par_.setParameterValue( currentParValue + par_.getStdDevValue() );
    LogInfo << "Processing " << parSavePath_ << " -> " << par_.getParameterValue() << std::endl;

    _owner_->getPropagator().propagateParametersOnSamples();

    auto* saveDir = GenericToolbox::mkdirTFile(_saveDir_, parSavePath_ );
    saveDir->cd();

    _owner_->getPropagator().getPlotGenerator().generateSampleHistograms(nullptr, 1);

    auto oneSigmaHistList = _owner_->getPropagator().getPlotGenerator().getHistHolderList(1);
    _owner_->getPropagator().getPlotGenerator().generateComparisonPlots( oneSigmaHistList, refHistList, saveDir );
    par_.setParameterValue( currentParValue );
    _owner_->getPropagator().propagateParametersOnSamples();

    const auto& compHistList = _owner_->getPropagator().getPlotGenerator().getComparisonHistHolderList();

//      // Since those were not saved, delete manually
//      // Don't delete? -> slower each time
////      for( auto& hist : oneSigmaHistList ){ delete hist.histPtr; }
//      oneSigmaHistList.clear();
  };

  // +1 sigma
  for( auto& parSet : _owner_->getPropagator().getParameterSetsList() ){

    if( not parSet.isEnabled() ) continue;

    if( JsonUtils::fetchValue(parSet.getConfig(), "disableOneSigmaPlots", false) ){
      LogInfo << "+1σ plots disabled for \"" << parSet.getName() << "\"" << std::endl;
      continue;
    }

    if( parSet.isUseEigenDecompInFit() ){
      for( auto& eigenPar : parSet.getEigenParameterList() ){
        if( not eigenPar.isEnabled() ) continue;
        std::string tag;
        if( eigenPar.isFixed() ){ tag += "_FIXED"; }
        std::string savePath = savePath_;
        if( not savePath.empty() ) savePath += "/";
        savePath += "oneSigma/eigen/" + parSet.getName() + "/" + eigenPar.getTitle() + tag;
        makeOneSigmaPlotFct(eigenPar, savePath);
      }
    }
    else{
      for( auto& par : parSet.getParameterList() ){
        if( not par.isEnabled() ) continue;
        std::string tag;
        if( par.isFixed() ){ tag += "_FIXED"; }
        std::string savePath = savePath_;
        if( not savePath.empty() ) savePath += "/";
        savePath += "oneSigma/original/" + parSet.getName() + "/" + par.getTitle() + tag;
        makeOneSigmaPlotFct(par, savePath);
      }
    }

  }

  _saveDir_->cd();

  // Since those were not saved, delete manually
//  for( auto& refHist : refHistList ){ delete refHist.histPtr; }
  refHistList.clear();

}
void ParScanner::varyEvenRates(const std::vector<double>& paramVariationList_, const std::string& savePath_){
  GenericToolbox::mkdirTFile(_saveDir_, savePath_)->cd();

  auto makeVariedEventRatesFct = [&](FitParameter& par_, std::vector<double> variationList_, const std::string& parSavePath_){

    LogInfo << "Making varied event rates for " << parSavePath_ << std::endl;

    // First make sure all params are at their prior <- is it necessary?
    for( auto& parSet : _owner_->getPropagator().getParameterSetsList() ){
      if( not parSet.isEnabled() ) continue;
      for( auto& par : parSet.getParameterList() ){
        par.setParameterValue(par.getPriorValue());
      }
    }
    _owner_->getPropagator().propagateParametersOnSamples();

    auto* saveDir = GenericToolbox::mkdirTFile(_saveDir_, parSavePath_ );
    saveDir->cd();

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

      _owner_->getPropagator().propagateParametersOnSamples();

      for(auto & sample : _owner_->getPropagator().getFitSampleSet().getFitSampleList()){
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
    GenericToolbox::writeInTFile(saveDir,
                                 variationList_TVectorD,
                                 "paramValues");

    TVectorD* buffVariedEvtRates_TVectorD{nullptr};

    for( size_t iSample = 0 ; iSample < _owner_->getPropagator().getFitSampleSet().getFitSampleList().size() ; iSample++ ){

      buffVariedEvtRates_TVectorD = new TVectorD(int(variationList_.size()));

      for ( int iVar = 0 ; iVar < buffVariedEvtRates_TVectorD->GetNrows() ; iVar++ ){
        (*buffVariedEvtRates_TVectorD)(iVar) = buffEvtRatesMap[iVar][iSample];
      }

      GenericToolbox::writeInTFile(saveDir,
                                   buffVariedEvtRates_TVectorD,
                                   _owner_->getPropagator().getFitSampleSet().getFitSampleList()[iSample].getName());

    }


  };

  // vary parameters

  for( auto& parSet : _owner_->getPropagator().getParameterSetsList() ){

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

        std::string savePath = savePath_;
        if( not savePath.empty() ) savePath += "/";
        savePath += "varyEventRates/" + parSet.getName() + "/" + par.getTitle() + tag;

        makeVariedEventRatesFct(par, paramVariationList_, savePath);

      }
    }

  }

  _saveDir_->cd();

}



