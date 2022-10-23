//
// Created by Adrien BLANCHET on 07/04/2022.
//

#include "ParScanner.h"
#include "JsonUtils.h"

#include "Logger.h"

#include <utility>


LoggerInit([]{
  Logger::setUserHeaderStr("[Scanner]");
});


void ParScanner::readConfigImpl() {
  if( _config_.empty() ) return;

  _useParameterLimits_ = JsonUtils::fetchValue(_config_, "useParameterLimits", _useParameterLimits_);
  _nbPoints_ = JsonUtils::fetchValue(_config_, "nbPoints", _nbPoints_);
  _parameterSigmaRange_ = JsonUtils::fetchValue(_config_, "parameterSigmaRange", _parameterSigmaRange_);

  _varsConfig_ = JsonUtils::fetchValue(_config_, "varsConfig", nlohmann::json());
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

//
//void ParScanner::scanParameters(int nbSteps_, const std::string &saveDir_) {
//  LogInfo << "Performing parameter scans..." << std::endl;
//  for( int iPar = 0 ; iPar < _minimizer_->NDim() ; iPar++ ){
//    if( _minimizer_->IsFixedVariable(iPar) ) continue;
//    this->scanParameter(iPar, nbSteps_, saveDir_);
//  } // iPar
//}
//void ParScanner::scanParameter(int iPar, int nbSteps_, const std::string &saveDir_) {
//  if( nbSteps_ < 0 ){ nbSteps_ = _scanConfig_.getNbPoints(); }
//
//  std::vector<double> parPoints(nbSteps_+1,0);
//
//  std::stringstream ssPbar;
//  ssPbar << LogInfo.getPrefixString() << "Scanning fit parameter #" << iPar
//         << ": " << _minimizer_->VariableName(iPar) << " / " << nbSteps_ << " steps...";
//  GenericToolbox::displayProgressBar(0, nbSteps_, ssPbar.str());
//
//  scanDataDict.clear();
//  if( JsonUtils::fetchValue(_scanConfig_.getVarsConfig(), "llh", true) ){
//    scanDataDict.emplace_back();
//    auto& scanEntry = scanDataDict.back();
//    scanEntry.yPoints = std::vector<double>(nbSteps_+1,0);
//    scanEntry.folder = "llh";
//    scanEntry.title = "Total Likelihood Scan";
//    scanEntry.yTitle = "LLH value";
//    scanEntry.evalY = [this](){ return this->_chi2Buffer_; };
//  }
//  if( JsonUtils::fetchValue(_scanConfig_.getVarsConfig(), "llhPenalty", true) ){
//    scanDataDict.emplace_back();
//    auto& scanEntry = scanDataDict.back();
//    scanEntry.yPoints = std::vector<double>(nbSteps_+1,0);
//    scanEntry.folder = "llhPenalty";
//    scanEntry.yPoints = std::vector<double>(nbSteps_+1,0);
//    scanEntry.title = "Penalty Likelihood Scan";
//    scanEntry.yTitle = "Penalty LLH value";
//    scanEntry.evalY = [this](){ return this->_chi2PullsBuffer_; };
//  }
//  if( JsonUtils::fetchValue(_scanConfig_.getVarsConfig(), "llhStat", true) ){
//    scanDataDict.emplace_back();
//    auto& scanEntry = scanDataDict.back();
//    scanEntry.yPoints = std::vector<double>(nbSteps_+1,0);
//    scanEntry.folder = "llhStat";
//    scanEntry.title = "Stat Likelihood Scan";
//    scanEntry.yTitle = "Stat LLH value";
//    scanEntry.evalY = [this](){ return this->_chi2StatBuffer_; };
//  }
//  if( JsonUtils::fetchValue(_scanConfig_.getVarsConfig(), "llhStatPerSample", false) ){
//    for( auto& sample : _propagator_.getFitSampleSet().getFitSampleList() ){
//      scanDataDict.emplace_back();
//      auto& scanEntry = scanDataDict.back();
//      scanEntry.yPoints = std::vector<double>(nbSteps_+1,0);
//      scanEntry.folder = "llhStat/" + sample.getName() + "/";
//      scanEntry.title = Form("Stat Likelihood Scan of sample \"%s\"", sample.getName().c_str());
//      scanEntry.yTitle = "Stat LLH value";
//      auto* samplePtr = &sample;
//      scanEntry.evalY = [this, samplePtr](){ return _propagator_.getFitSampleSet().evalLikelihood(*samplePtr); };
//    }
//  }
//  if( JsonUtils::fetchValue(_scanConfig_.getVarsConfig(), "llhStatPerSamplePerBin", false) ){
//    for( auto& sample : _propagator_.getFitSampleSet().getFitSampleList() ){
//      for( int iBin = 1 ; iBin <= sample.getMcContainer().histogram->GetNbinsX() ; iBin++ ){
//        scanDataDict.emplace_back();
//        auto& scanEntry = scanDataDict.back();
//        scanEntry.yPoints = std::vector<double>(nbSteps_+1,0);
//        scanEntry.folder = "llhStat/" + sample.getName() + "/bin_" + std::to_string(iBin);
//        scanEntry.title = Form(R"(Stat LLH Scan of sample "%s", bin #%d "%s")",
//                               sample.getName().c_str(),
//                               iBin,
//                               sample.getBinning().getBinsList()[iBin-1].getSummary().c_str());
//        scanEntry.yTitle = "Stat LLH value";
//        auto* samplePtr = &sample;
//        scanEntry.evalY = [this, samplePtr, iBin](){ return _propagator_.getFitSampleSet().getJointProbabilityFct()->eval(*samplePtr, iBin); };
//      }
//    }
//  }
//  if( JsonUtils::fetchValue(_scanConfig_.getVarsConfig(), "weightPerSample", false) ){
//    for( auto& sample : _propagator_.getFitSampleSet().getFitSampleList() ){
//      scanDataDict.emplace_back();
//      auto& scanEntry = scanDataDict.back();
//      scanEntry.yPoints = std::vector<double>(nbSteps_+1,0);
//      scanEntry.folder = "weight/" + sample.getName();
//      scanEntry.title = Form("MC event weight scan of sample \"%s\"", sample.getName().c_str());
//      scanEntry.yTitle = "Total MC event weight";
//      auto* samplePtr = &sample;
//      scanEntry.evalY = [samplePtr](){ return samplePtr->getMcContainer().getSumWeights(); };
//    }
//  }
//  if( JsonUtils::fetchValue(_scanConfig_.getVarsConfig(), "weightPerSamplePerBin", false) ){
//    for( auto& sample : _propagator_.getFitSampleSet().getFitSampleList() ){
//      for( int iBin = 1 ; iBin <= sample.getMcContainer().histogram->GetNbinsX() ; iBin++ ){
//        scanDataDict.emplace_back();
//        auto& scanEntry = scanDataDict.back();
//        scanEntry.yPoints = std::vector<double>(nbSteps_+1,0);
//        scanEntry.folder = "weight/" + sample.getName() + "/bin_" + std::to_string(iBin);
//        scanEntry.title = Form(R"(MC event weight scan of sample "%s", bin #%d "%s")",
//                               sample.getName().c_str(),
//                               iBin,
//                               sample.getBinning().getBinsList()[iBin-1].getSummary().c_str());
//        scanEntry.yTitle = "Total MC event weight";
//        auto* samplePtr = &sample;
//        scanEntry.evalY = [samplePtr, iBin](){ return samplePtr->getMcContainer().histogram->GetBinContent(iBin); };
//      }
//    }
//  }
//
//  double origVal = _minimizerFitParameterPtr_[iPar]->getParameterValue();
//  double lowBound = origVal + _scanConfig_.getParameterSigmaRange().first * _minimizerFitParameterPtr_[iPar]->getStdDevValue();
//  double highBound = origVal + _scanConfig_.getParameterSigmaRange().second * _minimizerFitParameterPtr_[iPar]->getStdDevValue();
//
//  if( _scanConfig_.isUseParameterLimits() ){
//    lowBound = std::max(lowBound, _minimizerFitParameterPtr_[iPar]->getMinValue());
//    highBound = std::min(highBound, _minimizerFitParameterPtr_[iPar]->getMaxValue());
//  }
//
//  int offSet{0};
//  for( int iPt = 0 ; iPt < nbSteps_+1 ; iPt++ ){
//    GenericToolbox::displayProgressBar(iPt, nbSteps_, ssPbar.str());
//
//    double newVal = lowBound + double(iPt-offSet)/(nbSteps_-1)*( highBound - lowBound );
//    if( offSet == 0 and newVal > origVal ){
//      newVal = origVal;
//      offSet = 1;
//    }
//
//    _minimizerFitParameterPtr_[iPar]->setParameterValue(newVal);
//    this->updateChi2Cache();
//    parPoints[iPt] = _minimizerFitParameterPtr_[iPar]->getParameterValue();
//
//    for( auto& scanEntry : scanDataDict ){ scanEntry.yPoints[iPt] = scanEntry.evalY(); }
//  }
//
//
//  _minimizerFitParameterPtr_[iPar]->setParameterValue(origVal);
//
//  std::stringstream ss;
//  ss << GenericToolbox::replaceSubstringInString(_minimizer_->VariableName(iPar), "/", "_");
//  ss << "_TGraph";
//
//  for( auto& scanEntry : scanDataDict ){
//    TGraph scanGraph(int(parPoints.size()), &parPoints[0], &scanEntry.yPoints[0]);
//    scanGraph.SetTitle(scanEntry.title.c_str());
//    scanGraph.GetYaxis()->SetTitle(scanEntry.yTitle.c_str());
//    scanGraph.GetXaxis()->SetTitle(_minimizer_->VariableName(iPar).c_str());
//    scanGraph.SetDrawOption("AP");
//    scanGraph.SetMarkerStyle(kFullDotLarge);
//    if( _saveDir_ != nullptr ){
//      GenericToolbox::mkdirTFile(_saveDir_, saveDir_ + "/" + scanEntry.folder )->cd();
//      scanGraph.Write( ss.str().c_str() );
//    }
//  }
//}

