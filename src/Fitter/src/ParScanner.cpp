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

void ParScanner::scanFitParameters(const std::string& saveSubdir_){
  LogInfo << "Performing scans of fit parameters..." << std::endl;
  for( int iPar = 0 ; iPar < _owner_->getMinimizer().getMinimizer()->NDim() ; iPar++ ){
    if( _owner_->getMinimizer().getMinimizer()->IsFixedVariable(iPar) ){
      LogWarning << _owner_->getMinimizer().getMinimizer()->VariableName(iPar)
      << " is fixed. Skipping..." << std::endl;
      continue;
    }
    this->scanParameter(*_owner_->getMinimizer().getMinimizerFitParameterPtr()[iPar], saveSubdir_);
    _owner_->getMinimizer().getMinimizerFitParameterPtr();
  } // iPar
}
void ParScanner::scanParameter(FitParameter& par_, const std::string &saveDir_) {
  std::vector<double> parPoints(_nbPoints_+1,0);

  std::stringstream ssPbar;
  ssPbar << LogInfo.getPrefixString() << "Scanning fit parameter: " << par_.getFullTitle() << " / " << _nbPoints_ << " steps...";
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
      GenericToolbox::mkdirTFile(_saveDir_, saveDir_ + "/" + scanEntry.folder )->cd();
      scanGraph.Write( ss.str().c_str() );
    }
  }
}


