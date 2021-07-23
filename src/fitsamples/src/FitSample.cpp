//
// Created by Adrien BLANCHET on 22/07/2021.
//

#include "GenericToolbox.h"
#include "Logger.h"

#include "JsonUtils.h"
#include "FitSample.h"

LoggerInit([](){
  Logger::setUserHeaderStr("[FitSample]");
})


FitSample::FitSample() { this->reset(); }
FitSample::~FitSample() { this->reset(); }

void FitSample::reset() {
  _config_.clear();

  _isEnabled_ = false;
  _name_ = "";

  _binning_.reset();

  _selectionCuts_ = "";
  _selectionCutsTreeFormula_ = nullptr;

  _mcNorm_ = 1;
  _mcEventList_.clear();
  _mcHistogram_ = nullptr;

  _dataNorm_ = 1;
  _dataEventList_.clear();
  _dataHistogram_ = nullptr;
}

void FitSample::setConfig(const nlohmann::json &config_) {
  _config_ = config_;
}

void FitSample::initialize() {
  LogWarning << __METHOD_NAME__ << std::endl;

  if( _config_.empty() ){
    LogError << GET_VAR_NAME_VALUE(_config_.empty()) << std::endl;
    throw std::logic_error(GET_VAR_NAME_VALUE(_config_.empty()));
  }

  _name_ = JsonUtils::fetchValue<std::string>(_config_, "name");
  _isEnabled_ = JsonUtils::fetchValue(_config_, "isEnabled", true);
  if( not _isEnabled_ ){
    LogWarning << "\"" << _name_ << "\" is disabled." << std::endl;
    return;
  }

  _binning_.readBinningDefinition( JsonUtils::fetchValue<std::string>(_config_, "binning") );
  _mcHistogram_ = std::shared_ptr<TH1D>(
    new TH1D(
      Form("%s_MC_bins", _name_.c_str()), Form("%s_MC_bins", _name_.c_str()),
      int(_binning_.getBinsList().size()), 0, int(_binning_.getBinsList().size())
    )
  );
  _dataHistogram_ = std::shared_ptr<TH1D>(
    new TH1D(
      Form("%s_Data_bins", _name_.c_str()), Form("%s_Data_bins", _name_.c_str()),
      int(_binning_.getBinsList().size()), 0, int(_binning_.getBinsList().size())
    )
  );

  _selectionCuts_ = JsonUtils::fetchValue(_config_, "selectionCuts", _selectionCuts_);
  _selectedDataSets_ = JsonUtils::fetchValue(_config_, "dataSets", _selectedDataSets_);

  _mcNorm_ = JsonUtils::fetchValue(_config_, "mcNorm", _mcNorm_);
  _dataNorm_ = JsonUtils::fetchValue(_config_, "dataNorm", _dataNorm_);
}

void FitSample::fillEventsList(const std::vector<DataSet> &dataSetList_) {

  for( const auto& dataSet : dataSetList_){

    if( not dataSet.isEnabled() ){ continue; }
    if( not _selectedDataSets_.empty()
        and not GenericToolbox::doesElementIsInVector(dataSet.getName(), _selectedDataSets_) ){
      // skip
      continue;
    }



  }

}
