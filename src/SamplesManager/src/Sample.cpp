//
// Created by Nadrino on 22/07/2021.
//

#include "Sample.h"
#include "GundamGlobals.h"

#include "Logger.h"
#include "GenericToolbox.Thread.h"
#include "GenericToolbox.Loops.h"

#include <string>
#include <memory>


void Sample::prepareConfig(ConfigReader &config_){
  config_.clearFields();
  config_.defineFields({
    {"name", true},
    {"isEnabled"},
    {"disableEventMcThrow"},
    {"binning", {"binningFile", "binningFilePath"}},
    {"selectionCutStr", {"selectionCuts"}},
    {"datasets"},
  });
  config_.checkConfiguration();
}
void Sample::configureImpl(){
  prepareConfig(_config_);

  _config_.fillValue(_name_, "name");
  _config_.fillValue(_isEnabled_, "isEnabled");
  _config_.fillValue(_disableEventMcThrow_, "disableEventMcThrow");
  _config_.fillValue(_binningConfig_, "binning");
  _config_.fillValue(_selectionCutStr_, "selectionCutStr");
  _config_.fillValue(_enabledDatasetList_, "datasets");

  LogThrowIf(_name_.empty(), "No name was provided for sample #" << _index_ << std::endl << _config_);
  LogDebugIf(GundamGlobals::isDebug()) << "Defining sample \"" << _name_ << "\"" << std::endl;
  if( not _isEnabled_ ){
    LogDebugIf(GundamGlobals::isDebug()) << "-> disabled" << std::endl;
    return;
  }

  LogDebugIf(GundamGlobals::isDebug()) << "Reading binning: " << _config_ << std::endl;
  _histogram_.build(_binningConfig_);
}

void Sample::writeEventRates(const GenericToolbox::TFilePath& saveDir_) const{
  GenericToolbox::writeInTFileWithObjTypeExt(saveDir_.getSubDir(_name_).getDir(), getSumWeights(), "sumWeights");
}
bool Sample::isDatasetValid(const std::string& datasetName_) const {
  if( _enabledDatasetList_.empty() ) return true;
  return std::any_of(
      _enabledDatasetList_.begin(), _enabledDatasetList_.end(),
      [&](const std::string& enabled_){
        return (enabled_ == "*" or enabled_ == datasetName_);
      }
  );
}
double Sample::getSumWeights() const{
  double output = std::accumulate(_eventList_.begin(), _eventList_.end(), double(0.),
                                  [](double sum_, const Event& ev_){ return sum_ + ev_.getEventWeight(); });
  return output;
}
size_t Sample::getNbBinnedEvents() const{
  return std::accumulate(
      _eventList_.begin(), _eventList_.end(), size_t(0.),
      []( size_t sum_, const Event &ev_ ){
        return sum_ + (ev_.getIndices().bin != -1);
      });
}

void Sample::reserveEventMemory(size_t dataSetIndex_, size_t nEvents, const Event &eventBuffer_) {
  // adding one dataset:
  _loadedDatasetList_.emplace_back();

  // filling up properties:
  auto& datasetProperties{_loadedDatasetList_.back()};
  datasetProperties.dataSetIndex = dataSetIndex_;
  datasetProperties.eventOffSet = _eventList_.size();
  datasetProperties.eventNb = nEvents;

  _eventList_.resize(datasetProperties.eventOffSet + datasetProperties.eventNb, eventBuffer_);
}
void Sample::shrinkEventList(size_t newTotalSize_){

  if( _loadedDatasetList_.empty() and newTotalSize_ == 0 ){
    LogAlert << "Empty dataset list. Nothing to shrink." << std::endl;
    return;
  }

  LogThrowIf(_eventList_.size() < newTotalSize_,
             "Can't shrink since eventList is too small: " << GET_VAR_NAME_VALUE(newTotalSize_)
                                                           << " > " << GET_VAR_NAME_VALUE(_eventList_.size()));

  LogThrowIf(not _loadedDatasetList_.empty() and _loadedDatasetList_.back().eventNb < (_eventList_.size() - newTotalSize_),
             "Can't shrink since eventList of the last dataSet is too small.");

  LogInfo << _name_ << ": shrinking event list from " << _eventList_.size() << " to " << newTotalSize_ << "..." << std::endl;

  _loadedDatasetList_.back().eventNb -= (_eventList_.size() - newTotalSize_);
  _eventList_.resize(newTotalSize_);
  _eventList_.shrink_to_fit();
}

void Sample::printConfiguration() const{

  LogInfo << "#" << _index_;
  LogInfo << ", name(" << _name_ << ")";
  LogInfo << ", nBins(" << _histogram_.getNbBins() << ")";
  LogInfo << std::endl;

}
[[nodiscard]] std::string Sample::getSummary() const{
  std::stringstream ss;
  ss << "Sample: " << _name_ << std::endl;
  ss << " - " << "Nb bins: " << _histogram_.getNbBins() << std::endl;
  ss << " - " << "Nb events: " << _eventList_.size();
  return ss.str();
}
std::ostream& operator <<( std::ostream& o, const Sample& this_ ){
  o << this_.getSummary(); return o;
}

void Sample::indexEventInHistogramBin( int iThread_){
  _histogram_.updateBinEventList(_eventList_, iThread_);
}
