//
// Created by Adrien BLANCHET on 19/05/2021.
//

#include "../include/DataBin.h"

#include "Logger.h"
#include "GenericToolbox.h"
#include "stdexcept"
#include "sstream"

DataBin::DataBin() {
  Logger::setUserHeaderStr("[DataBin]");
  reset();
}
DataBin::~DataBin() {
  reset();
}

void DataBin::reset() {
  _edgesList_.clear();
  _variableNameList_.clear();
  _isLowMemoryUsageMode_ = false;
  _isZeroWideRangesTolerated_ = false;
}

// Setters
void DataBin::setIsLowMemoryUsageMode(bool isLowMemoryUsageMode_){
  _isLowMemoryUsageMode_ = isLowMemoryUsageMode_;
}
void DataBin::setIsZeroWideRangesTolerated(bool isZeroWideRangesTolerated_){
  _isZeroWideRangesTolerated_ = isZeroWideRangesTolerated_;
}

// Getters
bool DataBin::isLowMemoryUsageMode() const {
  return _isLowMemoryUsageMode_;
}
bool DataBin::isZeroWideRangesTolerated() const {
  return _isZeroWideRangesTolerated_;
}
const std::vector<std::string> &DataBin::getVariableNameList() const {
  return _variableNameList_;
}
const std::vector<std::pair<double, double>> &DataBin::getEdgesList() const {
  return _edgesList_;
}

// Management
void DataBin::addBinEdge(double lowEdge_, double highEdge_){
  if( lowEdge_ == highEdge_ and not _isZeroWideRangesTolerated_ ){
    LogError << GET_VAR_NAME_VALUE(_isZeroWideRangesTolerated_) << " but lowEdge_ == highEdge_." << std::endl;
    throw std::logic_error(GET_VAR_NAME_VALUE(_isZeroWideRangesTolerated_) + " but lowEdge_ == highEdge_.");
  }
  _edgesList_.emplace_back(std::make_pair(lowEdge_, highEdge_));
}
void DataBin::addBinEdge(const std::string &variableName_, double lowEdge_, double highEdge_) {
  if( this->isVariableSet(variableName_) ){
    LogError << __METHOD_NAME__ << ": variableName_ = " << variableName_ << " is already set." << std::endl;
    throw std::logic_error("Input variableName_ is already set.");
  }
  _variableNameList_.emplace_back(variableName_);

  this->addBinEdge(lowEdge_, highEdge_);
}
bool DataBin::isInBin(const std::vector<double>& valuesList_){
  if( valuesList_.size() != _edgesList_.size() ){
    LogError << "Provided " << GET_VAR_NAME_VALUE(valuesList_.size()) << " does not match " << GET_VAR_NAME_VALUE(_edgesList_.size()) << std::endl;
    throw std::logic_error("Values list size does not match the bin edge list size.");
  }

  for( size_t iVar = 0 ; iVar < _edgesList_.size() ; iVar++ ){
    if( not this->isBetweenEdges(iVar, valuesList_.at(iVar)) ){
      return false;
    }
  }
  return true;
}
bool DataBin::isBetweenEdges(const std::string& variableName_, double value_){
  if( not this->isVariableSet(variableName_) ){
    LogError << "variableName_ = " << variableName_ << " is not set." << std::endl;
    throw std::logic_error("variableName_ not set.");
  }
  int varIndex = GenericToolbox::findElementIndex(variableName_, _variableNameList_);
  this->isBetweenEdges(varIndex, value_);
}

// Misc
bool DataBin::isVariableSet(const std::string& variableName_){
  if( _isLowMemoryUsageMode_ ){
    LogError << "Can't fetch variable name while in low memory mode. (var name is not stored)" << std::endl;
    throw std::logic_error("can't fetch var name while _isLowMemoryUsageMode_");
  }
  return GenericToolbox::findElementIndex(variableName_, _variableNameList_) != -1;
}
std::string DataBin::generateSummary() const{
  std::stringstream ss;
  if( _edgesList_.empty() ) ss << "undefined bin." << std::endl;
  else{
    for( size_t iEdge = 0 ; iEdge < _edgesList_.size() ; iEdge++ ){
      if( iEdge != 0 ) ss << ", ";
      if( not _variableNameList_.empty() ){
        ss << _variableNameList_[iEdge] << ": ";
      }
      ss << "[ " << _edgesList_.at(iEdge).first;
      if( _edgesList_.at(iEdge).first != _edgesList_.at(iEdge).second ){
        ss << ", " << _edgesList_.at(iEdge).second;
      }
      ss << " ]";
    }
  }
  return ss.str();
}

// Protected
bool DataBin::isBetweenEdges(size_t varIndex_, double value_){

  if( varIndex_ >= _edgesList_.size() ){
    LogError << "Provided " << GET_VAR_NAME_VALUE(varIndex_) << " is invalid: " << GET_VAR_NAME_VALUE(_edgesList_.size()) << std::endl;
    throw std::runtime_error("varIndex out of range.");
  }

  const auto* edgePairPtr = &_edgesList_.at(varIndex_);

  if( edgePairPtr->first == edgePairPtr->second ){
    return edgePairPtr->first == value_;
  }

  if( edgePairPtr->first <= value_  and value_ < edgePairPtr->second ){
    return true;
  }
  return false;
}

