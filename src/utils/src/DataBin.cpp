//
// Created by Adrien BLANCHET on 19/05/2021.
//

#include "stdexcept"
#include "sstream"

#include "Logger.h"
#include "GenericToolbox.h"
#include "GenericToolboxRootExt.h"

#include "DataBin.h"

LoggerInit([](){
  Logger::setUserHeaderStr("[DataBin]");
} )

DataBin::DataBin() {
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
  _formulaStr_ = "";
  _treeFormulaStr_ = "";
  _formula_ = nullptr;
  _treeFormula_ = nullptr;
}

// Setters
void DataBin::setIsLowMemoryUsageMode(bool isLowMemoryUsageMode_){
  _isLowMemoryUsageMode_ = isLowMemoryUsageMode_;
}
void DataBin::setIsZeroWideRangesTolerated(bool isZeroWideRangesTolerated_){
  _isZeroWideRangesTolerated_ = isZeroWideRangesTolerated_;
}
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
const std::string &DataBin::getFormulaStr() const {
  return _formulaStr_;
}
const std::string &DataBin::getTreeFormulaStr() const {
  return _treeFormulaStr_;
}
TFormula *DataBin::getFormula() const {
  return _formula_.get();
}
TTreeFormula *DataBin::getTreeFormula() const {
  return _treeFormula_.get();
}

// Management
bool DataBin::isInBin(const std::vector<double>& valuesList_) const{
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
bool DataBin::isBetweenEdges(const std::string& variableName_, double value_) const {
  if( not this->isVariableSet(variableName_) ){
    LogError << "variableName_ = " << variableName_ << " is not set." << std::endl;
    throw std::logic_error("variableName_ not set.");
  }
  int varIndex = GenericToolbox::findElementIndex(variableName_, _variableNameList_);
  this->isBetweenEdges(varIndex, value_);
}
bool DataBin::isEventInBin(AnaEvent *eventPtr_) const {
  for( size_t iVar = 0 ; iVar < _edgesList_.size() ; iVar++ ){
    if( not this->isBetweenEdges(iVar, eventPtr_->GetEventVarAsDouble( _variableNameList_.at(iVar) )) ){
      return false;
    }
  }
  return true;
}

// Misc
bool DataBin::isVariableSet(const std::string& variableName_) const{
  if( _isLowMemoryUsageMode_ ){
    LogError << "Can't fetch variable name while in low memory mode. (var name is not stored)" << std::endl;
    throw std::logic_error("can't fetch var name while _isLowMemoryUsageMode_");
  }
  return GenericToolbox::findElementIndex(variableName_, _variableNameList_) != -1;
}
std::string DataBin::getSummary() const{
  std::stringstream ss;
  ss << "DataBin: ";

  if( _edgesList_.empty() ) ss << "Not set.";
  else{

    if( _treeFormula_ != nullptr ){
      ss << "\"" << _treeFormula_->GetExpFormula() << "\"";
    }
    else{
      ss << "\"";
      for( size_t iEdge = 0 ; iEdge < _edgesList_.size() ; iEdge++ ){
        if( iEdge != 0 ) ss << ", ";
        if( not _variableNameList_.empty() ){
          ss << _variableNameList_[iEdge] << ": ";
        }
        ss << "[" << _edgesList_.at(iEdge).first;
        if( _edgesList_.at(iEdge).first != _edgesList_.at(iEdge).second ){
          ss << ", " << _edgesList_.at(iEdge).second;
          ss << "[";
        }
        else{
          ss << "]";
        }
      }
      ss << "\"";
    }
  }
  return ss.str();
}
void DataBin::generateFormula() {

  _formulaStr_ = generateFormulaStr(false);
  _formula_ = std::make_shared<TFormula>(_formulaStr_.c_str(), _formulaStr_.c_str());

}
void DataBin::generateTreeFormula() {

  _treeFormulaStr_ = generateFormulaStr(true);
  // For treeFormula we need a fake tree to compile the formula
  std::vector<std::string> varNameList;
  for( size_t iEdge = 0 ; iEdge < _edgesList_.size() ; iEdge++ ){
    std::string nameCandidate;
    if( not _variableNameList_.empty() ){
      // Careful: array might be defined
      nameCandidate = GenericToolbox::splitString(_variableNameList_.at(iEdge), "[")[0];
    }
    else{
      nameCandidate = "var" + std::to_string(iEdge);
    }
    if( not GenericToolbox::doesElementIsInVector(nameCandidate, varNameList) ){
      varNameList.emplace_back(nameCandidate);
    }
  }
  _treeFormula_ = std::shared_ptr<TTreeFormula>(GenericToolbox::createTreeFormulaWithoutTree(_treeFormulaStr_, varNameList));

}

// Protected
std::string DataBin::generateFormulaStr(bool varNamesAsTreeFormula_) {

  std::stringstream ss;

  for( size_t iVar = 0 ; iVar < _variableNameList_.size() ; iVar++ ){

    std::string varName;
    if( not _variableNameList_.empty() ){
      varName += _variableNameList_.at(iVar);
    }
    else{
      varName += "var" + std::to_string(iVar);
    }

    if(not varNamesAsTreeFormula_){
      // Formula: no array authorized, putting parenthesis instead
      GenericToolbox::replaceSubstringInsideInputString(varName, "[", "(");
      GenericToolbox::replaceSubstringInsideInputString(varName, "]", ")");
      // Wrapping varname in brackets
      varName = "[" + varName;
      varName += "]";
    }

    if( not ss.str().empty() ){
      ss << " && ";
    }

    if( _edgesList_.at(iVar).first == _edgesList_.at(iVar).second ){
      ss << varName << " == " << _edgesList_.at(iVar).first;
    }
    else{
      ss << _edgesList_.at(iVar).first << " <= " << varName << " && ";
      ss << varName << " < " << _edgesList_.at(iVar).second;
    }
  }

  return ss.str();
}
bool DataBin::isBetweenEdges(size_t varIndex_, double value_) const{

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
