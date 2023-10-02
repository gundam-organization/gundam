//
// Created by Nadrino on 19/05/2021.
//

#include "DataBin.h"

#include "Logger.h"
#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"

#include <stdexcept>
#include <sstream>

LoggerInit([]{
  Logger::setUserHeaderStr("[DataBin]");
} );

// Setters
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
const std::pair<double, double>& DataBin::getVarEdges( const std::string& varName_ ) const{
  int varIndex = GenericToolbox::findElementIndex(varName_, _variableNameList_);
  LogThrowIf(varIndex == -1, varName_ << " not found in: " << GenericToolbox::parseVectorAsString(_variableNameList_));
  return _edgesList_.at(varIndex);
}
double DataBin::getVolume() const{
  double out{1};
  for( auto& edges : _edgesList_ ){
    if( edges.first == edges.second ) continue; // no volume, just a condition variable
    out *= std::max(edges.first, edges.second) - std::min(edges.first, edges.second);
  }
  return out;
}


// Management
bool DataBin::isInBin(const std::vector<double>& valuesList_) const{
  LogThrowIf( valuesList_.size() != _edgesList_.size(),
              "Provided " << GET_VAR_NAME_VALUE(valuesList_.size()) << " does not match " << GET_VAR_NAME_VALUE(_edgesList_.size()));
  const double* buf = &valuesList_[0];

  // is "all_of" the variables between defined edges
  return std::all_of(_edgesList_.begin(), _edgesList_.end(), [&](const std::pair<double, double>& edge){ return (this->isBetweenEdges(edge, *(buf++))); });
}
bool DataBin::isVariableSet(const std::string& variableName_) const{
  if( _isLowMemoryUsageMode_ ){
    LogError << "Can't fetch variable name while in low memory mode. (var name is not stored)" << std::endl;
    throw std::logic_error("can't fetch var name while _isLowMemoryUsageMode_");
  }
  return GenericToolbox::findElementIndex(variableName_, _variableNameList_) != -1;
}
bool DataBin::isBetweenEdges(const std::string& variableName_, double value_) const {
  if( not this->isVariableSet(variableName_) ){
    LogError << "variableName_ = " << variableName_ << " is not set." << std::endl;
    throw std::logic_error("variableName_ not set.");
  }
  int varIndex = GenericToolbox::findElementIndex(variableName_, _variableNameList_);
  return this->isBetweenEdges(varIndex, value_);
}
bool DataBin::isBetweenEdges(size_t varIndex_, double value_) const{

  if( varIndex_ >= _edgesList_.size() ){
    LogError << "Provided " << GET_VAR_NAME_VALUE(varIndex_) << " is invalid: " << GET_VAR_NAME_VALUE(_edgesList_.size()) << std::endl;
    throw std::runtime_error("varIndex out of range.");
  }

  return this->isBetweenEdges(_edgesList_.at(varIndex_), value_);
}
bool DataBin::isBetweenEdges(const std::pair<double,double>& edges_, double value_) const {
  // condition variable?
  if(edges_.first == edges_.second ){ return (edges_.first == value_); }

  // reject?
  if( _includeLowerBoundVal_ ? (edges_.first > value_ ) : (edges_.first >= value_) ){ return false; }
  if( _includeHigherBoundVal_ ? (edges_.second < value_ ) : (edges_.second <= value_) ){ return false; }

  // inside
  return true;
}


// Misc
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
std::string DataBin::getSummary() const{
  std::stringstream ss;

  if( _edgesList_.empty() ) ss << "bin not set.";
  else{

    if( _treeFormula_ != nullptr ){
      ss << "\"" << _treeFormula_->GetExpFormula() << "\"";
    }
    else{
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
    }
  }
  return ss.str();
}
std::vector<double> DataBin::generateBinTarget( const std::vector<std::string>& varNameList_ ) const{
  std::vector<double> out;
  out.reserve( _edgesList_.size() );

  for( auto& var : (varNameList_.empty() ? _variableNameList_ : varNameList_) ){
    LogThrowIf( not GenericToolbox::doesElementIsInVector(var, _variableNameList_),
                "Could not find " << var << " within " << GenericToolbox::parseVectorAsString(_variableNameList_));
    auto& edges = this->getVarEdges( var );
    out.emplace_back(edges.first);
    if( edges.first != edges.second ){
      out.back() = edges.first + (edges.second - edges.first )/ 2.;
    }
  }
  return out;
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


