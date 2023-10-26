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

bool DataBin::Edges::isOverlapping(const DataBin::Edges& other_) const{
  if( this->isConditionVar != other_.isConditionVar ){
    LogError << "Mismatch with a variable: in one bin, it is a condition variable while in another it's a range." << std::endl;
    LogError << "ref edge: " << this->getSummary() << std::endl;
    LogError << "other edge: " << this->getSummary() << std::endl;
  }

  if( this->min == other_.min ){ return true; }
  if( this->max == other_.max ){ return true; }
  if( this->min > other_.min and this->min  < other_.max ){ return true; } // other is supposed to be lower
  if( this->max < other_.max and other_.min < this->max  ){ return true; } // other is supposed to be higher

  return false;
}
std::string DataBin::Edges::getSummary() const {
  std::stringstream ss;
  if( not this->varName.empty() ){ ss << this->varName << ": "; }
  ss << "[" << this->min;
  if( this->isConditionVar ){ ss << "]"; }
  else{
    ss << ", " << this->max << "[";
  }
  return ss.str();
}


// Setters
void DataBin::addBinEdge(const std::string &variableName_, double lowEdge_, double highEdge_) {
  if( this->isVariableSet(variableName_) ){
    LogError << __METHOD_NAME__ << ": variableName_ = " << variableName_ << " is already set." << std::endl;
    throw std::logic_error("Input variableName_ is already set.");
  }
  _binEdgesList_.emplace_back(_binEdgesList_.size());
  _binEdgesList_.back().min = std::min( lowEdge_, highEdge_ );
  _binEdgesList_.back().max = std::max( lowEdge_, highEdge_ );
  if( lowEdge_ == highEdge_ ){
    if( not _isZeroWideRangesTolerated_ ){
      LogError << GET_VAR_NAME_VALUE(_isZeroWideRangesTolerated_) << " but lowEdge_ == highEdge_ = " << lowEdge_ << std::endl;
      throw std::logic_error(GET_VAR_NAME_VALUE(_isZeroWideRangesTolerated_) + " but lowEdge_ == highEdge_.");
    }
    _binEdgesList_.back().isConditionVar = true;
  }
  _binEdgesList_.back().varName = variableName_;
}


// Getters
const DataBin::Edges& DataBin::getVarEdges( const std::string& varName_ ) const{
  auto* varEgdesPtr{getVarEdgesPtr(varName_)};
  LogThrowIf(varEgdesPtr == nullptr, "Couldn't find varEdges corresponding to varName:" << varName_);
  return *varEgdesPtr;
}
const DataBin::Edges* DataBin::getVarEdgesPtr( const std::string& varName_ ) const{
  int varIndex = GenericToolbox::findElementIndex( varName_, _binEdgesList_, [](const Edges& e_){ return e_.varName; });
  if( varIndex == -1 ){ return nullptr; }
  return &_binEdgesList_[varIndex];
}
double DataBin::getVolume() const{
  double out{1};
  for( auto& edges : _binEdgesList_ ){
    if( edges.isConditionVar ){ continue; } // no volume, just a condition variable
    out *= (edges.max - edges.min);
  }
  return out;
}


// Management
bool DataBin::isOverlapping(const DataBin& other_) const{
  std::vector<std::string> varNameList{this->buildVariableNameList()};
  GenericToolbox::mergeInVector(varNameList, other_.buildVariableNameList());

  // is overlapping only if all edges report "true"

  for( auto& var : varNameList ){
    auto* edgesPtr = this->getVarEdgesPtr(var);
    if( edgesPtr == nullptr ){ continue; } // no condition, considered as a possible overlap

    auto* edgesOtherPtr = other_.getVarEdgesPtr(var);
    if( edgesOtherPtr == nullptr ){ continue; } // no condition, considered as a possible overlap

    if( not edgesPtr->isOverlapping(*edgesOtherPtr) ){ return false; }
  }

  // if reached this point, all edges reported an overlap
  return true;
}
bool DataBin::isInBin(const std::vector<double>& valuesList_) const{
  LogThrowIf( valuesList_.size() != _binEdgesList_.size(),
              "Provided " << GET_VAR_NAME_VALUE(valuesList_.size()) << " does not match " << GET_VAR_NAME_VALUE(_binEdgesList_.size()));
  const double* buf = &valuesList_[0];

  // is "all_of" the variables between defined edges
  return std::all_of(_binEdgesList_.begin(), _binEdgesList_.end(), [&](const Edges& edge){ return (this->isBetweenEdges(edge, *(buf++))); });
}
bool DataBin::isVariableSet(const std::string& variableName_) const{
  if( _isLowMemoryUsageMode_ ){
    LogError << "Can't fetch variable name while in low memory mode. (var name is not stored)" << std::endl;
    throw std::logic_error("can't fetch var name while _isLowMemoryUsageMode_");
  }
  return GenericToolbox::findElementIndex(variableName_, _binEdgesList_, [](const Edges& e_){ return e_.varName; }) != -1;
}
bool DataBin::isBetweenEdges(const std::string& variableName_, double value_) const {
  if( not this->isVariableSet(variableName_) ){
    LogError << "variableName_ = " << variableName_ << " is not set." << std::endl;
    throw std::logic_error("variableName_ not set.");
  }
  int varIndex = GenericToolbox::findElementIndex(variableName_, _binEdgesList_, [](const Edges& e_){ return e_.varName; });
  return this->isBetweenEdges(varIndex, value_);
}
bool DataBin::isBetweenEdges(size_t varIndex_, double value_) const{

  if( varIndex_ >= _binEdgesList_.size() ){
    LogError << "Provided " << GET_VAR_NAME_VALUE(varIndex_) << " is invalid: " << GET_VAR_NAME_VALUE(_binEdgesList_.size()) << std::endl;
    throw std::runtime_error("varIndex out of range.");
  }

  return this->isBetweenEdges(_binEdgesList_[varIndex_], value_);
}
bool DataBin::isBetweenEdges(const Edges& edges_, double value_) const {
  // condition variable?
  if(edges_.min == edges_.max ){ return (edges_.min == value_); }

  // reject?
  if( _includeLowerBoundVal_ ? (edges_.min > value_ ) : (edges_.min >= value_) ){ return false; }
  if( _includeHigherBoundVal_ ? (edges_.max < value_ ) : (edges_.max <= value_) ){ return false; }

  // inside
  return true;
}
std::vector<std::string> DataBin::buildVariableNameList() const{
  std::vector<std::string> out;
  for( auto& edges : this->getEdgesList() ){
    GenericToolbox::addIfNotInVector(edges.varName, out);
  }
  return out;
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
  varNameList.reserve( _binEdgesList_.size() );
  for( auto& edges : _binEdgesList_ ){
    std::string nameCandidate{(edges.varName.empty() ? "var" + std::to_string(edges.index): edges.varName)};
    if( not GenericToolbox::doesElementIsInVector(nameCandidate, varNameList) ){
      varNameList.emplace_back(nameCandidate);
    }
  }
  _treeFormula_ = std::shared_ptr<TTreeFormula>(GenericToolbox::createTreeFormulaWithoutTree(_treeFormulaStr_, varNameList));
}
std::string DataBin::getSummary() const{
  std::stringstream ss;

  if( _binEdgesList_.empty() ) ss << "bin not set.";
  else{

    if( _treeFormula_ != nullptr ){
      ss << "\"" << _treeFormula_->GetExpFormula() << "\"";
    }
    else{
      for( auto& edges : _binEdgesList_ ){
        if( edges.index != 0 ){ ss << ", "; }
        ss << edges.getSummary();
      }
    }
  }
  return ss.str();
}
std::vector<double> DataBin::generateBinTarget( const std::vector<std::string>& varNameList_ ) const{
  std::vector<double> out;
  out.reserve( _binEdgesList_.size() );
  for( auto& varName : varNameList_ ){
    auto& edges{this->getVarEdges(varName)};
    out.emplace_back( edges.min + (edges.max - edges.min )/ 2. );
  }
  return out;
}


// Protected
std::string DataBin::generateFormulaStr(bool varNamesAsTreeFormula_) {
  std::stringstream ss;
  for( auto& edges : _binEdgesList_ ){

    std::string varName{};
    if( edges.varName.empty() ){ varName = "var" + std::to_string(edges.index); }
    else{ varName = edges.varName; }

    if(not varNamesAsTreeFormula_){
      // Formula: no array authorized, putting parenthesis instead
      GenericToolbox::replaceSubstringInsideInputString(varName, "[", "(");
      GenericToolbox::replaceSubstringInsideInputString(varName, "]", ")");
      // Wrapping varname in brackets
      varName.insert(varName.begin(), '[');
      varName += ']';
    }

    if( not ss.str().empty() ){
      ss << " && ";
    }

    if( edges.isConditionVar ){ ss << varName << " == " << edges.min; }
    else{ ss << edges.min << " <= " << varName << " && " << varName << " < " << edges.max; }
  }

  return ss.str();
}


