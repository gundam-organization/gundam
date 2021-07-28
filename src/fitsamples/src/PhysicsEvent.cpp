//
// Created by Nadrino on 22/07/2021.
//

#include "Logger.h"

#include "PhysicsEvent.h"

LoggerInit([](){
  Logger::setUserHeaderStr("[PhysicsEvent]");
})

PhysicsEvent::PhysicsEvent() { this->reset(); }
PhysicsEvent::~PhysicsEvent() { this->reset(); }

void PhysicsEvent::reset() {
  _commonLeafNameListPtr_ = nullptr;
  _leafContentList_.clear();
  _dataSetIndex_=-1;

  // Weight carriers
  _treeWeight_ = 1;
  _nominalWeight_ = 1;
  _eventWeight_ = 1;
}

void PhysicsEvent::setLeafNameListPtr(const std::vector<std::string> *leafNameListPtr) {
  _commonLeafNameListPtr_ = leafNameListPtr;
}
void PhysicsEvent::setDataSetIndex(int dataSetIndex_) {
  _dataSetIndex_ = dataSetIndex_;
}
void PhysicsEvent::setEntryIndex(Long64_t entryIndex_) {
  _entryIndex_ = entryIndex_;
}
void PhysicsEvent::setTreeWeight(double treeWeight) {
  _treeWeight_ = treeWeight;
}
void PhysicsEvent::setNominalWeight(double nominalWeight) {
  _nominalWeight_ = nominalWeight;
}
void PhysicsEvent::setEventWeight(double eventWeight) {
  _eventWeight_ = eventWeight;
}

int PhysicsEvent::getDataSetIndex() const {
  return _dataSetIndex_;
}
Long64_t PhysicsEvent::getEntryIndex() const {
  return _entryIndex_;
}
double PhysicsEvent::getTreeWeight() const {
  return _treeWeight_;
}
double PhysicsEvent::getNominalWeight() const {
  return _nominalWeight_;
}
double PhysicsEvent::getEventWeight() const {
  return _eventWeight_;
}
std::map<FitParameterSet *, std::vector<Dial *>>* PhysicsEvent::getDialCachePtr() {
  return &_dialCache_;
}

void PhysicsEvent::hookToTree(TTree* tree_, bool throwIfLeafNotFound_){
  LogThrowIf(_commonLeafNameListPtr_ == nullptr, "_commonLeafNameListPtr_ is not set.");

  _leafContentList_.clear();

  if(throwIfLeafNotFound_){
    _leafContentList_.resize(_commonLeafNameListPtr_->size());
    for( size_t iLeaf = 0 ; iLeaf < _commonLeafNameListPtr_->size() ; iLeaf++ ){
      _leafContentList_.at(iLeaf).hookToTree(tree_, _commonLeafNameListPtr_->at(iLeaf));
    }
  }
  else{
    GenericToolbox::LeafHolder buf;
    for( size_t iLeaf = 0 ; iLeaf < _commonLeafNameListPtr_->size() ; iLeaf++ ){
      try{
        buf.hookToTree(tree_, _commonLeafNameListPtr_->at(iLeaf));
      }
      catch (...) {
        continue;
      }
      _leafContentList_.emplace_back(buf);
    }
  }

}

void PhysicsEvent::addEventWeight(double weight_){
  _eventWeight_ *= weight_;
}
void PhysicsEvent::resetEventWeight(){
  _eventWeight_ = _nominalWeight_;
}

int PhysicsEvent::findVarIndex(const std::string& leafName_, bool throwIfNotFound_) const{
  LogThrowIf(_commonLeafNameListPtr_ == nullptr, "Can't " << __METHOD_NAME__ << " while _commonLeafNameListPtr_ is empty.");
  for( size_t iLeaf = 0 ; iLeaf < _leafContentList_.size() ; iLeaf++ ){
    if( _commonLeafNameListPtr_->at(iLeaf) == leafName_ ){
      return int(iLeaf);
    }
  }
  if( throwIfNotFound_ ){
    LogThrow(leafName_ << " not found in: " << GenericToolbox::parseVectorAsString(*_commonLeafNameListPtr_));
  }
  return -1;
}
template<typename T> auto PhysicsEvent::fetchValue(const std::string &leafName_, size_t arrayIndex_) -> T {
  int index = this->findVarIndex(leafName_, true);
  return _leafContentList_.at(index).template getVariable<T>(arrayIndex_);
}
double PhysicsEvent::getVarAsDouble(const std::string& leafName_, size_t arrayIndex_) const{
  int index = this->findVarIndex(leafName_, true);
  return this->getVarAsDouble(index, arrayIndex_);
}
double PhysicsEvent::getVarAsDouble(int varIndex_, size_t arrayIndex_) const{
  return _leafContentList_.at(varIndex_).getVariableAsDouble(arrayIndex_);
}
double PhysicsEvent::evalFormula(TFormula* formulaPtr_, std::vector<int>* indexDict_) const{
  LogThrowIf(formulaPtr_ == nullptr, GET_VAR_NAME_VALUE(formulaPtr_));

  for( int iPar = 0 ; iPar < formulaPtr_->GetNpar() ; iPar++ ){
    if(indexDict_ == nullptr){ formulaPtr_->SetParameter(iPar, this->getVarAsDouble(formulaPtr_->GetParName(iPar))); }
    else                     { formulaPtr_->SetParameter(iPar, this->getVarAsDouble(indexDict_->at(iPar))); }
  }
  return formulaPtr_->Eval(0);
}

std::string PhysicsEvent::getSummary() const {
  std::stringstream ss;
  ss << typeid(this).name() << " :";
  if( _leafContentList_.empty() ){
    ss << "empty";
  }
  else{
    for( size_t iLeaf = 0 ; iLeaf < _leafContentList_.size() ; iLeaf++ ){
      ss << std::endl;
      if(_commonLeafNameListPtr_ != nullptr and _commonLeafNameListPtr_->size() == _leafContentList_.size()) {
        ss << _commonLeafNameListPtr_->at(iLeaf) << " -> ";
      }
      ss << _leafContentList_.at(iLeaf);
    }
  }
  ss << std::endl << GET_VAR_NAME_VALUE(_dataSetIndex_);
  ss << std::endl << GET_VAR_NAME_VALUE(_entryIndex_);
  ss << std::endl << GET_VAR_NAME_VALUE(_treeWeight_);
  ss << std::endl << GET_VAR_NAME_VALUE(_nominalWeight_);
  ss << std::endl << GET_VAR_NAME_VALUE(_eventWeight_);
  for( const auto& dialCachePair : _dialCache_ ){
    ss << std::endl << dialCachePair.first->getName() << ": " << GenericToolbox::parseVectorAsString(dialCachePair.second);
  }
  return ss.str();
}
void PhysicsEvent::print() const {
  LogInfo << *this << std::endl;
}

std::ostream& operator <<( std::ostream& o, const PhysicsEvent& p ){
  o << p.getSummary();
  return o;
}
