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
  _rawDialPtrList_.clear();

  // Weight carriers
  _dataSetIndex_=-1;
  _entryIndex_=-1;
  _treeWeight_ = 1;
  _nominalWeight_ = 1;
  _eventWeight_ = 1;
  _fakeDataWeight_ = 1;
  _sampleBinIndex_ = -1;
}

void PhysicsEvent::setLeafNameListPtr(const std::vector<std::string> *leafNameListPtr) {
  _commonLeafNameListPtr_ = leafNameListPtr;
  _leafContentList_.resize(_commonLeafNameListPtr_->size());
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
void PhysicsEvent::setFakeDataWeight(double fakeDataWeight) {
  _fakeDataWeight_ = fakeDataWeight;
}
void PhysicsEvent::setSampleBinIndex(int sampleBinIndex) {
  _sampleBinIndex_ = sampleBinIndex;
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
double PhysicsEvent::getFakeDataWeight() const {
  return _fakeDataWeight_;
}
int PhysicsEvent::getSampleBinIndex() const {
  return _sampleBinIndex_;
}
const GenericToolbox::LeafHolder& PhysicsEvent::getLeafHolder(std::string leafName_) const{
  int index = this->findVarIndex(leafName_, true);
  return this->getLeafHolder(index);
}
const GenericToolbox::LeafHolder& PhysicsEvent::getLeafHolder(int index_) const{
  return _leafContentList_[index_];
}
const std::vector<GenericToolbox::LeafHolder> &PhysicsEvent::getLeafContentList() const {
  return _leafContentList_;
}
std::vector<Dial *> &PhysicsEvent::getRawDialPtrList() {
  return _rawDialPtrList_;
}
const std::vector<Dial *> &PhysicsEvent::getRawDialPtrList() const{
  return _rawDialPtrList_;
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
        LogWarning << this->getSummary() << std::endl;
        continue;
      }
      _leafContentList_.emplace_back(buf);
    }
  }

}
void PhysicsEvent::clonePointerLeaves(){
  for( auto& leaf : _leafContentList_ ){
    leaf.clonePointerLeaves();
  }
}
void PhysicsEvent::copyOnlyExistingLeaves(const PhysicsEvent& other_){
  LogThrowIf(_commonLeafNameListPtr_ == nullptr, "_commonLeafNameListPtr_ not set")
  for( size_t iLeaf = 0 ; iLeaf < _commonLeafNameListPtr_->size() ; iLeaf++ ){
    _leafContentList_[iLeaf] = other_.getLeafHolder((*_commonLeafNameListPtr_)[iLeaf]);
  }
}

void PhysicsEvent::addEventWeight(double weight_){
  _eventWeight_ *= weight_;
}
void PhysicsEvent::resetEventWeight(){
  _eventWeight_ = _treeWeight_;
}
void PhysicsEvent::reweightUsingDialCache(){
  this->resetEventWeight();
  for( auto& dial : _rawDialPtrList_ ){
    if( dial == nullptr ) return;
    this->addEventWeight( dial->evalResponse() );
  }
}

int PhysicsEvent::findVarIndex(const std::string& leafName_, bool throwIfNotFound_) const{
  LogThrowIf(_commonLeafNameListPtr_ == nullptr, "Can't " << __METHOD_NAME__ << " while _commonLeafNameListPtr_ is empty.");
  for( size_t iLeaf = 0 ; iLeaf < _leafContentList_.size() ; iLeaf++ ){
    if( _commonLeafNameListPtr_->at(iLeaf) == leafName_ ){
      return int(iLeaf);
    }
  }
  if( throwIfNotFound_ ){
    LogWarning << leafName_ << " not found in: " << GenericToolbox::parseVectorAsString(_leafContentList_) << std::endl;
    LogThrow(leafName_ << " not found in: " << GenericToolbox::parseVectorAsString(*_commonLeafNameListPtr_));
  }
  return -1;
}
void* PhysicsEvent::getVariableAddress(const std::string& leafName_, size_t arrayIndex_){
  int index = this->findVarIndex(leafName_, true);
  return _leafContentList_.at(index).getVariableAddress(arrayIndex_);
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

  std::vector<double> parArray(formulaPtr_->GetNpar());
  for( int iPar = 0 ; iPar < formulaPtr_->GetNpar() ; iPar++ ){
    if(indexDict_ == nullptr){ parArray[iPar] = this->getVarAsDouble(formulaPtr_->GetParName(iPar)); }
    else                     { parArray[iPar] = this->getVarAsDouble(indexDict_->at(iPar)); }
  }

  return formulaPtr_->EvalPar(nullptr, &parArray[0]);
}

std::string PhysicsEvent::getSummary() const {
  std::stringstream ss;
  ss << "PhysicsEvent :";
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
  ss << std::endl << GET_VAR_NAME_VALUE(_fakeDataWeight_);
  ss << std::endl << GET_VAR_NAME_VALUE(_sampleBinIndex_);

  if( not _rawDialPtrList_.empty() ){
    ss << std::endl << "List of dials: ";
    for( auto* dialPtr : _rawDialPtrList_ ){
      ss << std::endl << "- ";
      ss << dialPtr->getSummary() << " = " << dialPtr->getDialResponseCache();
    }
  }
  else{
    ss << "No cached dials." << std::endl;
  }
  return ss.str();
}
void PhysicsEvent::print() const {
  LogInfo << *this << std::endl;
}
bool PhysicsEvent::isSame(AnaEvent& anaEvent_) const{

  bool isSame = true;
  for( const auto& varName : *_commonLeafNameListPtr_ ){
    int anaIndex = anaEvent_.GetGlobalIndex(varName);
    if( anaIndex == -1 ) continue;
    if(this->getVarAsDouble(varName) != anaEvent_.GetEventVarAsDouble(varName) ){
      isSame = false;
      LogError << GET_VAR_NAME_VALUE(varName) << std::endl;
      break;
    }
  }

  if( _sampleBinIndex_ != anaEvent_.GetRecoBinIndex() ){
    isSame = false;
  }

  if(not isSame){
    this->print();
    anaEvent_.Print();
  }

  return isSame;
}
void PhysicsEvent::deleteLeaf(long index_){
  // UNTESTED
  _leafContentList_.erase(_leafContentList_.begin() + index_);
  _leafContentList_.shrink_to_fit();
}
void PhysicsEvent::trimDialCache(){
  size_t newSize{0};
  for( auto& dial : _rawDialPtrList_ ){
    if( dial == nullptr ) break;
    newSize++;
  }
  _rawDialPtrList_.resize(newSize);
  _rawDialPtrList_.shrink_to_fit();
}
void PhysicsEvent::addDialRefToCache(Dial* dialPtr_){
  if( dialPtr_ == nullptr ) return; // don't store null ptr

  // fetch the next free slot:
  for( auto& dial : _rawDialPtrList_ ){
    if( dial == nullptr ){
      dial = dialPtr_;
      return;
    }
  }

  // no new slot available:
  _rawDialPtrList_.emplace_back(dialPtr_);
}
std::map<std::string, std::function<void(GenericToolbox::RawDataArray&, const GenericToolbox::LeafHolder&)>> PhysicsEvent::generateLeavesDictionary(bool disableArrays_) const{
  std::map<std::string, std::function<void(GenericToolbox::RawDataArray&, const GenericToolbox::LeafHolder&)>> out;

  for( auto& leafName : *_commonLeafNameListPtr_ ){

    const auto& lH = this->getLeafHolder(leafName);
    char typeTag = lH.findOriginalVariableType();
    LogThrowIf( typeTag == 0 or typeTag == char(0xFF), leafName << " has an invalid leaf type." )

    std::string leafDefStr{leafName};
    if(not disableArrays_ and lH.getArraySize() > 1){ leafDefStr += "[" + std::to_string(lH.getArraySize()) + "]"; }
    leafDefStr += "/";
    leafDefStr += typeTag;
    if(not disableArrays_){
      out[leafDefStr] = [](GenericToolbox::RawDataArray& arr_, const GenericToolbox::LeafHolder& lH_){
        for(size_t iIndex = 0 ; iIndex < lH_.getArraySize() ; iIndex++){
          arr_.writeMemoryContent(lH_.getLeafDataAddress(iIndex).getPlaceHolderPtr()->getVariableAddress(), lH_.getVariableSize(iIndex));
        }
      };
    }
    else{
      out[leafDefStr] = [](GenericToolbox::RawDataArray& arr_, const GenericToolbox::LeafHolder& lH_){
        arr_.writeMemoryContent(lH_.getLeafDataAddress().getPlaceHolderPtr()->getVariableAddress(), lH_.getVariableSize());
      };
    }

  }
  return out;
}

std::ostream& operator <<( std::ostream& o, const PhysicsEvent& p ){
  o << p.getSummary();
  return o;
}

const std::vector<std::string> *PhysicsEvent::getCommonLeafNameListPtr() const {
  return _commonLeafNameListPtr_;
}


