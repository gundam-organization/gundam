//
// Created by Nadrino on 22/07/2021.
//

#include "PhysicsEvent.h"
#include "SplineDial.h"

#include "GenericToolbox.Root.h"
#include "Logger.h"

#include <cmath>

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
  _sampleBinIndex_ = -1;
}

void PhysicsEvent::setCommonLeafNameListPtr(const std::shared_ptr<std::vector<std::string>>& commonLeafNameListPtr_){
  _commonLeafNameListPtr_ = commonLeafNameListPtr_;
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
#ifdef GUNDAM_USING_CUDA
    if (_CacheManagerValue_) {
        if (_CacheManagerValid_ && !(*_CacheManagerValid_)) {
            // This is slowish, but will make sure that the cached result is
            // updated when the cache has changed.  The values pointed to by
            // _CacheManagerValue_ and _CacheManagerValid_ are inside
            // of the weights cache (a bit of evil coding here), and are
            // updated by the cache.  The update is triggered by
            // _CacheManagerUpdate().
            if (_CacheManagerUpdate_) (*_CacheManagerUpdate_)();
        }
#ifdef CACHE_MANAGER_SLOW_VALIDATION
#warning CACHE_MANAGER_SLOW_VALIDATION used in PhysicsEvent::getEventWeight
        do {
            static double maxDelta = 1.0E-20;
            static double sumDelta = 0.0;
            static double sum2Delta = 0.0;
            static long long int numDelta = 0;
            double res = *_CacheManagerValue_;
            double avg = 0.5*(std::abs(res) + std::abs(_eventWeight_));
            if (avg < getTreeWeight()) avg = getTreeWeight();
            double delta = std::abs(res - _eventWeight_);
            delta /= avg;
            sumDelta += delta;
            sum2Delta += delta*delta;
            ++numDelta;
            if (numDelta < 0) throw std::runtime_error("validation wrap");
            maxDelta = std::max(maxDelta,delta);
            if ((numDelta % 1000000) == 0) {
                LogInfo << "VALIDATION: Average event weight delta: "
                        << sumDelta/numDelta
                        << " +/- " << std::sqrt(
                            sum2Delta/numDelta
                            - sumDelta*sumDelta/numDelta/numDelta)
                        << " Maximum: " << maxDelta
                        << std::endl;
            }
            if (maxDelta < 1E-5) break;
            if (delta < maxDelta) break;
            LogWarning << "WARNING: Event weight difference: " << delta
                       << " Cache: " << res
                       << " Dial: " << _eventWeight_
                       << " Tree: " << getTreeWeight()
                       << std::endl;
        } while(false);
#endif
#ifdef CACHE_MANAGER_SLOW_VALIDATION
#warning CACHE_MANAGER_SLOW_VALIDATION force CPU _eventWeight
        // When the slow validation is running, the "CPU" event weight is
        // calculated after Cache::Manager::Fill
        return _eventWeight_;
#endif
        return *_CacheManagerValue_;
    }
#endif
    return _eventWeight_;
}
int PhysicsEvent::getSampleBinIndex() const {
  return _sampleBinIndex_;
}
const std::vector<GenericToolbox::AnyType>& PhysicsEvent::getLeafHolder(const std::string &leafName_) const{
  int index = this->findVarIndex(leafName_, true);
  return this->getLeafHolder(index);
}
const std::vector<GenericToolbox::AnyType>& PhysicsEvent::getLeafHolder(int index_) const{
  return _leafContentList_[index_];
}
const std::vector<std::vector<GenericToolbox::AnyType>> &PhysicsEvent::getLeafContentList() const {
  return _leafContentList_;
}
std::vector<std::vector<GenericToolbox::AnyType>> &PhysicsEvent::getLeafContentList(){
  return _leafContentList_;
}
std::vector<Dial *> &PhysicsEvent::getRawDialPtrList() {
  return _rawDialPtrList_;
}
const std::vector<Dial *> &PhysicsEvent::getRawDialPtrList() const{
  return _rawDialPtrList_;
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

  // bare dials
  for( auto& dial : _rawDialPtrList_ ){
    if( dial == nullptr or dial->isMasked() ) return;
    this->addEventWeight( dial->evalResponse() );
#ifdef CACHE_MANAGER_SLOW_VALIDATION
    double response = dial->evalResponse();
#warning CACHE_MANAGER_SLOW_VALIDATION in PhysicsEvent::reweightUsingDialCache
    /////////////////////////////////////////////////////////////////
    // The internal GPU values for the splines are made available during slow
    // validation, but are never used in the CPU calculation.  This code here
    // is checking that the GPU value for the spline agrees with the direct
    // dial calculation of the spline.
    static std::map<std::string,double> maxDelta;
    static std::map<std::string,double> sumDelta;
    static std::map<std::string,double> sum2Delta;
    static std::map<std::string,long long int> numDelta;
    static int deltaTrials = 0;
    const SplineDial* sDial = dynamic_cast<const SplineDial*>(dial);
    while (sDial) {
        if (!dial->getCacheManagerValuePointer()) {
            LogWarning << "VALIDATION: SplineDial without cache" << std::endl;
            break;
        }
        // This only compiles with slow validation, and the cache validity
        // is managed elsewhere.
        if (!std::isfinite(response)) {
            LogWarning << "VALIDATION: Dial response is not finite" << std::endl;
            break;
        }
        if (response < 0) {
            LogWarning << "VALIDATION: Dial response is negative" << std::endl;
            break;
        }
        double cacheResponse = *dial->getCacheManagerValuePointer();
        std::string cacheName = dial->getCacheManagerName();
        std::string parName = dial->getOwner()->getParameterName();
        std::string sumName = parName;
        if (!std::isfinite(cacheResponse)) {
            LogError << "GPU cache response is not finite" << std::endl;
            std::runtime_error("GPU cache response is not finite");
        }
        if (cacheResponse < 0) {
            LogWarning << "VALIDATION: GPU cache response is negative" << std::endl;
        }
        double avg = 0.5*(std::abs(cacheResponse)+std::abs(response));
        if (avg < 1.0) avg = 1.0;
        double delta = std::abs(cacheResponse-response)/avg;
        sumDelta[sumName] += delta;
        sum2Delta[sumName] += delta*delta;
        ++numDelta[sumName];
        if (numDelta[sumName] < 0) {
            LogError << "Wrap around during validation" << std::endl;
            throw std::runtime_error("Validation wrap around");
        }
        if (delta > maxDelta[sumName]) {
            maxDelta[sumName] = delta;
            LogInfo << "VALIDATION: Increase GPU and Dial max delta"
                    << " GPU (" << cacheName << ") : " << cacheResponse
                    << " Dial: " << response
                    << " delta: " << delta
                    << " par: " << dial->getAssociatedParameter()
                    << " name: " << parName
                    << std::endl;
            LogInfo << "VALIDATION: Maximum Dial ("
                    << sumName << ") delta: "
                    << maxDelta[sumName]
                    << " Average value delta: "
                    << sumDelta[sumName]/numDelta[sumName]
                    << " +/- "
                    << std::sqrt(
                        sum2Delta[sumName]/numDelta[sumName]
                        - sumDelta[sumName]*sumDelta[sumName]
                        /numDelta[sumName]/numDelta[sumName])
                    << std::endl;
        }

        if ((deltaTrials++ % 1000000) == 0) {
            for (auto maxD : maxDelta) {
                std::string name = maxD.first;
                LogInfo << "VALIDATION: Average cache delta for"
                        << " " << name << ": "
                        << sumDelta[name]/numDelta[name]
                        << " +/- "
                        << std::sqrt(
                            sum2Delta[name]/numDelta[name]
                            - sumDelta[name]*sumDelta[name]
                            /numDelta[name]/numDelta[name])
                        << " Maximum: " << maxDelta[name]
                        << std::endl;
            }
        }

        break;
    }
#endif
  }

  // nested dials
//  for( auto& nestedDialEntry : _nestedDialRefList_ ){
//    if( nestedDialEntry.first == nullptr ) return;
//    this->addEventWeight( nestedDialEntry.first->eval(nestedDialEntry.second) );
//  }
}

int PhysicsEvent::findVarIndex(const std::string& leafName_, bool throwIfNotFound_) const{
  LogThrowIf(_commonLeafNameListPtr_ == nullptr, "Can't " << __METHOD_NAME__ << " while _commonLeafNameListPtr_ is empty.");
  for( size_t iLeaf = 0 ; iLeaf < _leafContentList_.size() ; iLeaf++ ){
    if( _commonLeafNameListPtr_->at(iLeaf) == leafName_ ){
      return int(iLeaf);
    }
  }
  if( throwIfNotFound_ ){
    LogWarning << leafName_ << " not found in:";
    for( auto& leaf : _leafContentList_  ){
      LogWarning << GenericToolbox::parseVectorAsString(leaf) << std::endl;
    }
    LogThrow(leafName_ << " not found in: " << GenericToolbox::parseVectorAsString(*_commonLeafNameListPtr_));
  }
  return -1;
}
void* PhysicsEvent::getVariableAddress(const std::string& leafName_, size_t arrayIndex_){
  int index = this->findVarIndex(leafName_, true);
  return _leafContentList_[index][arrayIndex_].getPlaceHolderPtr();
}
double PhysicsEvent::getVarAsDouble(const std::string& leafName_, size_t arrayIndex_) const{
  int index = this->findVarIndex(leafName_, true);
  return this->getVarAsDouble(index, arrayIndex_);
}
double PhysicsEvent::getVarAsDouble(int varIndex_, size_t arrayIndex_) const{
  return _leafContentList_[varIndex_][arrayIndex_].getValueAsDouble();
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

  ss << std::endl << GET_VAR_NAME_VALUE(_dataSetIndex_);
  ss << std::endl << GET_VAR_NAME_VALUE(_entryIndex_);
  ss << std::endl << GET_VAR_NAME_VALUE(_treeWeight_);
  ss << std::endl << GET_VAR_NAME_VALUE(_nominalWeight_);
  ss << std::endl << GET_VAR_NAME_VALUE(_eventWeight_);
  ss << std::endl << GET_VAR_NAME_VALUE(_sampleBinIndex_);

  if( _leafContentList_.empty() ){ ss << std::endl << "LeafContent: { empty }"; }
  else{
    ss << std::endl << "_leafContentList_ = { ";
    for( size_t iLeaf = 0 ; iLeaf < _leafContentList_.size() ; iLeaf++ ){
      ss << std::endl;
      if(_commonLeafNameListPtr_ != nullptr and _commonLeafNameListPtr_->size() == _leafContentList_.size()) {
        ss << "  " << _commonLeafNameListPtr_->at(iLeaf) << " -> ";
      }
      ss << GenericToolbox::parseVectorAsString(_leafContentList_[iLeaf]);
    }
    ss << std::endl << "}";
  }

  ss << std::endl << "_rawDialPtrList_: {";
  if( not _rawDialPtrList_.empty() ){
    for( auto* dialPtr : _rawDialPtrList_ ){
      ss << std::endl << "  ";
      if( dialPtr != nullptr ) ss << dialPtr->getSummary() << " = " << dialPtr->getDialResponseCache();
      else ss << "nullptr";
    }
    ss << std::endl << "}";
  }
  else{
    ss << "}";
  }
  ss << std::endl << "===========================";
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
void PhysicsEvent::trimDialCache(){
  size_t newSize{0};
  for( auto& dial : _rawDialPtrList_ ){
    if( dial == nullptr ) break;
    newSize++;
  }
  _rawDialPtrList_.resize(newSize);
  _rawDialPtrList_.shrink_to_fit();

  // nested dials
  newSize = 0;
  for( auto& nestedDial : _nestedDialRefList_ ){
    if( nestedDial.first == nullptr ) break;
    newSize++;
  }
  _nestedDialRefList_.resize(newSize);
  _nestedDialRefList_.shrink_to_fit();
}
void PhysicsEvent::addNestedDialRefToCache(NestedDialTest* nestedDialPtr_, const std::vector<Dial*>& dialPtrList_) {
  if (nestedDialPtr_ == nullptr) return; // don't store null ptr

  // fetch the next free slot:
  for (auto &nestedDialEntry: _nestedDialRefList_) {
    if (nestedDialEntry.first == nullptr) {
      nestedDialEntry.first = nestedDialPtr_;
      nestedDialEntry.second = dialPtrList_;
      return;
    }
  }

  // no new slot available:
  _nestedDialRefList_.emplace_back();
  _nestedDialRefList_.back().first = nestedDialPtr_;
  _nestedDialRefList_.back().second = dialPtrList_;
}
std::map<std::string, std::function<void(GenericToolbox::RawDataArray&, const std::vector<GenericToolbox::AnyType>&)>> PhysicsEvent::generateLeavesDictionary(bool disableArrays_) const{
  std::map<std::string, std::function<void(GenericToolbox::RawDataArray&, const std::vector<GenericToolbox::AnyType>&)>> out;

  for( auto& leafName : *_commonLeafNameListPtr_ ){

    const auto& lH = this->getLeafHolder(leafName);
    char typeTag = GenericToolbox::findOriginalVariableType(lH[0]);
    LogThrowIf( typeTag == 0 or typeTag == char(0xFF), leafName << " has an invalid leaf type." )

    std::string leafDefStr{leafName};
    if(not disableArrays_ and lH.size() > 1){ leafDefStr += "[" + std::to_string(lH.size()) + "]"; }
    leafDefStr += "/";
    leafDefStr += typeTag;
    if(not disableArrays_){
      out[leafDefStr] = [](GenericToolbox::RawDataArray& arr_, const std::vector<GenericToolbox::AnyType>& lH_){
        for(size_t iIndex = 0 ; iIndex < lH_.size() ; iIndex++){
          auto ph = lH_[iIndex].getPlaceHolderPtr();
          arr_.writeMemoryContent(ph->getVariableAddress(), ph->getVariableSize());
        }
      };
    }
    else{
      out[leafDefStr] = [](GenericToolbox::RawDataArray& arr_, const std::vector<GenericToolbox::AnyType>& lH_){
        auto ph = lH_[0].getPlaceHolderPtr();
        arr_.writeMemoryContent(ph->getVariableAddress(), ph->getVariableSize());
      };
    }

  }
  return out;
}
void PhysicsEvent::copyData(const std::vector<std::pair<const GenericToolbox::LeafHolder*, int>>& dict_, bool disableArrayStorage_){
  // Don't check for size? should be very fast
  for( int iLeaf = 0 ; iLeaf < dict_.size() ; iLeaf++ ){
    if(dict_[iLeaf].second == -1 and not disableArrayStorage_){ dict_[iLeaf].first->copyToAny(_leafContentList_[iLeaf]); }
    else{
      if( _leafContentList_[iLeaf].empty() ) _leafContentList_[iLeaf].emplace_back(GenericToolbox::leafToAnyType(dict_[iLeaf].first->getLeafTypeName()));
      (dict_[iLeaf].second==-1 and disableArrayStorage_) ?
      dict_[iLeaf].first->copyToAny(_leafContentList_[iLeaf][0], 0) :
      dict_[iLeaf].first->copyToAny(_leafContentList_[iLeaf][0], dict_[iLeaf].second);
    }
  }
}
std::vector<std::pair<const GenericToolbox::LeafHolder*, int>> PhysicsEvent::generateDict(const GenericToolbox::TreeEventBuffer& h_, const std::map<std::string, std::string>& leafDict_){
  std::vector<std::pair<const GenericToolbox::LeafHolder*, int>> out;
  out.reserve(_commonLeafNameListPtr_->size());
  std::string strBuf;
  for( const auto& leafName : (*_commonLeafNameListPtr_)){
    out.emplace_back(std::pair<GenericToolbox::LeafHolder*, int>{nullptr, -1});
    strBuf = leafName;
    if( GenericToolbox::doesKeyIsInMap(leafName, leafDict_) ){
      std::vector<std::string> argBuf;
      strBuf = leafDict_.at(leafName);
      strBuf = GenericToolbox::stripBracket(strBuf, '[', ']', false, &argBuf);
      if     ( argBuf.size() == 1 ){ out.back().second = std::stoi(argBuf[0]); }
      else if( argBuf.size() > 1 ){ LogThrow("No support for multi-dim array."); }
    }
    out.back().first = &h_.getLeafContent(strBuf);
  }
  return out;
}
void PhysicsEvent::copyLeafContent(const PhysicsEvent& ref_){
  LogThrowIf(ref_.getCommonLeafNameListPtr() != _commonLeafNameListPtr_, "source event don't have the same leaf name list")
  _leafContentList_ = ref_.getLeafContentList();
//  for( size_t iLeaf = 0 ; iLeaf < _commonLeafNameListPtr_->size() ; iLeaf++ ){
//    _leafContentList_[iLeaf] = ref_.getLeafContentList()[iLeaf];
//  }
}

std::ostream& operator <<( std::ostream& o, const PhysicsEvent& p ){
  o << p.getSummary();
  return o;
}

const std::shared_ptr<std::vector<std::string>>& PhysicsEvent::getCommonLeafNameListPtr() const {
  return _commonLeafNameListPtr_;
}
