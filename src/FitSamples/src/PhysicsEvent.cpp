//
// Created by Nadrino on 22/07/2021.
//

#include "PhysicsEvent.h"

#include "GenericToolbox.Root.h"
#include "Logger.h"

#include <cmath>

LoggerInit([]{
  Logger::setUserHeaderStr("[PhysicsEvent]");
});

PhysicsEvent::PhysicsEvent() = default;
PhysicsEvent::~PhysicsEvent()  = default;

void PhysicsEvent::setCommonVarNameListPtr(const std::shared_ptr<std::vector<std::string>>& commonVarNameListPtr_){
  _commonVarNameListPtr_ = commonVarNameListPtr_;
  _varHolderList_.resize(_commonVarNameListPtr_->size());
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
void PhysicsEvent::setSampleIndex(int sampleIndex) {
  _sampleIndex_ = sampleIndex;
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
#ifdef GUNDAM_USING_CACHE_MANAGER
    if (_cacheManagerValue_) {
        if (_cacheManagerValid_ != nullptr and not (*_cacheManagerValid_)) {
            // This is slowish, but will make sure that the cached result is
            // updated when the cache has changed.  The values pointed to by
            // _CacheManagerValue_ and _CacheManagerValid_ are inside
            // of the weights cache (a bit of evil coding here), and are
            // updated by the cache.  The update is triggered by
            // (*_CacheManagerUpdate_)().
            if (_cacheManagerUpdate_) (*_cacheManagerUpdate_)();
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
            maxDelta = std::max(maxDelta,delta);
            if (delta < 1e-4) {
                sumDelta += delta;
                sum2Delta += delta*delta;
                ++numDelta;
                if (numDelta < 0) throw std::runtime_error("validation wrap");
                if ((numDelta % 1000000) == 0) {
                    LogInfo << "VALIDATION: Average event weight delta: "
                            << sumDelta/numDelta
                            << " +/- " << std::sqrt(
                                sum2Delta/numDelta
                                - sumDelta*sumDelta/numDelta/numDelta)
                            << " Maximum: " << maxDelta
                            << " " << numDelta
                            << std::endl;
                }
            }
            if (maxDelta < 1E-5) break;
            if (delta > 100.0*sumDelta/numDelta) break;
            LogWarning << "WARNING: Event weight difference: " << delta
                       << " Cache: " << res
                       << " Dial: " << _eventWeight_
                       << " Tree: " << getTreeWeight()
                       << " Delta: " << delta
                       << " Max: " << maxDelta
                       << std::endl;
        } while(false);
#endif
#ifdef CACHE_MANAGER_SLOW_VALIDATION
#warning CACHE_MANAGER_SLOW_VALIDATION force CPU _eventWeight_
        // When the slow validation is running, the "CPU" event weight is
        // calculated after Cache::Manager::Fill
        return _eventWeight_;
#endif
        LogThrowIf(not std::isfinite(*_cacheManagerValue_), "NaN weight: " << this->getSummary());
      return *_cacheManagerValue_;
    }
#endif
    return _eventWeight_;
}
int PhysicsEvent::getSampleBinIndex() const {
  return _sampleBinIndex_;
}
int PhysicsEvent::getSampleIndex() const {
  return _sampleIndex_;
}
const std::vector<GenericToolbox::AnyType>& PhysicsEvent::getLeafHolder(const std::string &leafName_) const{
  int index = this->findVarIndex(leafName_, true);
  return this->getLeafHolder(index);
}
const std::vector<GenericToolbox::AnyType>& PhysicsEvent::getLeafHolder(int index_) const{
  return _varHolderList_[index_];
}
const std::vector<std::vector<GenericToolbox::AnyType>> &PhysicsEvent::getLeafContentList() const {
  return _varHolderList_;
}
std::vector<std::vector<GenericToolbox::AnyType>> &PhysicsEvent::getLeafContentList(){
  return _varHolderList_;
}

void PhysicsEvent::copyOnlyExistingLeaves(const PhysicsEvent& other_){
  LogThrowIf(_commonVarNameListPtr_ == nullptr, "_commonLeafNameListPtr_ not set")
  for(size_t iLeaf = 0 ; iLeaf < _commonVarNameListPtr_->size() ; iLeaf++ ){
    _varHolderList_[iLeaf] = other_.getLeafHolder((*_commonVarNameListPtr_)[iLeaf]);
  }
}

void PhysicsEvent::addEventWeight(double weight_){
  _eventWeight_ *= weight_;
}
void PhysicsEvent::resetEventWeight(){
  _eventWeight_ = _treeWeight_;
}

int PhysicsEvent::findVarIndex(const std::string& leafName_, bool throwIfNotFound_) const{
  LogThrowIf(_commonVarNameListPtr_ == nullptr, "Can't " << __METHOD_NAME__ << " while _commonLeafNameListPtr_ is empty.");
  for(size_t iLeaf = 0 ; iLeaf < _varHolderList_.size() ; iLeaf++ ){
    if(_commonVarNameListPtr_->at(iLeaf) == leafName_ ){
      return int(iLeaf);
    }
  }
  if( throwIfNotFound_ ){
    LogWarning << leafName_ << " not found in:";
    for( auto& leaf : _varHolderList_  ){
      LogWarning << GenericToolbox::parseVectorAsString(leaf) << std::endl;
    }
    LogThrow(leafName_ << " not found in: " << GenericToolbox::parseVectorAsString(*_commonVarNameListPtr_));
  }
  return -1;
}
const GenericToolbox::AnyType& PhysicsEvent::getVariableAsAnyType(const std::string& leafName_, size_t arrayIndex_) const{
  int index = this->findVarIndex(leafName_, true);
  return _varHolderList_[index][arrayIndex_];
}
GenericToolbox::AnyType& PhysicsEvent::getVariableAsAnyType(const std::string& leafName_, size_t arrayIndex_){
  return const_cast<GenericToolbox::AnyType&>(const_cast<const PhysicsEvent*>(this)->getVariableAsAnyType(leafName_, arrayIndex_));
}
void* PhysicsEvent::getVariableAddress(const std::string& leafName_, size_t arrayIndex_){
  return this->getVariableAsAnyType(leafName_, arrayIndex_).getPlaceHolderPtr()->getVariableAddress();
}
double PhysicsEvent::getVarAsDouble(const std::string& leafName_, size_t arrayIndex_) const{
  int index = this->findVarIndex(leafName_, true);
  return this->getVarAsDouble(index, arrayIndex_);
}
double PhysicsEvent::getVarAsDouble(int varIndex_, size_t arrayIndex_) const{
  if( _varToDoubleCache_.empty() ) return _varHolderList_[varIndex_][arrayIndex_].getValueAsDouble();
  else{
    // if using double cache:
    if( std::isnan(_varToDoubleCache_[varIndex_][arrayIndex_]) ){
      return _varToDoubleCache_[varIndex_][arrayIndex_] = _varHolderList_[varIndex_][arrayIndex_].getValueAsDouble();
    }
    return _varToDoubleCache_[varIndex_][arrayIndex_];
  }
}
const GenericToolbox::AnyType& PhysicsEvent::getVar(int varIndex_, size_t arrayIndex_) const{
  return _varHolderList_[varIndex_][arrayIndex_];
}
void PhysicsEvent::fillBuffer(const std::vector<int>& indexList_, std::vector<double>& buffer_) const{
  buffer_.resize(indexList_.size()); double* slot = &buffer_[0];
  std::for_each(indexList_.begin(), indexList_.end(), [&](auto& index){ *(slot++) = this->getVarAsDouble(index); });
}


double PhysicsEvent::evalFormula(const TFormula* formulaPtr_, std::vector<int>* indexDict_) const{
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

  ss << GET_VAR_NAME_VALUE(_dataSetIndex_);
  ss << std::endl << GET_VAR_NAME_VALUE(_entryIndex_);
  ss << std::endl << GET_VAR_NAME_VALUE(_treeWeight_);
  ss << std::endl << GET_VAR_NAME_VALUE(_nominalWeight_);
  ss << std::endl << GET_VAR_NAME_VALUE(_eventWeight_);
  ss << std::endl << GET_VAR_NAME_VALUE(_sampleBinIndex_);

  if( _varHolderList_.empty() ){ ss << std::endl << "LeafContent: { empty }"; }
  else{
    ss << std::endl << "_leafContentList_ = { ";
    for(size_t iLeaf = 0 ; iLeaf < _varHolderList_.size() ; iLeaf++ ){
      ss << std::endl;
      if(_commonVarNameListPtr_ != nullptr and _commonVarNameListPtr_->size() == _varHolderList_.size()) {
        ss << "  " << _commonVarNameListPtr_->at(iLeaf) << " -> ";
      }
      ss << GenericToolbox::parseVectorAsString(_varHolderList_[iLeaf]);
    }
    ss << std::endl << "}";
  }

  return ss.str();
}
void PhysicsEvent::print() const {
  LogInfo << *this << std::endl;
}

std::map< std::string,
  std::function<void(GenericToolbox::RawDataArray&, const std::vector<GenericToolbox::AnyType>&)>> PhysicsEvent::generateLeavesDictionary(bool disableArrays_) const {
  std::map<std::string, std::function<void(GenericToolbox::RawDataArray&, const std::vector<GenericToolbox::AnyType>&)>> out;
  if(_commonVarNameListPtr_ != nullptr ){
    for( auto& leafName : *_commonVarNameListPtr_ ){

      const auto& lH = this->getLeafHolder(leafName);
      char typeTag = GenericToolbox::findOriginalVariableType(lH[0]);
      LogThrowIf( typeTag == 0 or typeTag == char(0xFF), leafName << " has an invalid leaf type." )

      std::string leafDefStr{leafName};
      if(not disableArrays_ and lH.size() > 1){ leafDefStr += "[" + std::to_string(lH.size()) + "]"; }
      leafDefStr += "/";
      leafDefStr += typeTag;
      if(not disableArrays_){
        out[leafDefStr] = [](GenericToolbox::RawDataArray& arr_, const std::vector<GenericToolbox::AnyType>& variablesList_){
          for(const auto & variable : variablesList_){
            arr_.writeMemoryContent(
                variable.getPlaceHolderPtr()->getVariableAddress(),
                variable.getPlaceHolderPtr()->getVariableSize()
            );
          }
        };
      }
      else{
        out[leafDefStr] = [](GenericToolbox::RawDataArray& arr_, const std::vector<GenericToolbox::AnyType>& variablesList_){
          arr_.writeMemoryContent(
              variablesList_[0].getPlaceHolderPtr()->getVariableAddress(),
              variablesList_[0].getPlaceHolderPtr()->getVariableSize()
          );
        };
      }

    }
  }
  return out;
}


void PhysicsEvent::allocateMemory(const std::vector<const GenericToolbox::LeafForm*>& leafFormList_){
  LogThrowIf( _commonVarNameListPtr_ == nullptr, "var name list not set." );
  LogThrowIf( _commonVarNameListPtr_->size() != leafFormList_.size(), "size mismatch." );

  auto nLeaf{_commonVarNameListPtr_->size()};
  for(size_t iVar = 0 ; iVar < nLeaf ; iVar++ ){
    _varHolderList_[iVar].emplace_back(
        GenericToolbox::leafToAnyType( leafFormList_[iVar]->getLeafTypeName() )
    );
  }
  this->resizeVarToDoubleCache();
}
void PhysicsEvent::copyData(const std::vector<const GenericToolbox::LeafForm*>& leafFormList_){
  // Don't check for the size? it has to be very fast
  size_t nLeaf{leafFormList_.size()};
  for( size_t iLeaf = 0 ; iLeaf < nLeaf ; iLeaf++ ){
    leafFormList_[iLeaf]->dropToAny(_varHolderList_[iLeaf][0]);
  }
  this->invalidateVarToDoubleCache();
}
void PhysicsEvent::copyData(const std::vector<std::pair<const GenericToolbox::LeafHolder *, int>> &dict_) {
  // Don't check for size? it has to be very fast
  size_t nLeaf{dict_.size()};
  for( size_t iLeaf = 0 ; iLeaf < nLeaf ; iLeaf++ ){
    if( _varHolderList_[iLeaf].empty() ){
      // resize buffer
      _varHolderList_[iLeaf].emplace_back(GenericToolbox::leafToAnyType(dict_[iLeaf].first->getLeafTypeName()) );
    }

    if( dict_[iLeaf].second==-1 ){
      // no index specified -> drop only the first element
      dict_[iLeaf].first->dropToAny(_varHolderList_[iLeaf][0], 0);
    }
    else{
      // drop the selected index
      dict_[iLeaf].first->dropToAny(_varHolderList_[iLeaf][0], dict_[iLeaf].second);
    }
  }
  this->invalidateVarToDoubleCache();
}
std::vector<std::pair<const GenericToolbox::LeafHolder*, int>> PhysicsEvent::generateDict(
    const GenericToolbox::TreeEntryBuffer& treeEventBuffer_,
    const std::map<std::string, std::string>& leafDict_
    ){
  std::vector<std::pair<const GenericToolbox::LeafHolder*, int>> out;
  out.reserve(_commonVarNameListPtr_->size());
  std::string strBuf;
  for( const auto& leafName : (*_commonVarNameListPtr_)){
    out.emplace_back(std::pair<GenericToolbox::LeafHolder*, int>{nullptr, -1});
    strBuf = leafName;
    if( GenericToolbox::doesKeyIsInMap(leafName, leafDict_) ){
      std::vector<std::string> argBuf;
      strBuf = leafDict_.at(leafName);
      LogDebug << strBuf << " -> ";
      strBuf = GenericToolbox::stripBracket(strBuf, '[', ']', false, &argBuf);

      LogDebug << strBuf << "/" << GenericToolbox::parseVectorAsString(argBuf) << std::endl;
      if     ( argBuf.size() == 1 ){ out.back().second = std::stoi(argBuf[0]); }
      else if( argBuf.size() > 1 ){ LogThrow("No support for multi-dim array."); }
    }
    out.back().first = &treeEventBuffer_.getLeafContent(strBuf);
  }
  return out;
}
void PhysicsEvent::copyLeafContent(const PhysicsEvent& ref_){
  LogThrowIf(ref_.getCommonLeafNameListPtr() != _commonVarNameListPtr_, "source event don't have the same leaf name list")
  _varHolderList_ = ref_.getLeafContentList();
//  for( size_t iLeaf = 0 ; iLeaf < _commonLeafNameListPtr_->size() ; iLeaf++ ){
//    _leafContentList_[iLeaf] = ref_.getLeafContentList()[iLeaf];
//  }
}
void PhysicsEvent::resizeVarToDoubleCache(){
  _varToDoubleCache_.reserve(_varHolderList_.size());
  for( auto& leaf: _varHolderList_ ){
    _varToDoubleCache_.emplace_back(leaf.size(), std::nan("unset"));
  }
  this->invalidateVarToDoubleCache();
}
void PhysicsEvent::invalidateVarToDoubleCache(){
  std::for_each(_varToDoubleCache_.begin(), _varToDoubleCache_.end(),
                [](auto& varArray)
                { std::for_each( varArray.begin(), varArray.end(), [](auto& var){ var = std::nan("unset"); }); });
}

std::ostream& operator <<( std::ostream& o, const PhysicsEvent& p ){
  o << p.getSummary();
  return o;
}

const std::shared_ptr<std::vector<std::string>>& PhysicsEvent::getCommonLeafNameListPtr() const {
  return _commonVarNameListPtr_;
}

//  A Lesser GNU Public License

//  Copyright (C) 2023 GUNDAM DEVELOPERS

//  This library is free software; you can redistribute it and/or
//  modify it under the terms of the GNU Lesser General Public
//  License as published by the Free Software Foundation; either
//  version 2.1 of the License, or (at your option) any later version.

//  This library is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  Lesser General Public License for more details.

//  You should have received a copy of the GNU Lesser General Public
//  License along with this library; if not, write to the
//
//  Free Software Foundation, Inc.
//  51 Franklin Street, Fifth Floor,
//  Boston, MA  02110-1301  USA

// Local Variables:
// mode:c++
// c-basic-offset:2
// compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
// End:
