//
// Created by Nadrino on 22/07/2021.
//

#ifndef GUNDAM_PHYSICSEVENT_H
#define GUNDAM_PHYSICSEVENT_H

#include "FitParameterSet.h"

#include "GenericToolbox.Root.TreeEntryBuffer.h"
#include "GenericToolbox.Root.LeafCollection.h"
#include "GenericToolbox.AnyType.h"

#include "TTree.h"
#include "TFormula.h"

#include <map>
#include <mutex>
#include <vector>
#include <string>

class PhysicsEvent {

public:
  PhysicsEvent() = default;
  virtual ~PhysicsEvent() = default;

  // setters
  void setSampleIndex(int sampleIndex){ _sampleIndex_ = sampleIndex; }
  void setDataSetIndex(int dataSetIndex_){ _dataSetIndex_ = dataSetIndex_; }
  void setSampleBinIndex(int sampleBinIndex){ _sampleBinIndex_ = sampleBinIndex; }
  void setEntryIndex(Long64_t entryIndex_){ _entryIndex_ = entryIndex_; }
  void setTreeWeight(double treeWeight){ _treeWeight_ = treeWeight; }
  void setEventWeight(double eventWeight){ _eventWeight_ = eventWeight; }
  void setNominalWeight(double nominalWeight){ _nominalWeight_ = nominalWeight; }
  void setCommonVarNameListPtr(const std::shared_ptr<std::vector<std::string>>& commonVarNameListPtr_);
  template<typename T> void setVariable(const T& value_, const std::string& leafName_, size_t arrayIndex_ = 0);

  // const getters
  int getSampleIndex() const{ return _sampleIndex_; }
  int getDataSetIndex() const { return _dataSetIndex_; }
  int getSampleBinIndex() const{ return _sampleBinIndex_; }
  Long64_t getEntryIndex() const { return _entryIndex_; }
  double getTreeWeight() const { return _treeWeight_; }
  double getNominalWeight() const { return _nominalWeight_; }
  double getEventWeight() const;
  const GenericToolbox::AnyType& getVar(int varIndex_, size_t arrayIndex_ = 0) const { return _varHolderList_[varIndex_][arrayIndex_]; }
  const std::vector<GenericToolbox::AnyType>& getVarHolder(int index_) const { return _varHolderList_[index_]; }
  const std::vector<GenericToolbox::AnyType>& getVarHolder(const std::string &leafName_) const;
  const std::vector<std::vector<GenericToolbox::AnyType>> &getVarHolderList() const { return _varHolderList_; }
  const std::shared_ptr<std::vector<std::string>>& getCommonVarNameListPtr() const { return _commonVarNameListPtr_; }
  const GenericToolbox::AnyType& getVariableAsAnyType(const std::string& leafName_, size_t arrayIndex_ = 0) const;
  template<typename T> auto getVarValue(const std::string& leafName_, size_t arrayIndex_ = 0) const -> T;
  template<typename T> auto getVariable(const std::string& leafName_, size_t arrayIndex_ = 0) const -> const T&;

  // non-const getters
  double& getEventWeightRef(){ return _eventWeight_; }
  void* getVariableAddress(const std::string& leafName_, size_t arrayIndex_ = 0);
  std::vector<std::vector<GenericToolbox::AnyType>> &getVarHolderList(){ return _varHolderList_; }
  GenericToolbox::AnyType& getVariableAsAnyType(const std::string& leafName_, size_t arrayIndex_ = 0);

  // core
  void resetEventWeight(){ _eventWeight_ = _treeWeight_; }
  void resizeVarToDoubleCache();
  void invalidateVarToDoubleCache();
  void copyData(const std::vector<const GenericToolbox::LeafForm*>& leafFormList_);
  void allocateMemory(const std::vector<const GenericToolbox::LeafForm*>& leafFormList_);
  int findVarIndex(const std::string& leafName_, bool throwIfNotFound_ = true) const;
  double getVarAsDouble(int varIndex_, size_t arrayIndex_ = 0) const;
  double getVarAsDouble(const std::string& leafName_, size_t arrayIndex_ = 0) const;
  double evalFormula(const TFormula* formulaPtr_, std::vector<int>* indexDict_ = nullptr) const;

  // misc
  void print() const;
  std::string getSummary() const;
  void copyVarHolderList(const PhysicsEvent& ref_);
  void copyOnlyExistingVarHolders(const PhysicsEvent& other_);
  void fillBuffer(const std::vector<int>& indexList_, std::vector<double>& buffer_) const;

  // operators
  friend std::ostream& operator <<( std::ostream& o, const PhysicsEvent& p );

private:
  // Context variables
  int _sampleIndex_{-1}; // this information is lost in the EventDialCache manager
  int _dataSetIndex_{-1};
  int _sampleBinIndex_{-1};
  Long64_t _entryIndex_{-1};
  double _treeWeight_{1};
  double _nominalWeight_{1};
  double _eventWeight_{1};

  // Data storage variables
  std::shared_ptr<std::vector<std::string>> _commonVarNameListPtr_{nullptr};
  std::vector<std::vector<GenericToolbox::AnyType>> _varHolderList_{};

  // Cache variables
  mutable std::vector<std::vector<double>> _varToDoubleCache_{};

#ifdef GUNDAM_USING_CACHE_MANAGER
public:
  void setCacheManagerIndex(int i) { _cacheManagerIndex_ = i;}
  void setCacheManagerValuePointer(const double* v) { _cacheManagerValue_ = v;}
  void setCacheManagerValidPointer(const bool* v) { _cacheManagerValid_ = v;}
  void setCacheManagerUpdatePointer(void (*p)()) { _cacheManagerUpdate_ = p;}

  int getCacheManagerIndex() const {return _cacheManagerIndex_;}
private:
  // An "opaque" index into the cache that is used to simplify bookkeeping.
  int _cacheManagerIndex_{-1};
  // A pointer to the cached result.
  const double* _cacheManagerValue_{nullptr};
  // A pointer to the cache validity flag.
  const bool* _cacheManagerValid_{nullptr};
  // A pointer to a callback to force the cache to be updated.
  void (*_cacheManagerUpdate_)(){nullptr};
#endif

};


// TEMPLATES IMPLEMENTATION
template<typename T> auto PhysicsEvent::getVarValue(const std::string &leafName_, size_t arrayIndex_) const -> T {
  return this->getVariable<T>(leafName_, arrayIndex_);
}
template<typename T> auto PhysicsEvent::getVariable(const std::string& leafName_, size_t arrayIndex_) const -> const T&{
  return this->getVariableAsAnyType(leafName_, arrayIndex_).template getValue<T>();
}
template<typename T> void PhysicsEvent::setVariable(const T& value_, const std::string& leafName_, size_t arrayIndex_){
  int index = this->findVarIndex(leafName_, true);
  _varHolderList_[index][arrayIndex_].template getValue<T>() = value_;
  if( not _varToDoubleCache_.empty() ){ _varToDoubleCache_[index][arrayIndex_] = std::nan("unset"); }
}


#endif //GUNDAM_PHYSICSEVENT_H
