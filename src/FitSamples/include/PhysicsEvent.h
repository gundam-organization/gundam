//
// Created by Nadrino on 22/07/2021.
//

#ifndef GUNDAM_PHYSICSEVENT_H
#define GUNDAM_PHYSICSEVENT_H

#include "FitParameterSet.h"
#include "Dial.h"
#include "NestedDialTest.h"

#include "GenericToolbox.Root.TreeEventBuffer.h"
#include "GenericToolbox.Root.LeafHolder.h"
#include <GenericToolbox.RawDataArray.h>
#include "GenericToolbox.AnyType.h"

#include "TTree.h"
#include "TFormula.h"

#include "vector"
#include "string"
#include "map"

class PhysicsEvent {

public:
  PhysicsEvent();
  virtual ~PhysicsEvent();

  void reset();

  // SETTERS
  void setDataSetIndex(int dataSetIndex_);
  void setEntryIndex(Long64_t entryIndex_);
  void setTreeWeight(double treeWeight);
  void setNominalWeight(double nominalWeight);
  void setEventWeight(double eventWeight);
  void setFakeDataWeight(double fakeDataWeight);
  void setSampleBinIndex(int sampleBinIndex);
  void setCommonLeafNameListPtr(const std::shared_ptr<std::vector<std::string>>& commonLeafNameListPtr_);

  // GETTERS
  int getDataSetIndex() const;
  Long64_t getEntryIndex() const;
  double getTreeWeight() const;
  double getNominalWeight() const;
  double getEventWeight() const;
  double getFakeDataWeight() const;
  int getSampleBinIndex() const;
  std::vector<Dial *> &getRawDialPtrList();
  const std::vector<Dial *> &getRawDialPtrList() const;
  const std::vector<GenericToolbox::AnyType>& getLeafHolder(const std::string &leafName_) const;
  const std::vector<GenericToolbox::AnyType>& getLeafHolder(int index_) const;
  const std::vector<std::vector<GenericToolbox::AnyType>> &getLeafContentList() const;
  const std::shared_ptr<std::vector<std::string>>& getCommonLeafNameListPtr() const;

  std::vector<std::vector<GenericToolbox::AnyType>> &getLeafContentList();

  // CORE
  // Filling up
  void copyOnlyExistingLeaves(const PhysicsEvent& other_);

  // Weight
  void addEventWeight(double weight_);
  void resetEventWeight();
  void reweightUsingDialCache();

  // Fetch var
  int findVarIndex(const std::string& leafName_, bool throwIfNotFound_ = true) const;
  template<typename T> auto getVarValue(const std::string& leafName_, size_t arrayIndex_ = 0) const -> T;
  template<typename T> auto getVariable(const std::string& leafName_, size_t arrayIndex_ = 0) -> T&;
  void* getVariableAddress(const std::string& leafName_, size_t arrayIndex_ = 0);
  double getVarAsDouble(const std::string& leafName_, size_t arrayIndex_ = 0) const;
  double getVarAsDouble(int varIndex_, size_t arrayIndex_ = 0) const;
  const GenericToolbox::AnyType& getVar(int varIndex_, size_t arrayIndex_ = 0) const;
  void fillBuffer(const std::vector<int>& indexList_, std::vector<double>& buffer_) const;

  // Eval
  double evalFormula(TFormula* formulaPtr_, std::vector<int>* indexDict_ = nullptr) const;

  // Misc
  void print() const;
  void trimDialCache();
  std::string getSummary() const;
  std::map<std::string, std::function<void(GenericToolbox::RawDataArray&, const std::vector<GenericToolbox::AnyType>&)>> generateLeavesDictionary(bool disableArrays_ = false) const;

  void copyData(const std::vector<std::pair<const GenericToolbox::LeafHolder*, int>>& dict_, bool disableArrayStorage_=false);
  std::vector<std::pair<const GenericToolbox::LeafHolder*, int>> generateDict(const GenericToolbox::TreeEventBuffer& h_, const std::map<std::string, std::string>& leafDict_={});
  void copyLeafContent(const PhysicsEvent& ref_);
  void resizeVarToDoubleCache();
  void invalidateVarToDoubleCache();

  // Stream operator
  friend std::ostream& operator <<( std::ostream& o, const PhysicsEvent& p );

  // DEV
  void addNestedDialRefToCache(NestedDialTest* nestedDialPtr_, const std::vector<Dial*>& dialPtrList_ = std::vector<Dial*>{});

private:
  // Context variables
  int _dataSetIndex_{-1};
  Long64_t _entryIndex_{-1};
  double _treeWeight_{1};
  double _nominalWeight_{1};
  double _eventWeight_{1};
  int _sampleBinIndex_{-1};

  // Data storage variables
  std::shared_ptr<std::vector<std::string>> _commonLeafNameListPtr_{nullptr};
  std::vector<std::vector<GenericToolbox::AnyType>> _leafContentList_;

  // Cache variables
  std::vector<Dial*> _rawDialPtrList_{};
  std::vector<std::pair<NestedDialTest*, std::vector<Dial*>>> _nestedDialRefList_{};
  mutable std::vector<std::vector<double>> _varToDoubleCache_{};
#ifdef GUNDAM_USING_CACHE_MANAGER
public:
  void setCacheManagerIndex(int i) {_CacheManagerIndex_ = i;}
  int  getCacheManagerIndex() {return _CacheManagerIndex_;}
  void setCacheManagerValuePointer(const double* v) {_CacheManagerValue_ = v;}
  void setCacheManagerValidPointer(const bool* v) {_CacheManagerValid_ = v;}
  void setCacheManagerUpdatePointer(void (*p)()) {_CacheManagerUpdate_ = p;}
private:
  // An "opaque" index into the cache that is used to simplify bookkeeping.
  int _CacheManagerIndex_{-1};
  // A pointer to the cached result.
  const double* _CacheManagerValue_{nullptr};
  // A pointer to the cache validity flag.
  const bool* _CacheManagerValid_{nullptr};
  // A pointer to a callback to force the cache to be updated.
  void (*_CacheManagerUpdate_)(){nullptr};
#endif


};


// TEMPLATES IMPLEMENTATION
template<typename T> auto PhysicsEvent::getVarValue(const std::string &leafName_, size_t arrayIndex_) const -> T {
  int index = this->findVarIndex(leafName_, true);
  return _leafContentList_[index][arrayIndex_].template getValue<T>();
}
template<typename T> auto PhysicsEvent::getVariable(const std::string& leafName_, size_t arrayIndex_) -> T&{
  int index = this->findVarIndex(leafName_, true);
  return _leafContentList_[index][arrayIndex_].template getValue<T>();
}


#endif //GUNDAM_PHYSICSEVENT_H
