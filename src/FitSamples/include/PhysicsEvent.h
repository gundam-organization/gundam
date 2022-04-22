//
// Created by Nadrino on 22/07/2021.
//

#ifndef GUNDAM_PHYSICSEVENT_H
#define GUNDAM_PHYSICSEVENT_H

#include <GenericToolbox.RawDataArray.h>
#include "vector"
#include "string"
#include "map"
#include "TTree.h"
#include "TFormula.h"

#include "GenericToolbox.Root.LeafHolder.h"

#include "AnaEvent.hh"
#include "FitParameterSet.h"
#include "Dial.h"
#include "NestedDialTest.h"


class PhysicsEvent {

public:
  PhysicsEvent();
  virtual ~PhysicsEvent();

  void reset();

  // SETTERS
  void setLeafNameListPtr(const std::vector<std::string> *leafNameListPtr);
  void setDataSetIndex(int dataSetIndex_);
  void setEntryIndex(Long64_t entryIndex_);
  void setTreeWeight(double treeWeight);
  void setNominalWeight(double nominalWeight);
  void setEventWeight(double eventWeight);
  void setFakeDataWeight(double fakeDataWeight);
  void setSampleBinIndex(int sampleBinIndex);

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
  const GenericToolbox::LeafHolder& getLeafHolder(std::string leafName_) const;
  const GenericToolbox::LeafHolder& getLeafHolder(int index_) const;
  const std::vector<GenericToolbox::LeafHolder> &getLeafContentList() const;
  const std::vector<std::string> *getCommonLeafNameListPtr() const;

  // CORE
  // Filling up
  void hookToTree(TTree* tree_, bool throwIfLeafNotFound_ = true);
  void clonePointerLeaves();
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

  // Eval
  double evalFormula(TFormula* formulaPtr_, std::vector<int>* indexDict_ = nullptr) const;

  // Misc
  std::string getSummary() const;
  void print() const;
  bool isSame(AnaEvent& anaEvent_) const;
  void deleteLeaf(long index_);
  void trimDialCache();
  void addDialRefToCache(Dial* dialPtr_);
  void addNestedDialRefToCache(NestedDialTest* nestedDialPtr_, const std::vector<Dial*>& dialPtrList_ = std::vector<Dial*>{});
  std::map<std::string, std::function<void(GenericToolbox::RawDataArray&, const GenericToolbox::LeafHolder&)>> generateLeavesDictionary(bool disableArrays_ = false) const;

  // Stream operator
  friend std::ostream& operator <<( std::ostream& o, const PhysicsEvent& p );

private:

  // TTree related members
  const std::vector<std::string>* _commonLeafNameListPtr_{nullptr};
  std::vector<GenericToolbox::LeafHolder> _leafContentList_;


  // Extra variables
  int _dataSetIndex_{-1};
  Long64_t _entryIndex_{-1};
  double _treeWeight_{1};
  double _nominalWeight_{1};
  double _eventWeight_{1};
  double _fakeDataWeight_{1};
  int _sampleBinIndex_{-1};

#ifdef GUNDAM_USING_CUDA
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

  // Caches
  std::vector<Dial*> _rawDialPtrList_{};
  std::vector<std::pair<NestedDialTest*, std::vector<Dial*>>> _nestedDialRefList_{};

};


// TEMPLATES IMPLEMENTATION
template<typename T> auto PhysicsEvent::getVarValue(const std::string &leafName_, size_t arrayIndex_) const -> T {
  int index = this->findVarIndex(leafName_, true);
  return _leafContentList_.at(index).template getVariable<T>(arrayIndex_);
}
template<typename T> auto PhysicsEvent::getVariable(const std::string& leafName_, size_t arrayIndex_) -> T&{
  int index = this->findVarIndex(leafName_, true);
  return _leafContentList_.at(index).template getVariable<T>(arrayIndex_);
}


#endif //GUNDAM_PHYSICSEVENT_H
