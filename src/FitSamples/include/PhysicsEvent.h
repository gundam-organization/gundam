//
// Created by Nadrino on 22/07/2021.
//

#ifndef GUNDAM_PHYSICSEVENT_H
#define GUNDAM_PHYSICSEVENT_H

#include "FitParameterSet.h"

#include "GenericToolbox.Root.TreeEntryBuffer.h"
#include "GenericToolbox.Root.LeafHolder.h"
#include "GenericToolbox.Root.LeafCollection.h"
#include <GenericToolbox.RawDataArray.h>
#include "GenericToolbox.AnyType.h"

#include "TTree.h"
#include "TFormula.h"

#include <map>
#include <mutex>
#include <vector>
#include <string>

class PhysicsEvent {

public:
  PhysicsEvent();
  virtual ~PhysicsEvent();

  // SETTERS
  void setDataSetIndex(int dataSetIndex_);
  void setEntryIndex(Long64_t entryIndex_);
  void setTreeWeight(double treeWeight);
  void setNominalWeight(double nominalWeight);
  void setEventWeight(double eventWeight);
  void setSampleBinIndex(int sampleBinIndex);
  void setSampleIndex(int sampleIndex);
  void setCommonVarNameListPtr(const std::shared_ptr<std::vector<std::string>>& commonVarNameListPtr_);

  // GETTERS
  int getDataSetIndex() const;
  Long64_t getEntryIndex() const;
  double getTreeWeight() const;
  double getNominalWeight() const;
  double getEventWeight() const;
  int getSampleBinIndex() const;
  int getSampleIndex() const;
  const std::vector<GenericToolbox::AnyType>& getLeafHolder(const std::string &leafName_) const;
  const std::vector<GenericToolbox::AnyType>& getLeafHolder(int index_) const;
  const std::vector<std::vector<GenericToolbox::AnyType>> &getLeafContentList() const;
  const std::shared_ptr<std::vector<std::string>>& getCommonLeafNameListPtr() const;
  std::vector<std::vector<GenericToolbox::AnyType>> &getLeafContentList();
  double& getEventWeightRef(){ return _eventWeight_; }

  // CORE
  // Filling up
  void copyOnlyExistingLeaves(const PhysicsEvent& other_);

  // Weight
  void addEventWeight(double weight_);
  void resetEventWeight();

  // Fetch var
  int findVarIndex(const std::string& leafName_, bool throwIfNotFound_ = true) const;
  template<typename T> auto getVarValue(const std::string& leafName_, size_t arrayIndex_ = 0) const -> T;
  template<typename T> auto getVariable(const std::string& leafName_, size_t arrayIndex_ = 0) -> const T&;
  template<typename T> void setVariable(const T& value_, const std::string& leafName_, size_t arrayIndex_ = 0);
  void* getVariableAddress(const std::string& leafName_, size_t arrayIndex_ = 0);
  double getVarAsDouble(const std::string& leafName_, size_t arrayIndex_ = 0) const;
  double getVarAsDouble(int varIndex_, size_t arrayIndex_ = 0) const;
  const GenericToolbox::AnyType& getVar(int varIndex_, size_t arrayIndex_ = 0) const;
  void fillBuffer(const std::vector<int>& indexList_, std::vector<double>& buffer_) const;

  // Eval
  double evalFormula(const TFormula* formulaPtr_, std::vector<int>* indexDict_ = nullptr) const;

  // Misc
  void print() const;
  std::string getSummary() const;

  std::map<std::string, std::function<void(GenericToolbox::RawDataArray&, const std::vector<GenericToolbox::AnyType>&)>> generateLeavesDictionary(bool disableArrays_ = false) const;

  void allocateMemory(const std::vector<const GenericToolbox::LeafForm*>& leafFormList_);
  void copyData(const std::vector<const GenericToolbox::LeafForm*>& leafFormList_);
  void copyData(const std::vector<std::pair<const GenericToolbox::LeafHolder *, int>> &dict_);
  std::vector<std::pair<const GenericToolbox::LeafHolder*, int>> generateDict(const GenericToolbox::TreeEntryBuffer& treeEventBuffer_, const std::map<std::string, std::string>& leafDict_={});
  void copyLeafContent(const PhysicsEvent& ref_);
  void resizeVarToDoubleCache();
  void invalidateVarToDoubleCache();

  // Stream operator
  friend std::ostream& operator <<( std::ostream& o, const PhysicsEvent& p );

private:
  // Context variables
  int _dataSetIndex_{-1};
  Long64_t _entryIndex_{-1};
  double _treeWeight_{1};
  double _nominalWeight_{1};
  double _eventWeight_{1};
  int _sampleBinIndex_{-1};
  int _sampleIndex_{-1}; // this information is lost in the EventDialCache manager

  // Data storage variables
  std::shared_ptr<std::vector<std::string>> _commonVarNameListPtr_{nullptr};
  std::vector<std::vector<GenericToolbox::AnyType>> _leafContentList_{};

  // Cache variables
  mutable std::vector<std::vector<double>> _varToDoubleCache_{};

#ifdef GUNDAM_USING_CACHE_MANAGER
public:
  void setCacheManagerIndex(int i) {_CacheManagerIndex_ = i;}
  int  getCacheManagerIndex() const {return _CacheManagerIndex_;}
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
template<typename T> auto PhysicsEvent::getVariable(const std::string& leafName_, size_t arrayIndex_) -> const T&{
  int index = this->findVarIndex(leafName_, true);
  return _leafContentList_[index][arrayIndex_].template getValue<T>();
}
template<typename T> void PhysicsEvent::setVariable(const T& value_, const std::string& leafName_, size_t arrayIndex_){
  int index = this->findVarIndex(leafName_, true);
  _leafContentList_[index][arrayIndex_].template getValue<T>() = value_;
  if( not _varToDoubleCache_.empty() ){ _varToDoubleCache_[index][arrayIndex_] = std::nan("unset"); }
}


#endif //GUNDAM_PHYSICSEVENT_H
