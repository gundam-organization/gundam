//
// Created by Nadrino on 22/07/2021.
//

#ifndef XSLLHFITTER_PHYSICSEVENT_H
#define XSLLHFITTER_PHYSICSEVENT_H

#include "vector"
#include "string"
#include "map"
#include "TTree.h"
#include "TFormula.h"

#include "GenericToolbox.Root.LeafHolder.h"

#include "AnaEvent.hh"
#include "FitParameterSet.h"
#include "Dial.h"


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
  void setSampleBinIndex(int sampleBinIndex);

  // GETTERS
  int getDataSetIndex() const;
  Long64_t getEntryIndex() const;
  double getTreeWeight() const;
  double getNominalWeight() const;
  double getEventWeight() const;
  int getSampleBinIndex() const;
//  std::map<FitParameterSet *, std::vector<Dial *>>& getDialCache();
//  std::map<FitParameterSet *, std::vector<Dial *>>* getDialCachePtr();
  std::vector<Dial *> &getRawDialPtrList();

  const std::vector<std::string> *getCommonLeafNameListPtr() const;

  // CORE
  // Filling up
  void hookToTree(TTree* tree_, bool throwIfLeafNotFound_ = true);

  // Weight
  void addEventWeight(double weight_);
  void resetEventWeight();
  void reweightUsingDialCache();

  // Fetch var
  int findVarIndex(const std::string& leafName_, bool throwIfNotFound_ = true) const;
  template<typename T> auto getVarValue(const std::string& leafName_, size_t arrayIndex_ = 0) const -> T;
  template<typename T> auto getVariable(const std::string& leafName_, size_t arrayIndex_ = 0) -> T&;
  double getVarAsDouble(const std::string& leafName_, size_t arrayIndex_ = 0) const;
  double getVarAsDouble(int varIndex_, size_t arrayIndex_ = 0) const;
  double evalFormula(TFormula* formulaPtr_, std::vector<int>* indexDict_ = nullptr) const;

  // Misc
  std::string getSummary() const;
  void print() const;
  bool isSame(AnaEvent& anaEvent_) const;
  void deleteLeaf(size_t index_);

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
  int _sampleBinIndex_{-1};

  // Caches
//  std::map<FitParameterSet*, std::vector<Dial*>> _dialCache_; // _dialCache_[fitParSetPtr][ParIndex] = correspondingDialPtr;
  std::vector<Dial*> _rawDialPtrList_;

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


#endif //XSLLHFITTER_PHYSICSEVENT_H
