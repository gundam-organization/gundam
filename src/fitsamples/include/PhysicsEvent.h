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

  // GETTERS
  int getDataSetIndex() const;
  Long64_t getEntryIndex() const;
  double getTreeWeight() const;
  double getNominalWeight() const;
  double getEventWeight() const;
  std::map<FitParameterSet *, std::vector<Dial *>>* getDialCachePtr();

  // CORE
  // Filling up
  void hookToTree(TTree* tree_, bool throwIfLeafNotFound_ = true);

  // Weight
  void addEventWeight(double weight_);
  void resetEventWeight();

  // Fetch var
  int findVarIndex(const std::string& leafName_, bool throwIfNotFound_ = true) const;
  template<typename T> auto fetchValue(const std::string& leafName_, size_t arrayIndex_ = 0) -> T;
  double getVarAsDouble(const std::string& leafName_, size_t arrayIndex_ = 0) const;
  double getVarAsDouble(int varIndex_, size_t arrayIndex_ = 0) const;
  double evalFormula(TFormula* formulaPtr_, std::vector<int>* indexDict_ = nullptr) const;

  // Misc
  std::string getSummary() const;
  void print() const;

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

  // Caches
  std::map<FitParameterSet*, std::vector<Dial*>> _dialCache_; // _dialCache_[fitParSetPtr][ParIndex] = correspondingDialPtr;

};


#endif //XSLLHFITTER_PHYSICSEVENT_H
