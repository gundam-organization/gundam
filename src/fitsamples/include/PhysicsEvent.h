//
// Created by Nadrino on 22/07/2021.
//

#ifndef XSLLHFITTER_PHYSICSEVENT_H
#define XSLLHFITTER_PHYSICSEVENT_H

#include "vector"
#include "string"
#include "TTree.h"

#include "GenericToolbox.Root.LeafHolder.h"


class PhysicsEvent {

public:
  PhysicsEvent();
  virtual ~PhysicsEvent();

  void reset();

  void setLeafNameListPtr(const std::vector<std::string> *leafNameListPtr);
  void setDataSetIndex(int dataSetIndex_);
  void setEntryIndex(Long64_t entryIndex_);
  void setTreeWeight(double treeWeight);
  void setNominalWeight(double nominalWeight);
  void setEventWeight(double eventWeight);

  void hookToTree(TTree* tree_, bool throwIfLeafNotFound_ = true);

  int getDataSetIndex() const;

  Long64_t getEntryIndex() const;

  double getTreeWeight() const;
  double getNominalWeight() const;
  double getEventWeight() const;

  void addEventWeight(double weight_);
  void resetEventWeight();

  int findVarIndex(const std::string& leafName_, bool throwIfNotFound_ = true);
  template<typename T> auto fetchValue(const std::string& leafName_, size_t arrayIndex_ = 0) -> T;
  double getVarAsDouble(const std::string& leafName_, size_t arrayIndex_ = 0);

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

};


#endif //XSLLHFITTER_PHYSICSEVENT_H
