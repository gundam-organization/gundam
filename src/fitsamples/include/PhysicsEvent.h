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

  void hookToTree(TTree* tree_);

  Int_t fetchIntValue( const std::string& leafName_ );

  std::string getSummary();
  void print();

private:

  // TTree related members
  const std::vector<std::string>* _commonLeafNameListPtr_{nullptr};
  std::vector<GenericToolbox::LeafHolder> _leafContentList_;

  // Weight carriers
  double _mcWeight_{1};
  double _nominalWeight_{1};
  double _eventWeight_{1};

  // Caches

};


#endif //XSLLHFITTER_PHYSICSEVENT_H
