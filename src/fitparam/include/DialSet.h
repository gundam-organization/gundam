//
// Created by Adrien BLANCHET on 21/05/2021.
//

#ifndef XSLLHFITTER_DIALSET_H
#define XSLLHFITTER_DIALSET_H

#include "string"
#include "vector"

#include "GenericToolbox.h"
#include "Dial.h"


class DialSet {

public:
  DialSet();
  virtual ~DialSet();

  void reset();

private:
  std::string _name_; // ie detector name
  std::vector<Dial> _dialList_;

};


#endif //XSLLHFITTER_DIALSET_H
