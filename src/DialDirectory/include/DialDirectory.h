//
// Created by Adrien BLANCHET on 13/04/2022.
//

#ifndef GUNDAM_DIALDIRECTORY_H
#define GUNDAM_DIALDIRECTORY_H

#include "DialCollection.h"

#include "vector"
#include "string"

// A DialDirectory contains a list of DialCollection
// DialDirectory is related to DataSet (different datasets may define their own way of propagating systematics)
// Owned by Propagator

class DialDirectory {

public:
  DialDirectory();
  virtual ~DialDirectory();

  void initialize();

private:
  std::vector<std::string> _dataSetSelection_{1, ""};
  std::vector<DialCollection> _dialSetList_{};

};


#endif //GUNDAM_DIALDIRECTORY_H
