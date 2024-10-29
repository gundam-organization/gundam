//
// Created by Nadrino on 28/10/2024.
//

#ifndef GUNDAM_CACHEEVENTWEIGHTFILLER_H
#define GUNDAM_CACHEEVENTWEIGHTFILLER_H

#include "Event.h"


class CacheEventWeightFiller{

public:
  CacheEventWeightFiller(Event* eventPtr_, int valueIndex_): eventPtr(eventPtr_), valueIndex(valueIndex_) {}

  [[nodiscard]] Event *getEventPtr() const{ return eventPtr; }
  [[nodiscard]] int getValueIndex() const{ return valueIndex; }

  void copyCacheToCpu(const double* eventWeightsArray_);

private:
  Event* eventPtr{nullptr};
  int valueIndex{-1};

};


#endif //GUNDAM_CACHEEVENTWEIGHTFILLER_H
