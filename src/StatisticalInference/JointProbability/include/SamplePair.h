//
// Created by Nadrino on 10/09/2024.
//

#ifndef GUNDAM_SAMPLEPAIR_H
#define GUNDAM_SAMPLEPAIR_H

#include "Sample.h"

struct SamplePair{
  // associate two samples from MC and Data for the statistal inference
  Sample* model{nullptr};
  Sample* data{nullptr};
};

#endif //GUNDAM_SAMPLEPAIR_H
