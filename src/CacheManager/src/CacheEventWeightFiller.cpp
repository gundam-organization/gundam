//
// Created by Nadrino on 28/10/2024.
//

#include "CacheEventWeightFiller.h"

void CacheEventWeightFiller::copyCacheToCpu(const double* eventWeightsArray_){

  eventPtr->getWeights().current = eventWeightsArray_[valueIndex];

//  if (not GundamGlobals::isForceCpuCalculation()) return value;
//  if (not GundamUtils::almostEqual(value, _weights_.current)) {
//    const double magnitude = std::abs(value) + std::abs(_weights_.current);
//    double delta = std::abs(value - _weights_.current);
//    if (magnitude > 0.0) delta /= 0.5*magnitude;
//    LogError << "Inconsistent event weight -- "
//             << " Calculated: " << value
//             << " Cached: " << _weights_.current
//             << " Precision: " << delta
//             << std::endl;
//  }

}
