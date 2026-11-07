#ifndef GUNDAM_CUDA_BINNING_TEST_MODEL_H
#define GUNDAM_CUDA_BINNING_TEST_MODEL_H

#include "EngineView.h"

#include <cmath>
#include <limits>

namespace Backends::Testing {
// Fused polynomial evaluation can round a transition to one while the host
// retains an epsilon-sized neighbouring/outside contribution. Preserve order
// and bin identities for all significant records and bound total missing mass.
inline bool binningContributionsAgree(const std::vector<EventBinContribution>& first_,
                                     const std::vector<EventBinContribution>& second_) {
  constexpr double tolerance = 64. * std::numeric_limits<double>::epsilon();
  double firstTotal = 0., secondTotal = 0., missing = 0.;
  for( const auto& value : first_ ){
    if( not std::isfinite(value.fraction) or value.fraction < 0. ){ return false; }
    firstTotal += value.fraction;
  }
  for( const auto& value : second_ ){
    if( not std::isfinite(value.fraction) or value.fraction < 0. ){ return false; }
    secondTotal += value.fraction;
  }
  if( std::abs(firstTotal - secondTotal) > tolerance ){ return false; }
  std::size_t i = 0, j = 0;
  while( i < first_.size() or j < second_.size() ){
    if( i < first_.size() and j < second_.size() and first_[i].binIndex == second_[j].binIndex ){
      if( std::abs(first_[i].fraction - second_[j].fraction) > tolerance ){ return false; }
      ++i; ++j;
    }
    else if( i < first_.size() and first_[i].fraction <= tolerance ){ missing += first_[i++].fraction; }
    else if( j < second_.size() and second_[j].fraction <= tolerance ){ missing += second_[j++].fraction; }
    else{ return false; }
    if( missing > tolerance ){ return false; }
  }
  return true;
}

inline EngineView makeCudaBinningHierarchyTestModel() {
  EngineView view;
  auto& model = view.propagation;
  model.totalBins = 128;
  for( int sampleIndex = 0; sampleIndex < 2; ++sampleIndex ){
    SampleView sample;
    sample.binOffset = 64 * sampleIndex;
    sample.binCount = 64;
    sample.useDynamicBinning = sample.readHostVariables = true;
    sample.binning.isSmoothed = sampleIndex == 1;
    sample.binning.variableNames = {"X", "Y"};
    for( int i = 0; i < 64; ++i ){
      const double x = i % 8, y = i / 8;
      sample.binning.bins.push_back({(i * 17) % 64, sample.binning.intervals.size(), 2});
      // Adjacent bins must share the same width at their common edge.
      const double lowerInverse = (i % 8) % 2 ? 5. : 2.5;
      const double upperInverse = ((i % 8) + 1) % 2 ? 5. : 2.5;
      sample.binning.intervals.push_back({0, x, x + 1.,
          sampleIndex ? lowerInverse : 0., sampleIndex ? upperInverse : 0.});
      sample.binning.intervals.push_back({1, y, y + 1.});
    }
    model.samples.push_back(sample);
  }
  // Partial final warp and independent result/event order exercise aggregation.
  for( std::size_t i = 0; i < 521; ++i ){
    EventView event;
    event.resultIndex = (37*i) % 521;
    event.samplePosition = i % 2;
    event.firstVariable = 2*i;
    event.variableCount = 2;
    model.events.push_back(event);
    model.variableValues.insert(model.variableValues.end(), {double(i % 81) / 10. - .1, double((i / 7) % 81) / 10.});
    model.eventWeights.push_back({i % 3 ? 1. : -.5, 0, 0});
  }
  return view;
}
}

#endif
