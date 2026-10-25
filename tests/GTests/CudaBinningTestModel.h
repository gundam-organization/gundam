#ifndef GUNDAM_CUDA_BINNING_TEST_MODEL_H
#define GUNDAM_CUDA_BINNING_TEST_MODEL_H

#include "EngineView.h"

namespace Backends::Testing {
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
