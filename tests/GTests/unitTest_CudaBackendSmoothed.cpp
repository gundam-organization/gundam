#include "CudaBackend.h"
#include "CpuBackend.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <stdexcept>
#include <vector>

namespace {
  using namespace Backends;

  EngineView makeModel() {
    EngineView view;
    auto& model = view.propagation;
    model.parameterCount = 3;
    model.variableInputBufferCount = 1;
    model.hasVariableResponses = true;
    model.hasReweightCap = true;
    model.maxReweight = 2.;
    model.totalBins = 9;
    model.dialInputs = {{0}, {1}, {2}};
    for( std::size_t i = 0; i < 3; ++i ){
      BackendDialDescriptor dial;
      dial.firstInput = i;
      dial.inputCount = 1;
      model.dials.push_back(dial);
    }
    BackendDialDescriptor external;
    external.type = BackendDialType::ExternalWeight;
    model.dials.push_back(external);
    model.externalWeightBlocks = {{0, 1}};
    model.variableDialIndices = {0, 1};
    SampleView surface;
    surface.binCount = 4;
    surface.useDynamicBinning = true;
    surface.binning.isSmoothed = true;
    surface.binning.variableNames = {"X", "Y"};
    for( int x = 0; x < 2; ++x ){
      for( int y = 0; y < 2; ++y ){
        surface.binning.bins.push_back({2*x + y, surface.binning.intervals.size(), 2});
        surface.binning.intervals.push_back({0, double(x), double(x + 1), 5., 5.});
        surface.binning.intervals.push_back({1, double(y), double(y + 1), 5., 5.});
      }
    }
    SampleView gap;
    gap.binOffset = 4;
    gap.binCount = 2;
    gap.binning.isSmoothed = true; // Smoothing works without useDynamicBinning.
    gap.readHostVariables = true;
    gap.binning.variableNames = {"X"};
    gap.binning.bins = {{1, 0, 1}, {0, 1, 1}};
    gap.binning.intervals = {{0, 0., 1., 5., 2.5}, {0, 3., 4., 5., 2.5}};
    SampleView strict;
    strict.binOffset = 6;
    strict.binCount = 2;
    strict.useDynamicBinning = true;
    strict.binning.variableNames = {"X"};
    strict.binning.intervals = {{0, 0., 1.}, {0, 1., 2.}};
    strict.binning.bins = {{0, 0, 1}, {1, 1, 1}};
    SampleView fixed;
    fixed.binOffset = 8;
    fixed.binCount = 1;
    model.samples = {surface, gap, strict, fixed};
    const double positions[]{-.4, 0., .1, .9, 1., 1.1, 1.9, 2., 3.5, 4.1};
    model.eventWeights.resize(521);
    for( std::size_t i = 0; i < 521; ++i ){
      EventView event;
      event.resultIndex = (37*i) % 521;
      event.samplePosition = i % 4;
      event.binIndex = i % 4 == 3 ? 0 : -1;
      event.globalBinIndex = i % 4 == 3 ? 8 : -1;
      event.firstVariable = model.variableValues.size();
      event.variableCount = 2;
      model.variableValues.insert(model.variableValues.end(), {positions[i % 10], positions[(i / 10) % 10]});
      event.firstVariableResponse = model.variableResponses.size();
      for( std::size_t j = 0; j < 2; ++j ){
        BackendVariableResponseDescriptor response;
        response.variableIndex = event.firstVariable + j;
        response.referenceValue = model.variableValues[event.firstVariable + j];
        response.firstAddDial = j;
        response.addDialCount = 1;
        if( j == 0 and i % 5 == 0 ){ response.overrideDial = 3; }
        model.variableResponses.push_back(response);
      }
      event.variableResponseCount = 2;
      model.events.push_back(event);
      model.eventWeights[event.resultIndex] = {i % 17 == 0 ? 0. : (i % 11 == 0 ? -.25 : .5), model.eventDialIndices.size(), 1};
      model.eventDialIndices.push_back(2);
    }
    LikelihoodSampleView sample;
    sample.dataSums.assign(9, 1.);
    sample.evalBin = [](double data_, double prediction_, double error_, int){
      return (prediction_ - data_) * (prediction_ - data_) + error_*error_;
    };
    view.likelihood.samples.push_back(sample);
    return view;
  }

  void close(double actual_, double expected_) {
    EXPECT_NEAR(actual_, expected_, 2e-11 * std::max(1., std::abs(expected_)));
  }

  void compare(CudaBackend& cuda_, CpuBackend& cpu_, const PropagationInputs& inputs_) {
    const auto token = cuda_.requestPropagation(inputs_), reference = cpu_.requestPropagation(inputs_);
    const auto& a = cuda_.getEventStateHostView(token);
    const auto& b = cpu_.getEventStateHostView(reference);
    EXPECT_EQ(a.bins, b.bins);
    EXPECT_EQ(a.revision, b.revision);
    EXPECT_EQ(a.variableRevision, b.variableRevision);
    EXPECT_EQ(a.sampleRevisions, b.sampleRevisions);
    ASSERT_EQ(a.variables.size(), b.variables.size());
    for( std::size_t i = 0; i < a.variables.size(); ++i ){ close(a.variables[i], b.variables[i]); }
    ASSERT_EQ(a.contributions.size(), b.contributions.size());
    for( std::size_t i = 0; i < a.contributions.size(); ++i ){
      ASSERT_EQ(a.contributions[i].size(), b.contributions[i].size());
      double total = 0.;
      for( std::size_t j = 0; j < a.contributions[i].size(); ++j ){
        EXPECT_EQ(a.contributions[i][j].binIndex, b.contributions[i][j].binIndex);
        close(a.contributions[i][j].fraction, b.contributions[i][j].fraction);
        total += a.contributions[i][j].fraction;
      }
      if( not a.contributions[i].empty() ){ close(total, 1.); }
    }
    const auto& weights = cuda_.getEventWeightsHostView(token);
    const auto& cpuWeights = cpu_.getEventWeightsHostView(reference);
    for( std::size_t i = 0; i < weights.size(); ++i ){ close(weights[i], cpuWeights[i]); }
    const auto& sums = cuda_.getHistogramSumsHostView(token);
    const auto& squares = cuda_.getHistogramSumSquaresHostView(token);
    for( std::size_t i = 0; i < sums.size(); ++i ){
      close(sums[i], cpu_.getHistogramSumsHostView(reference)[i]);
      close(squares[i], cpu_.getHistogramSumSquaresHostView(reference)[i]);
    }
    const auto& outside = cuda_.getOutsideSumsHostView(token);
    const auto& outsideSquares = cuda_.getOutsideSumSquaresHostView(token);
    ASSERT_EQ(outside.size(), cpu_.getOutsideSumsHostView(reference).size());
    for( std::size_t i = 0; i < outside.size(); ++i ){
      close(outside[i], cpu_.getOutsideSumsHostView(reference)[i]);
      close(outsideSquares[i], cpu_.getOutsideSumSquaresHostView(reference)[i]);
    }
    EXPECT_EQ(cuda_.getLastTimingSummary().binContributionCount, cpu_.getLastTimingSummary().binContributionCount);
    if( not cuda_.getEngineView().likelihood.empty() ){ close(cuda_.getLikelihood(token), cpu_.getLikelihood(reference)); }
  }
}

TEST(CudaBackendSmoothed, MixedSamplesMigrationCachingAndRebuildMatchCpu) {
  for( const auto* mode : {"cachedResponses", "recomputeResponses"} ){
    for( const auto* layout : {"eventMajor", "warpInterleaved"} ){
      SCOPED_TRACE(std::string(mode) + "/" + layout);
      auto view = makeModel();
      CudaBackend cuda;
      cuda.configure(ConfigReader(JsonType{{"propagationMode", mode}, {"weightLayout", layout}}));
      CpuBackend cpu;
      cuda.build(view);
      cpu.build(view);
      double external = .5;
      PropagationInputs inputs;
      inputs.parameters.values = {0., 0., 1.};
      inputs.externalWeights = {{&external, 1, 0, 1}};
      inputs.variableInputRevisions = {1};
      compare(cuda, cpu, inputs);
      compare(cuda, cpu, inputs);
      EXPECT_EQ(cuda.getLastTimingSummary().rebinnedEventCount, 0);
      inputs.parameters.values[2] = 3.; // Weight-only update, including cap.
      compare(cuda, cpu, inputs);
      EXPECT_EQ(cuda.getLastTimingSummary().rebinnedEventCount, 0);
      for( const auto& shift : {std::vector<double>{.15, 0., 1.}, {.15, .35, 1.},
                                {2., -1., 1.}, {10., 10., 1.}, {-10., -10., 1.}, {0., 0., 1.}} ){
        inputs.parameters.values = shift;
        compare(cuda, cpu, inputs);
      }
      external = 1.05;
      ++inputs.externalWeights[0].generation;
      compare(cuda, cpu, inputs);
      ++inputs.variableInputRevisions[0];
      compare(cuda, cpu, inputs);
      inputs.hostVariables = view.propagation.variableValues;
      for( const auto& event : view.propagation.events ){
        if( event.samplePosition == 1 ){ inputs.hostVariables[event.firstVariable] = .05; }
      }
      compare(cuda, cpu, inputs);
      inputs.forceBinningUpdate = true;
      compare(cuda, cpu, inputs);
      inputs.forceBinningUpdate = false;
      compare(cuda, cpu, inputs);
      const auto stale = cuda.requestPropagation(inputs);
      cuda.build(view);
      cpu.build(view);
      EXPECT_THROW((void)cuda.getEventStateHostView(stale), std::runtime_error);
      compare(cuda, cpu, inputs);
    }
  }
}

TEST(CudaBackendSmoothed, HierarchyAcrossSamplesAndLargeDisplacementsMatchesCpu) {
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
      sample.binning.intervals.push_back({0, x, x + 1., sampleIndex ? 5. : 0., sampleIndex ? 2.5 : 0.});
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
  for( const auto* mode : {"cachedResponses", "recomputeResponses"} ){
    CudaBackend cuda;
    cuda.configure(ConfigReader(JsonType{{"propagationMode", mode}}));
    CpuBackend cpu;
    cuda.build(view);
    cpu.build(view);
    PropagationInputs inputs;
    compare(cuda, cpu, inputs);
    for( double scale : {.01, .5, 1., 2.5, 100., 1.} ){
      inputs.hostVariables = model.variableValues;
      for( const auto& event : model.events ){ inputs.hostVariables[event.firstVariable] *= scale; }
      compare(cuda, cpu, inputs);
      EXPECT_EQ(cuda.getLastTimingSummary().rebinnedEventCount, model.events.size());
    }
  }
}

TEST(CudaBackendSmoothed, AnalyticQuarterFractionsAndOutsideSquares) {
  EngineView view;
  auto& model = view.propagation;
  SampleView sample;
  sample.binning.isSmoothed = sample.readHostVariables = true;
  sample.binCount = model.totalBins = 4;
  sample.binning.variableNames = {"X", "Y"};
  for( int x = 0; x < 2; ++x ){
    for( int y = 0; y < 2; ++y ){
      sample.binning.bins.push_back({2*x + y, sample.binning.intervals.size(), 2});
      sample.binning.intervals.push_back({0, double(x), double(x+1), 5., 5.});
      sample.binning.intervals.push_back({1, double(y), double(y+1), 5., 5.});
    }
  }
  model.samples = {sample};
  model.variableValues = {1., 1.};
  model.events.resize(1);
  model.events[0].samplePosition = 0;
  model.events[0].variableCount = 2;
  model.eventWeights = {{-2., 0, 0}};
  CudaBackend cuda;
  cuda.build(view);
  PropagationInputs inputs;
  const auto token = cuda.requestPropagation(inputs);
  for( auto value : cuda.getHistogramSumsHostView(token) ){ EXPECT_DOUBLE_EQ(value, -.5); }
  for( auto value : cuda.getHistogramSumSquaresHostView(token) ){ EXPECT_DOUBLE_EQ(value, .25); }
  EXPECT_DOUBLE_EQ(cuda.getOutsideSumsHostView(token)[0], 0.);
  inputs.hostVariables = {0., 0.};
  const auto edge = cuda.requestPropagation(inputs);
  // Only a quarter remains in the first bin; three quarters are outside.
  EXPECT_DOUBLE_EQ(cuda.getHistogramSumsHostView(edge)[0], -.5);
  EXPECT_DOUBLE_EQ(cuda.getHistogramSumSquaresHostView(edge)[0], .25);
  EXPECT_DOUBLE_EQ(cuda.getOutsideSumsHostView(edge)[0], -1.5);
  EXPECT_DOUBLE_EQ(cuda.getOutsideSumSquaresHostView(edge)[0], 2.25);
}

TEST(CudaBackendSmoothed, InvalidOverlapsAbortAndRetryWithoutPublishingPartialState) {
  EngineView view;
  auto& model = view.propagation;
  SampleView sample;
  sample.binCount = model.totalBins = 2;
  sample.binning.isSmoothed = sample.readHostVariables = true;
  sample.binning.variableNames = {"X"};
  sample.binning.bins = {{0, 0, 1}, {1, 1, 1}};
  sample.binning.intervals = {{0, 0., 1.}, {0, .5, 1.5}};
  model.samples = {sample};
  model.variableValues = {0.};
  model.events.resize(1);
  model.events[0].samplePosition = 0;
  model.events[0].variableCount = 1;
  model.eventWeights = {{2., 0, 0}};
  CudaBackend cuda;
  cuda.build(view);
  PropagationInputs inputs;
  const auto stale = cuda.requestPropagation(inputs);
  inputs.hostVariables = {.75};
  EXPECT_THROW(cuda.requestPropagation(inputs), std::runtime_error);
  EXPECT_THROW((void)cuda.getEventStateHostView(stale), std::runtime_error);
  inputs.hostVariables = {std::numeric_limits<double>::quiet_NaN()};
  EXPECT_THROW(cuda.requestPropagation(inputs), std::runtime_error);
  inputs.hostVariables = {0.};
  const auto retry = cuda.requestPropagation(inputs);
  EXPECT_DOUBLE_EQ(cuda.getHistogramSumsHostView(retry)[0], 2.);
  EXPECT_DOUBLE_EQ(cuda.getHistogramSumSquaresHostView(retry)[0], 4.);
  EXPECT_DOUBLE_EQ(cuda.getOutsideSumsHostView(retry)[0], 0.);
  EXPECT_DOUBLE_EQ(cuda.getEventStateHostView(retry).variables[0], 0.);
}

TEST(CudaBackendSmoothed, EmptyEventsAndNoSelectedBinsHaveDefinedOutsideState) {
  EngineView view;
  view.propagation.samples.emplace_back().binning.isSmoothed = true;
  CudaBackend cuda;
  cuda.build(view);
  auto token = cuda.requestPropagation({});
  EXPECT_TRUE(cuda.getEventStateHostView(token).contributions.empty());
  ASSERT_EQ(cuda.getOutsideSumsHostView(token).size(), 1);
  EXPECT_DOUBLE_EQ(cuda.getOutsideSumsHostView(token)[0], 0.);
  view.propagation.events.resize(1);
  view.propagation.events[0].samplePosition = 0;
  view.propagation.eventWeights = {{3., 0, 0}};
  cuda.build(view);
  token = cuda.requestPropagation({});
  EXPECT_DOUBLE_EQ(cuda.getOutsideSumsHostView(token)[0], 3.);
  EXPECT_DOUBLE_EQ(cuda.getOutsideSumSquaresHostView(token)[0], 9.);
  const auto& state = cuda.getEventStateHostView(token);
  ASSERT_EQ(state.contributions[0].size(), 1);
  EXPECT_EQ(state.contributions[0][0].binIndex, -1);
  EXPECT_DOUBLE_EQ(state.contributions[0][0].fraction, 1.);
}
