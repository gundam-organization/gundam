#include "CudaBackend.h"
#include "CpuBackend.h"
#include "CudaWeightModel.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
  using namespace Backends;

  EngineView makeModel() {
    EngineView view;
    auto& model = view.propagation;
    model.parameterCount = 3;
    model.variableInputBufferCount = 1;
    model.hasVariableResponses = true;
    model.totalBins = 7;
    model.dialInputs = {{0}, {1}, {2}};
    model.externalWeightBlocks = {{0, 1}, {1, 2}};
    BackendDialDescriptor norm;
    norm.inputCount = 1;
    BackendDialDescriptor surface;
    surface.type = BackendDialType::Bilinear;
    surface.inputCount = 2;
    surface.payloadSize = 10;
    model.dialPayloads = {2., 2., -2., 2., -2., 2., -.1, .7, .3, 1.1}; // .5 + .1*x + .2*y
    auto scale = norm;
    scale.firstInput = 1;
    BackendDialDescriptor shift;
    shift.type = BackendDialType::Shift;
    shift.payloadOffset = 10;
    shift.payloadSize = 1;
    model.dialPayloads.push_back(.1);
    BackendDialDescriptor external;
    external.type = BackendDialType::ExternalWeight;
    external.inputCount = 99; // A producer dependency list is not a device input sequence.
    BackendDialDescriptor table;
    table.type = BackendDialType::Tabulated;
    table.externalBlockIndex = 1;
    table.tableTermCount = 2;
    auto kriged = table;
    kriged.type = BackendDialType::Kriged;
    kriged.firstTableTerm = 2;
    kriged.tableTermCount = 3;
    auto weight = norm;
    weight.firstInput = 2;
    model.dials = {norm, surface, scale, shift, external, table, kriged, weight};
    model.tableTerms = {{0, .25}, {1, .75}, {1, -.25}, {0, .5}, {1, .75}};
    model.variableDialIndices = {2, 0, 3, 5, 6, 3};
    SampleView dynamic;
    dynamic.binCount = 3;
    dynamic.useDynamicBinning = true;
    dynamic.binning.variableNames = {"X", "Y"};
    dynamic.binning.intervals = {{0, 0., 1.}, {1, -1., 2.},
                                {0, .5, 2.}, {1, -1., 2.},
                                {0, 2., 2.}, {1, -1., 2.}};
    // Overlap and deliberately reordered identifiers: first matching bin wins.
    dynamic.binning.bins = {{2, 0, 2}, {0, 2, 2}, {1, 4, 2}};
    auto fixed = dynamic;
    fixed.binOffset = 3;
    fixed.binCount = 2;
    fixed.useDynamicBinning = false;
    fixed.binning.bins.clear();
    SampleView host;
    host.binOffset = 5;
    host.binCount = 2;
    host.useDynamicBinning = host.readHostVariables = true;
    host.binning.variableNames = {"X"};
    host.binning.intervals = {{0, -1., 0.}, {0, 0., 1.}};
    host.binning.bins = {{0, 0, 1}, {1, 1, 1}};
    model.samples = {dynamic, fixed, host};
    for( std::size_t i = 0; i < 67; ++i ){
      EventView event;
      event.resultIndex = (7 * i) % 67;
      event.samplePosition = i % 3;
      event.binIndex = i % 3 == 0 ? 2 : int(i % 2);
      event.globalBinIndex = model.samples[event.samplePosition].binOffset + event.binIndex;
      event.firstVariable = model.variableValues.size();
      event.variableCount = 3;
      const double x = i % 3 == 2 ? (i % 2 ? .25 : -.25) : .2 + .1 * (i % 4);
      model.variableValues.insert(model.variableValues.end(), {x, 0., 1.});
      event.firstVariableResponse = model.variableResponses.size();
      if( i % 3 != 2 ){
        BackendVariableResponseDescriptor target;
        target.variableIndex = event.firstVariable;
        target.referenceValue = x;
        target.firstMultiplyDial = 0;
        target.multiplyDialCount = 1;
        target.firstAddDial = 1;
        target.addDialCount = 2;
        model.variableResponses.push_back(target);
      }
      BackendVariableResponseDescriptor y;
      y.variableIndex = event.firstVariable + 1;
      y.overrideDial = 1;
      model.variableResponses.push_back(y);
      BackendVariableResponseDescriptor z;
      z.variableIndex = event.firstVariable + 2;
      z.overrideDial = 4;
      z.firstMultiplyDial = 3;
      z.multiplyDialCount = 2;
      z.firstAddDial = 5;
      z.addDialCount = 1;
      model.variableResponses.push_back(z);
      event.variableResponseCount = model.variableResponses.size() - event.firstVariableResponse;
      model.events.push_back(event);
      model.eventWeights.resize(67);
      model.eventWeights[event.resultIndex] = {i % 11 == 0 ? -.125 : .25, model.eventDialIndices.size(), 2};
      model.eventDialIndices.insert(model.eventDialIndices.end(), {7, 4});
    }
    LikelihoodSampleView sample;
    sample.dataSums.assign(7, 1.);
    sample.evalBin = [](double data_, double prediction_, double error_, int){
      return (prediction_ - data_) * (prediction_ - data_) + error_ * error_;
    };
    view.likelihood.samples.push_back(sample);
    return view;
  }

  void expectClose(double actual_, double expected_) {
    EXPECT_NEAR(actual_, expected_, 5e-12 * std::max(1., std::abs(expected_)));
  }

  void compare(CudaBackend& cuda_, CpuBackend& cpu_, const PropagationInputs& inputs_) {
    const auto token = cuda_.requestPropagation(inputs_), reference = cpu_.requestPropagation(inputs_);
    ASSERT_EQ(cuda_.getStatus(token).eventState, OutputState::ReadyOnDevice);
    const auto& actual = cuda_.getEventStateHostView(token);
    const auto& expected = cpu_.getEventStateHostView(reference);
    EXPECT_EQ(cuda_.getStatus(token).eventState, OutputState::ReadyOnHost);
    EXPECT_EQ(actual.bins, expected.bins);
    EXPECT_EQ(actual.revision, expected.revision);
    EXPECT_EQ(actual.variableRevision, expected.variableRevision);
    EXPECT_EQ(actual.sampleRevisions, expected.sampleRevisions);
    ASSERT_EQ(actual.variables.size(), expected.variables.size());
    for( std::size_t i = 0; i < actual.variables.size(); ++i ){ expectClose(actual.variables[i], expected.variables[i]); }
    ASSERT_EQ(actual.contributions.size(), expected.contributions.size());
    for( std::size_t i = 0; i < actual.contributions.size(); ++i ){
      ASSERT_EQ(actual.contributions[i].size(), expected.contributions[i].size());
      for( std::size_t j = 0; j < actual.contributions[i].size(); ++j ){
        EXPECT_EQ(actual.contributions[i][j].binIndex, expected.contributions[i][j].binIndex);
        expectClose(actual.contributions[i][j].fraction, expected.contributions[i][j].fraction);
      }
    }
    const auto bytes = cuda_.getLastTimingSummary().eventStateReadbackBytes;
    cuda_.materialize(token, OutputRequest::EventState);
    EXPECT_EQ(cuda_.getLastTimingSummary().eventStateReadbackBytes, bytes);
    const auto& weights = cuda_.getEventWeightsHostView(token);
    const auto& cpuWeights = cpu_.getEventWeightsHostView(reference);
    for( std::size_t i = 0; i < weights.size(); ++i ){ expectClose(weights[i], cpuWeights[i]); }
    const auto& sums = cuda_.getHistogramSumsHostView(token);
    const auto& squares = cuda_.getHistogramSumSquaresHostView(token);
    for( std::size_t i = 0; i < sums.size(); ++i ){
      expectClose(sums[i], cpu_.getHistogramSumsHostView(reference)[i]);
      expectClose(squares[i], cpu_.getHistogramSumSquaresHostView(reference)[i]);
    }
    if( not cuda_.getEngineView().likelihood.empty() ){ expectClose(cuda_.getLikelihood(token), cpu_.getLikelihood(reference)); }
  }
}

TEST(CudaBackendObservable, CompositionMigrationAndCachesMatchCpu) {
  for( const auto* mode : {"cachedResponses", "recomputeResponses"} ){
    for( const auto* layout : {"eventMajor", "warpInterleaved"} ){
      SCOPED_TRACE(std::string(mode) + "/" + layout);
      auto view = makeModel();
      CudaBackend cuda;
      cuda.configure(ConfigReader(JsonType{{"propagationMode", mode}, {"weightLayout", layout}}));
      CpuBackend cpu;
      cuda.build(view);
      cpu.build(view);
      std::vector<double> external{.5}, table{1., 2.};
      PropagationInputs inputs;
      inputs.parameters.values = {0., 1., 1.};
      inputs.externalWeights = {{external.data(), 1, 0, 1}, {table.data(), 2, 1, 1}};
      inputs.variableInputRevisions = {1};
      compare(cuda, cpu, inputs);
      const auto first = cuda.requestPropagation(inputs);
      const auto& variables = cuda.getEventStateHostView(first).variables;
      expectClose(variables[0], .3); // Reference * Multiply + Add, never accumulated.
      expectClose(variables[1], .7); // Surface override.
      expectClose(variables[2], 1.4125); // External override * Tabulated * Kriged + Add.
      for( int i = 0; i < 4; ++i ){ compare(cuda, cpu, inputs); }
      EXPECT_EQ(cuda.getLastTimingSummary().rebinnedEventCount, 0);
      inputs.parameters.values[2] = 2.; // Weight-only parameter: resident state untouched.
      compare(cuda, cpu, inputs);
      EXPECT_EQ(cuda.getLastTimingSummary().rebinnedEventCount, 0);
      for( double shift : {.6, 1.7, 3., -2., 0.} ){
        inputs.parameters.values[0] = shift;
        compare(cuda, cpu, inputs);
        EXPECT_GT(cuda.getLastTimingSummary().rebinnedEventCount, 0);
      }
      inputs.parameters.values[1] = .7; // Second surface input and Multiply.
      compare(cuda, cpu, inputs);
      external[0] = -.25; // Only a non-binning observable changes.
      ++inputs.externalWeights[0].generation;
      compare(cuda, cpu, inputs);
      table[1] = .25;
      ++inputs.externalWeights[1].generation;
      compare(cuda, cpu, inputs);
      ++inputs.variableInputRevisions[0]; // Same numerical responses, new input revision.
      compare(cuda, cpu, inputs);
      inputs.hostVariables = view.propagation.variableValues;
      for( const auto& event : view.propagation.events ){
        if( event.samplePosition == 2 ){ inputs.hostVariables[event.firstVariable] = .75; }
      }
      compare(cuda, cpu, inputs);
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

TEST(CudaBackendObservable, ErrorsInvalidateTokensAndRetryAllState) {
  auto view = makeModel();
  CudaBackend cuda;
  CpuBackend cpu;
  cuda.build(view);
  cpu.build(view);
  std::vector<double> external{.5}, table{1., 2.};
  PropagationInputs inputs;
  inputs.parameters.values = {0., 1., 1.};
  inputs.externalWeights = {{external.data(), 1, 0, 1}, {table.data(), 2, 1, 1}};
  const auto stale = cuda.requestPropagation(inputs);
  external[0] = std::numeric_limits<double>::quiet_NaN();
  ++inputs.externalWeights[0].generation;
  EXPECT_THROW(cuda.requestPropagation(inputs), std::runtime_error);
  EXPECT_THROW((void)cuda.getEventStateHostView(stale), std::runtime_error);
  external[0] = .25;
  // Both caches restart. Use a fresh CPU instance so revisions also restart.
  cpu.build(view);
  const auto token = cuda.requestPropagation(inputs);
  const auto reference = cpu.requestPropagation(inputs);
  const auto& state = cuda.getEventStateHostView(token);
  const auto& expected = cpu.getEventStateHostView(reference);
  EXPECT_EQ(state.bins, expected.bins);
  for( std::size_t i = 0; i < state.variables.size(); ++i ){ expectClose(state.variables[i], expected.variables[i]); }
  EXPECT_EQ(cuda.getLastTimingSummary().cacheCandidateCount, view.propagation.dials.size());
  // Finite individual responses can still overflow during observable composition.
  // Remove weight dials so only the variable kernel can report this failure.
  for( auto& event : view.propagation.eventWeights ){ event.dialCount = 0; }
  cuda.build(view);
  cpu.build(view);
  external[0] = std::numeric_limits<double>::max();
  ++inputs.externalWeights[0].generation;
  EXPECT_THROW(cuda.requestPropagation(inputs), std::runtime_error);
  external[0] = .25;
  const auto retry = cuda.requestPropagation(inputs), cpuRetry = cpu.requestPropagation(inputs);
  const auto& retried = cuda.getEventStateHostView(retry).variables;
  const auto& cpuRetried = cpu.getEventStateHostView(cpuRetry).variables;
  for( std::size_t i = 0; i < retried.size(); ++i ){ expectClose(retried[i], cpuRetried[i]); }
  auto bad = inputs;
  bad.hostVariables.assign(view.propagation.variableValues.size(), 0.);
  bad.hostVariables[0] = std::numeric_limits<double>::infinity();
  EXPECT_THROW(cuda.requestPropagation(bad), std::runtime_error);
  bad = inputs;
  bad.variableInputRevisions = {1, 2};
  EXPECT_THROW(cuda.requestPropagation(bad), std::runtime_error);
  bad = inputs;
  bad.hostVariables = {1.};
  EXPECT_THROW(cuda.requestPropagation(bad), std::runtime_error);
  EXPECT_NO_THROW(cuda.requestPropagation(inputs));
}

TEST(CudaBackendObservable, EmptyModelAndZeroWidthStrictBin) {
  EngineView view;
  view.propagation.hasVariableResponses = true;
  CudaBackend cuda;
  cuda.build(view);
  const auto token = cuda.requestPropagation({});
  EXPECT_TRUE(cuda.getEventStateHostView(token).variables.empty());
  EXPECT_TRUE(cuda.getEventStateHostView(token).bins.empty());
  auto& model = view.propagation;
  SampleView sample;
  sample.useDynamicBinning = sample.readHostVariables = true;
  sample.binCount = model.totalBins = 2;
  sample.binning.variableNames = {"X"};
  sample.binning.intervals = {{0, 1., 1.}, {0, 0., 2.}};
  sample.binning.bins = {{1, 0, 1}, {0, 1, 1}};
  model.samples = {sample};
  model.variableValues = {0.};
  model.events.resize(1);
  model.events[0].samplePosition = 0;
  model.events[0].variableCount = 1;
  model.eventWeights = {{1., 0, 0}};
  cuda.build(view);
  CpuBackend cpu;
  cpu.build(view);
  PropagationInputs inputs;
  for( double x : {0., 1., 1.5, 2., -1., 1.} ){
    inputs.hostVariables = {x};
    compare(cuda, cpu, inputs);
  }
}
