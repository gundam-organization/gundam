#include "CpuBackend.h"
#include "EventStateCache.h"
#ifdef GUNDAM_TEST_MPS_AVAILABLE
#include "MpsBackend.h"
#endif
#include "Kriged.h"
#include "Tabulated.h"
#include "GundamGlobals.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace {
  Backends::EngineView makeTableModel() {
    using namespace Backends;
    EngineView view;
    auto& model = view.propagation;
    model.totalBins = 1;
    model.externalWeightBlocks = {{0, 3}, {3, 2}};
    BackendDialDescriptor tabulated;
    tabulated.type = BackendDialType::Tabulated;
    tabulated.tableTermCount = 2;
    BackendDialDescriptor kriged;
    kriged.type = BackendDialType::Kriged;
    kriged.externalBlockIndex = 1;
    kriged.firstTableTerm = 2;
    kriged.tableTermCount = 3;
    model.dials = {tabulated, kriged};
    // Nontrivial fraction; signed, repeated and unsorted Kriged indices.
    model.tableTerms = {{1, .25}, {2, .75}, {1, -.25}, {0, .5}, {1, .75}};
    model.eventDialIndices = {0, 1, 0};
    for( std::size_t i = 0; i < 2; ++i ){
      EventView event;
      event.resultIndex = i;
      event.globalBinIndex = 0;
      model.events.push_back(event);
    }
    model.eventWeights = {{2., 0, 2}, {3., 2, 1}};
    return view;
  }
}

namespace {
  void checkUpdatedTableValues(Backends::Backend& backend_) {
    using namespace Backends;
    GundamGlobals::setNumberOfThreads(2);
    auto model = makeTableModel();
    backend_.build(model);
    std::vector<double> first{10., 2., 4.}, second{3., 5.};
    Tabulated tabulated(&first, 1, .25);
    Kriged kriged(&second, 3, {1, 0, 1}, {-.25, .5, .75});
    DialInputBuffer buffer;
    PropagationInputs inputs;
    inputs.externalWeights = {{first.data(), first.size(), 0, 1}, {second.data(), second.size(), 3, 1}};
    auto check = [&]{
      const auto token = backend_.requestPropagation(inputs);
      ASSERT_TRUE(token.isValid);
      backend_.materialize(token, OutputRequest::EventWeights);
      const auto& weights = backend_.getEventWeightsHostView(token);
      const double a = tabulated.evalResponse(buffer), b = kriged.evalResponse(buffer);
      ASSERT_EQ(weights.size(), 2);
      EXPECT_DOUBLE_EQ(weights[0], 2. * a * b);
      EXPECT_DOUBLE_EQ(weights[1], 3. * a);
      EXPECT_DOUBLE_EQ(backend_.getHistogramSumsHostView(token)[0], weights[0] + weights[1]);
      EXPECT_DOUBLE_EQ(backend_.getHistogramSumSquaresHostView(token)[0], weights[0]*weights[0] + weights[1]*weights[1]);
    };
    check();
    check();
    first[2] = 8.;
    ++inputs.externalWeights[0].generation;
    check();
    second[0] = -2.;
    ++inputs.externalWeights[1].generation;
    check();
    backend_.build(model);
    check();
    inputs.externalWeights[0].count = 1;
    EXPECT_THROW(backend_.requestPropagation(inputs), std::runtime_error);
  }
}
TEST(BackendTable, CpuReadsUpdatedTablesWithoutParameterChanges) {
  Backends::CpuBackend backend;
  checkUpdatedTableValues(backend);
}
#ifdef GUNDAM_TEST_MPS_AVAILABLE
TEST(BackendTable, MetalReadsUpdatedTablesWithoutParameterChanges) {
  for( const char* mode : {"cachedResponses", "recomputeResponses"} ){
    Backends::MpsBackend backend;
    if( !backend.getCapabilities().supportsGpu ) GTEST_SKIP() << "Metal unavailable";
    backend.configure(ConfigReader(JsonType{{"propagationMode", mode}}));
    checkUpdatedTableValues(backend);
  }
}
#endif

TEST(BackendTable, ObservableCacheTracksTableGenerations) {
  using namespace Backends;
  auto view = makeTableModel();
  auto& model = view.propagation;
  model.variableDialIndices = {0, 1};
  EventStateCache cache;
  cache.build(model);
  std::vector<double> first{1., 2., 3.}, second{4., 5.};
  PropagationInputs inputs;
  inputs.externalWeights = {{first.data(), first.size(), 0, 1}, {second.data(), second.size(), 3, 1}};
  cache.prepare(inputs);
  EXPECT_TRUE(cache.needsUpdate(inputs));
  cache.commit(inputs, {});
  cache.prepare(inputs);
  EXPECT_FALSE(cache.needsUpdate(inputs));
  ++inputs.externalWeights[1].generation;
  cache.prepare(inputs);
  EXPECT_TRUE(cache.needsUpdate(inputs));
  cache.commit(inputs, {});
  cache.prepare(inputs);
  EXPECT_FALSE(cache.needsUpdate(inputs));
  ++inputs.externalWeights[0].generation;
  cache.prepare(inputs);
  EXPECT_TRUE(cache.needsUpdate(inputs));
}

TEST(BackendTable, CpuObservableMigratesWhenTableChanges) {
  using namespace Backends;
  auto view = makeTableModel();
  auto& model = view.propagation;
  model.totalBins = 2;
  model.hasVariableResponses = true;
  model.variableValues = {0.};
  model.events.resize(1);
  model.eventWeights.resize(1);
  auto& event = model.events[0];
  event.samplePosition = 0;
  event.variableCount = event.variableResponseCount = 1;
  BackendVariableResponseDescriptor response;
  response.overrideDial = 0;
  response.multiplyDialCount = 1;
  model.variableResponses = {response};
  model.variableDialIndices = {1};
  SampleView sample;
  sample.useDynamicBinning = true;
  sample.binCount = 2;
  sample.binning.variableNames = {"X"};
  sample.binning.intervals = {{0, 0., 5.}, {0, 5., 20.}};
  sample.binning.bins = {{0, 0, 1}, {1, 1, 1}};
  model.samples = {sample};
  CpuBackend backend;
  backend.build(view);
  std::vector<double> first{10., 2., 4.}, second{3., 5.};
  PropagationInputs inputs;
  inputs.externalWeights = {{first.data(), first.size(), 0, 1}, {second.data(), second.size(), 3, 1}};
  auto check = [&](double expected_, int bin_){
    const auto token = backend.requestPropagation(inputs);
    ASSERT_TRUE(token.isValid);
    const auto& state = backend.getEventStateHostView(token);
    ASSERT_EQ(state.variables.size(), 1);
    EXPECT_DOUBLE_EQ(state.variables[0], expected_);
    EXPECT_EQ(state.bins[0], bin_);
    const auto& sums = backend.getHistogramSumsHostView(token);
    EXPECT_DOUBLE_EQ(sums[bin_], 2. * expected_);
    EXPECT_DOUBLE_EQ(sums[1 - bin_], 0.);
  };
  check(14., 1);
  check(14., 1);
  first[2] = 0.;
  ++inputs.externalWeights[0].generation;
  check(2., 0);
  second[0] = 30.;
  ++inputs.externalWeights[1].generation;
  check(8.75, 1);
}

TEST(BackendTable, RejectsInvalidDescriptors) {
  using namespace Backends;
  CpuBackend backend;
  auto good = makeTableModel();
  auto bad = good;
  bad.propagation.dials[0].externalBlockIndex = 2;
  EXPECT_THROW(backend.build(bad), std::runtime_error);
  bad = good;
  bad.propagation.tableTerms[0].index = 3;
  EXPECT_THROW(backend.build(bad), std::runtime_error);
  bad = good;
  bad.propagation.tableTerms.pop_back();
  EXPECT_THROW(backend.build(bad), std::runtime_error);
  bad = good;
  bad.propagation.dials[0].tableTermCount = 1;
  EXPECT_THROW(backend.build(bad), std::runtime_error);
#ifdef GUNDAM_TEST_MPS_AVAILABLE
  MpsBackend mps;
  if( !mps.getCapabilities().supportsGpu ) GTEST_SKIP() << "Metal unavailable";
  EXPECT_NO_THROW(mps.build(good));
  EXPECT_THROW(mps.build(bad), std::runtime_error);
#endif
}

TEST(BackendTable, ObservableCacheDistinguishesHostSnapshotsFromEdits) {
  using namespace Backends;
  PropagationView model;
  model.parameterCount = 2;
  model.dialInputs = {{0}};
  BackendDialDescriptor dial;
  dial.inputCount = 1;
  model.dials = {dial};
  model.variableDialIndices = {0};
  model.variableInputBufferCount = 1;
  EventStateCache cache;
  cache.build(model);
  PropagationInputs inputs;
  inputs.parameters.values = {1., 2.};
  inputs.hostVariables = {.5, 1.5};
  inputs.variableInputRevisions = {1};
  cache.prepare(inputs);
  ASSERT_TRUE(cache.needsUpdate(inputs));
  cache.commit(inputs, {});
  cache.prepare(inputs);
  EXPECT_FALSE(cache.needsUpdate(inputs));
  EXPECT_FALSE(cache.hostVariablesChanged());
  inputs.parameters.values[1] = 3.; // Weight-only input, unchanged host snapshot.
  cache.prepare(inputs);
  EXPECT_FALSE(cache.needsUpdate(inputs));
  inputs.parameters.values[0] = 2.;
  cache.prepare(inputs);
  EXPECT_TRUE(cache.needsUpdate(inputs));
  EXPECT_FALSE(cache.hostVariablesChanged());
  cache.commit(inputs, {});
  inputs.hostVariables.clear(); // Omission must retain the previous host baseline.
  ++inputs.variableInputRevisions[0];
  cache.prepare(inputs);
  EXPECT_TRUE(cache.needsUpdate(inputs));
  cache.commit(inputs, {});
  inputs.hostVariables = {.5, 1.5};
  cache.prepare(inputs);
  EXPECT_FALSE(cache.needsUpdate(inputs));
  inputs.hostVariables[1] = .75;
  cache.prepare(inputs);
  EXPECT_TRUE(cache.needsUpdate(inputs));
  EXPECT_FALSE(cache.hostVariableChanged(0, inputs));
  EXPECT_TRUE(cache.hostVariableChanged(1, inputs));
  cache.commit(inputs, {});
  inputs.forceBinningUpdate = true;
  cache.prepare(inputs);
  EXPECT_TRUE(cache.needsUpdate(inputs));
  EXPECT_TRUE(cache.hostVariableChanged(0, inputs));
  cache.commit(inputs, {});
  inputs.forceBinningUpdate = false;
  cache.prepare(inputs);
  EXPECT_FALSE(cache.needsUpdate(inputs));
  cache.invalidate();
  cache.prepare(inputs);
  EXPECT_TRUE(cache.needsUpdate(inputs));
}
