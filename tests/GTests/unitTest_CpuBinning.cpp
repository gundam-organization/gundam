#include "CpuBackend.h"
#include "DeviceBackendModel.h"
#include "Histogram.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <random>
#include <sstream>

namespace {
  Histogram makeHistogram(bool smooth_, bool relative_, bool irregular_, bool broad_ = false) {
    Histogram histogram;
    std::stringstream config;
    config << R"({"binList":[)";
    bool first{true};
    for( int category = 0; category < 2; ++category ){
      for( int y = 0; y < 2; ++y ){
        const double edges[]{0, 0.5, 1.5, 3., 4.};
        for( int x = 0; x < 4; ++x ){
          if( irregular_ and x == 3 and y == 1 ){ continue; }
          if( not first ){ config << ','; }
          first = false;
          config << R"({"edgesList":[{"name":"X","bounds":[)" << edges[x] << ',' << edges[x+1]
                 << R"(]},{"name":"Y","bounds":[)" << y << ',' << y+1
                 << R"(]},{"name":"category","value":)" << category << "}]}";
        }
      }
    }
    config << "]}";
    histogram.build(ConfigReader(JsonType::parse(config.str())));
    if( smooth_ ){
      auto response = relative_ ? R"({"isEnabled":true,"variables":[{"name":"X","widthFraction":0.4},{"name":"Y","width":0.3}]})"
          : (broad_ ? R"({"isEnabled":true,"variables":[{"name":"X","width":1.2},{"name":"Y","width":0.8}]})"
                    : R"({"isEnabled":true,"variables":[{"name":"X","width":0.2},{"name":"Y","width":0.3}]})");
      histogram.configureBinningResponse(ConfigReader(JsonType::parse(response)));
    }
    return histogram;
  }

  void compareWithHistogram(bool smooth_, bool relative_, bool irregular_, bool broad_) {
    using namespace Backends;
    auto histogram = makeHistogram(smooth_, relative_, irregular_, broad_);
    EngineView view;
    auto& model = view.propagation;
    SampleView sample;
    sample.sampleIndex = 0;
    sample.binCount = histogram.getNbBins();
    sample.useDynamicBinning = true;
    sample.readHostVariables = true;
    sample.binning = histogram.exportBinningLayout();
    sample.variableNames = sample.binning.variableNames;
    model.samples = {sample};
    model.totalBins = sample.binCount;
    model.variableValues = {0, 0, 0};
    EventView event;
    event.samplePosition = 0;
    event.variableCount = 3;
    event.weight.baseWeight = -2.; // Fractional sum-of-squares with negative weights.
    model.events = {event};
    Event hostEvent;
    hostEvent.getVariables().setVarNameList(sample.variableNames);
    CpuBackend backend;
    backend.build(view);
    PropagationInputs inputs;
    inputs.hostVariables = {0, 0, 0};
    std::mt19937 random(724);
    std::uniform_real_distribution<double> x(-1.5, 5.5), y(-1., 3.);
    for( int point = 0; point < 250; ++point ){
      double values[]{x(random), y(random), double(point % 3)};
      // Include exact internal edges and the outer domain boundary.
      if( point < 5 ){ values[0] = point * 0.5; values[1] = 1.; values[2] = 0.; }
      for( std::size_t i = 0; i < sample.variableNames.size(); ++i ){
        const auto& name = sample.variableNames[i];
        double value = values[name == "X" ? 0 : (name == "Y" ? 1 : 2)];
        inputs.hostVariables[i] = value;
        hostEvent.getVariables().getVarList()[i].set(value);
      }
      auto token = backend.requestPropagation(inputs);
      auto expected = histogram.evaluateEventBinningResponse(hostEvent);
      std::vector<double> sums(model.totalBins, 0.), squares(model.totalBins, 0.);
      for( const auto& fraction : expected.binFractions ){
        sums[fraction.binIndex] = -2. * fraction.fraction;
        squares[fraction.binIndex] = 4. * fraction.fraction * fraction.fraction;
      }
      const auto& actual = backend.getHistogramSumsHostView(token);
      const auto& actualSquares = backend.getHistogramSumSquaresHostView(token);
      ASSERT_EQ(actual.size(), sums.size());
      for( std::size_t i = 0; i < sums.size(); ++i ){
        EXPECT_NEAR(actual[i], sums[i], 1e-12) << point << ":" << i;
        EXPECT_NEAR(actualSquares[i], squares[i], 1e-12) << point << ":" << i;
      }
      EXPECT_EQ(backend.getEventStateHostView(token).bins[0], histogram.findBinIndex(hostEvent));
      if( smooth_ ){
        EXPECT_NEAR(backend.getOutsideSumsHostView(token)[0], -2. * expected.outsideFraction, 1e-12);
        EXPECT_NEAR(backend.getOutsideSumSquaresHostView(token)[0], 4. * expected.outsideFraction * expected.outsideFraction, 1e-12);
      }
    }
  }
}

TEST(CpuBinning, MatchesLegacyGeometryAndResponse) {
  for( bool smooth : {false, true} ){
    for( bool irregular : {false, true} ){
      compareWithHistogram(smooth, false, irregular, false);
      if( smooth ){
        compareWithHistogram(true, true, irregular, false);
        compareWithHistogram(true, false, irregular, true);
      }
    }
  }
}

TEST(CpuBinning, GridSearchAndCacheInvalidation) {
  using namespace Backends;
  Histogram histogram;
  std::stringstream config;
  config << R"({"binningDefinition":[{"name":"X","edges":[)";
  for( int edge = 0; edge <= 128; ++edge ){ if( edge ){ config << ','; } config << edge; }
  config << "]}]}";
  histogram.build(ConfigReader(JsonType::parse(config.str())));
  EngineView view;
  auto& model = view.propagation;
  SampleView sample;
  sample.useDynamicBinning = true;
  sample.binCount = 128;
  sample.binning = histogram.exportBinningLayout();
  model.samples = {sample};
  model.totalBins = 128;
  model.parameterCount = 2;
  model.hasVariableResponses = true;
  model.variableValues = {.25, 64.25};
  model.dialInputs = {{0}, {1}};
  BackendDialDescriptor norm;
  norm.inputCount = 1;
  model.dials = {norm, norm};
  model.dials[1].firstInput = 1;
  model.variableDialIndices = {0};
  for( std::size_t i = 0; i < 2; ++i ){
    EventView event;
    event.samplePosition = 0;
    event.firstVariable = i;
    event.variableCount = 1;
    event.firstVariableResponse = i;
    event.variableResponseCount = 1;
    event.resultIndex = i;
    event.weight.firstDial = 0;
    event.weight.dialCount = 1;
    model.events.emplace_back(event);
    BackendVariableResponseDescriptor target;
    target.variableIndex = i;
    target.referenceValue = model.variableValues[i];
    target.multiplyDialCount = 1;
    model.variableResponses.emplace_back(target);
  }
  model.eventDialIndices = {1};
  CpuBackend backend;
  backend.build(view);
  PropagationInputs inputs;
  inputs.parameters.values = {1., 2.};
  auto token = backend.requestPropagation(inputs);
  EXPECT_EQ(backend.getLastTimingSummary().rebinnedEventCount, 2);
  EXPECT_EQ(backend.getLastTimingSummary().binCandidateCount, 2);
  EXPECT_DOUBLE_EQ(backend.getHistogramSumsHostView(token)[64], 2.);
  inputs.parameters.values[1] = 3.;
  token = backend.requestPropagation(inputs);
  EXPECT_EQ(backend.getLastTimingSummary().rebinnedEventCount, 0);
  EXPECT_DOUBLE_EQ(backend.getHistogramSumsHostView(token)[64], 3.);
  inputs.parameters.values[0] = 2.;
  token = backend.requestPropagation(inputs);
  EXPECT_EQ(backend.getLastTimingSummary().rebinnedEventCount, 2);
  EXPECT_EQ(backend.getEventStateHostView(token).bins[1], -1);
  backend.requestPropagation(inputs);
  EXPECT_EQ(backend.getLastTimingSummary().rebinnedEventCount, 0);
  inputs.hostVariables = {.5, 1.5};
  inputs.forceBinningUpdate = true;
  token = backend.requestPropagation(inputs);
  EXPECT_EQ(backend.getLastTimingSummary().rebinnedEventCount, 2);
  EXPECT_EQ(backend.getEventStateHostView(token).bins[1], 1);
  backend.build(view);
  inputs.forceBinningUpdate = false;
  token = backend.requestPropagation(inputs);
  EXPECT_EQ(backend.getLastTimingSummary().rebinnedEventCount, 2);
  EXPECT_EQ(backend.getEventStateHostView(token).bins[1], -1);
  DevicePackedModel packed;
  BackendTimingSummary timing;
  std::string reason;
  EXPECT_FALSE(packDeviceBackendModel(model, true, packed, timing, reason, "test"));
  EXPECT_FALSE(reason.empty());
}

TEST(CpuBinning, OverlappingStrictBinsKeepFirstMatch) {
  // Public compiled layouts can also be supplied directly by backend clients.
  BinningLayout layout;
  layout.variableNames = {"X"};
  layout.intervals = {{0, 0., 2., 0., 0.}, {0, 1., 3., 0., 0.}};
  layout.bins = {{0, 0, 1}, {1, 1, 1}};
  BinningIndex index;
  index.build(layout);
  double value = 1.5;
  std::vector<std::size_t> candidates;
  index.findCandidates(&value, candidates);
  ASSERT_EQ(candidates.size(), 2);
  EXPECT_EQ(candidates.front(), 0);
}

TEST(CpuBinning, NonfiniteResponseCanBeRetried) {
  using namespace Backends;
  auto histogram = makeHistogram(true, true, true);
  EngineView view;
  auto& model = view.propagation;
  SampleView sample;
  sample.binning = histogram.exportBinningLayout();
  sample.binCount = histogram.getNbBins();
  sample.useDynamicBinning = sample.readHostVariables = true;
  model.samples = {sample};
  model.totalBins = sample.binCount;
  model.variableValues = {.25, .5, 0.};
  EventView event;
  event.samplePosition = 0;
  event.variableCount = 3;
  model.events = {event};
  CpuBackend backend;
  backend.build(view);
  PropagationInputs inputs;
  inputs.hostVariables = model.variableValues;
  auto first = backend.requestPropagation(inputs);
  const auto expected = backend.getHistogramSumsHostView(first);
  inputs.hostVariables[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(backend.requestPropagation(inputs), std::runtime_error);
  inputs.hostVariables = model.variableValues;
  auto recovered = backend.requestPropagation(inputs);
  EXPECT_EQ(backend.getHistogramSumsHostView(recovered), expected);
}

TEST(CpuBinning, ExternalVariableResponseUsesPublishedGeneration) {
  using namespace Backends;
  EngineView view;
  auto& model = view.propagation;
  SampleView sample;
  sample.useDynamicBinning = true;
  sample.binCount = 2;
  sample.binning.variableNames = {"X"};
  sample.binning.intervals = {{0, 0., 1., 0., 0.}, {0, 1., 2., 0., 0.}};
  sample.binning.bins = {{0, 0, 1}, {1, 1, 1}};
  model.samples = {sample};
  model.totalBins = 2;
  model.hasVariableResponses = true;
  model.variableValues = {.25};
  model.externalWeightBlocks = {{0, 1}};
  BackendDialDescriptor external;
  external.type = BackendDialType::ExternalWeight;
  model.dials = {external};
  BackendVariableResponseDescriptor target;
  target.overrideDial = 0;
  model.variableResponses = {target};
  EventView event;
  event.samplePosition = 0;
  event.variableCount = event.variableResponseCount = 1;
  model.events = {event};
  CpuBackend backend;
  backend.build(view);
  double value{.25};
  PropagationInputs inputs;
  inputs.externalWeights = {{&value, 1, 0, 1}};
  auto token = backend.requestPropagation(inputs);
  EXPECT_DOUBLE_EQ(backend.getHistogramSumsHostView(token)[0], 1.);
  backend.requestPropagation(inputs);
  EXPECT_EQ(backend.getLastTimingSummary().rebinnedEventCount, 0);
  value = 1.25;
  ++inputs.externalWeights[0].generation;
  token = backend.requestPropagation(inputs);
  EXPECT_EQ(backend.getLastTimingSummary().rebinnedEventCount, 1);
  EXPECT_DOUBLE_EQ(backend.getHistogramSumsHostView(token)[1], 1.);
}
