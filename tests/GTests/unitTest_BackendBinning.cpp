#include "CpuBackend.h"
#ifdef __APPLE__
#include "MpsBackend.h"
#endif
#include "DeviceWeightModel.h"
#include "Histogram.h"
#include "DevicePropagationModel.h"
#include "CompactSpline.h"
#include "UniformSpline.h"
#include "MonotonicSpline.h"
#include "GeneralSpline.h"
#include "Graph.h"

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

  void compareWithHistogram(Backends::Backend& backend, bool smooth_, bool relative_, bool irregular_, bool broad_) {
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
    backend.build(view);
    PropagationInputs inputs;
    inputs.hostVariables = {0, 0, 0};
    std::mt19937 random(724);
    std::uniform_real_distribution<double> x(-1.5, 5.5), y(-1., 3.);
    const double tolerance = backend.getCapabilities().supportsGpu ? 2e-5 : 1e-12;
    for( int point = 0; point < (backend.getCapabilities().supportsGpu ? 50 : 250); ++point ){
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
        EXPECT_NEAR(actual[i], sums[i], tolerance) << point << ":" << i;
        EXPECT_NEAR(actualSquares[i], squares[i], tolerance) << point << ":" << i;
      }
      EXPECT_EQ(backend.getEventStateHostView(token).bins[0], histogram.findBinIndex(hostEvent));
      if( smooth_ ){
        EXPECT_NEAR(backend.getOutsideSumsHostView(token)[0], -2. * expected.outsideFraction, tolerance);
        EXPECT_NEAR(backend.getOutsideSumSquaresHostView(token)[0], 4. * expected.outsideFraction * expected.outsideFraction, tolerance);
      }
    }
  }
}

class BinningBackend : public testing::TestWithParam<bool> {
protected:
  std::unique_ptr<Backends::Backend> backend;
  void SetUp() override {
#ifdef __APPLE__
    if( GetParam() ){ backend = std::make_unique<Backends::MpsBackend>(); }
    else
#endif
    { backend = std::make_unique<Backends::CpuBackend>(); }
  }
};
INSTANTIATE_TEST_SUITE_P(Cpu, BinningBackend, testing::Values(false));
#ifdef __APPLE__
INSTANTIATE_TEST_SUITE_P(Metal, BinningBackend, testing::Values(true));
#endif

TEST_P(BinningBackend, MatchesLegacyGeometryAndResponse) {
  for( bool smooth : {false, true} ){
    for( bool irregular : {false, true} ){
      compareWithHistogram(*backend, smooth, false, irregular, false);
      if( smooth ){
        compareWithHistogram(*backend, true, true, irregular, false);
        compareWithHistogram(*backend, true, false, irregular, true);
      }
    }
  }
}

TEST_P(BinningBackend, GridSearchAndCacheInvalidation) {
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
  auto& backend = *this->backend;
  backend.build(view);
  PropagationInputs inputs;
  inputs.parameters.values = {1., 2.};
  auto token = backend.requestPropagation(inputs);
  EXPECT_EQ(backend.getLastTimingSummary().rebinnedEventCount, 2);
  EXPECT_LE(backend.getLastTimingSummary().binCandidateCount, backend.getCapabilities().supportsGpu ? 8 : 2);
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
  DeviceWeightModel packed;
  BackendTimingSummary timing;
  std::string reason;
  EXPECT_FALSE(packDeviceWeightModel(model, true, packed, timing, reason, "test"));
  EXPECT_FALSE(reason.empty());
}

TEST(BinningIndex, OverlappingStrictBinsKeepFirstMatch) {
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

TEST_P(BinningBackend, NonfiniteResponseCanBeRetried) {
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
  auto& backend = *this->backend;
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

TEST_P(BinningBackend, ExternalVariableResponseUsesPublishedGeneration) {
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
  auto& backend = *this->backend;
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


TEST_P(BinningBackend, VariableAndWeightDialsShareSemantics) {
  using namespace Backends;
  EngineView view;
  auto& model = view.propagation;
  model.parameterCount = 1;
  model.dialInputs = {{0, true, -3., 6.}};
  model.totalBins = 2;
  model.hasVariableResponses = true;
  model.hasReweightCap = true;
  model.maxReweight = 2.5;
  model.variableValues = {.6};
  model.externalWeightBlocks = {{0, 1}};
  SampleView sample;
  sample.useDynamicBinning = true;
  sample.binCount = 2;
  sample.binning.variableNames = {"X"};
  sample.binning.intervals = {{0, 0., 1., 5., 5.}, {0, 1., 2., 5., 5.}};
  sample.binning.bins = {{0, 0, 1}, {1, 1, 1}};
  sample.binning.isSmoothed = true;
  model.samples = {sample};
  BackendDialDescriptor norm;
  norm.inputCount = 1;
  norm.hasMinResponse = true;
  norm.minResponse = -.75;
  model.dials.push_back(norm);
  auto append = [&](BackendDialType type_, const std::vector<double>& payload_){
    BackendDialDescriptor dial;
    dial.type = type_;
    dial.inputCount = type_ == BackendDialType::Shift ? 0 : 1;
    dial.payloadOffset = model.dialPayloads.size();
    dial.payloadSize = payload_.size();
    dial.hasMaxResponse = true;
    dial.maxResponse = 1.75;
    model.dialPayloads.insert(model.dialPayloads.end(), payload_.begin(), payload_.end());
    model.dials.push_back(dial);
  };
  append(BackendDialType::Shift, {1.1});
  std::vector<DialUtils::DialPoint> points;
  for( int x = -2; x <= 2; ++x ){ points.push_back({double(x), 1 + .1*x + .05*x*x, .1 + .1*x}); }
  CompactSpline compact; compact.buildDial(points);
  UniformSpline uniform; uniform.buildDial(points);
  MonotonicSpline monotonic; monotonic.buildDial(points);
  GeneralSpline general; general.buildDial(points);
  Graph graph; graph.buildDial(points);
  append(BackendDialType::CompactSpline, compact.getDialData());
  append(BackendDialType::UniformSpline, uniform.getDialData());
  append(BackendDialType::MonotonicSpline, monotonic.getDialData());
  append(BackendDialType::GeneralSpline, general.getDialData());
  append(BackendDialType::Graph, graph.getDialData());
  BackendDialDescriptor external;
  external.type = BackendDialType::ExternalWeight;
  model.dials.push_back(external);
  model.eventDialIndices = {0, 1, 2, 3, 4, 5, 6, 7};
  model.variableDialIndices = {1, 2, 3, 4, 5, 6, 7};
  BackendVariableResponseDescriptor target;
  target.overrideDial = 0;
  target.multiplyDialCount = 6;
  target.firstAddDial = 6;
  target.addDialCount = 1;
  model.variableResponses = {target};
  EventView event;
  event.samplePosition = 0;
  event.variableCount = event.variableResponseCount = 1;
  event.weight.dialCount = model.eventDialIndices.size();
  event.weight.baseWeight = -2.;
  model.events = {event};
  double externalValue{.2};
  PropagationInputs inputs;
  inputs.externalWeights = {{&externalValue, 1, 0, 1}};
  CpuBackend reference;
  for( bool extrapolate : {false, true} ){
    for( auto& dial : model.dials ){ dial.allowExtrapolation = extrapolate; }
    reference.build(view);
    backend->build(view);
    for( double parameter : {-8., -2.7, -.25, .6, 2.7, 5., 1e9} ){
      inputs.parameters.values = {parameter};
      auto expected = reference.requestPropagation(inputs);
      auto actual = backend->requestPropagation(inputs);
      backend->materialize(actual, OutputRequest::EventWeights);
      const double tolerance = backend->getCapabilities().supportsGpu ? 2e-5 : 1e-12;
      EXPECT_NEAR(backend->getEventWeightsHostView(actual)[0], reference.getEventWeightsHostView(expected)[0], tolerance);
      EXPECT_NEAR(backend->getEventStateHostView(actual).variables[0], reference.getEventStateHostView(expected).variables[0], tolerance);
      for( int bin = 0; bin < 2; ++bin ){
        EXPECT_NEAR(backend->getHistogramSumsHostView(actual)[bin], reference.getHistogramSumsHostView(expected)[bin], tolerance);
        EXPECT_NEAR(backend->getHistogramSumSquaresHostView(actual)[bin], reference.getHistogramSumSquaresHostView(expected)[bin], tolerance);
      }
    }
  }
}

TEST(DevicePropagationModel, RejectsCollapsedGeometry) {
  Backends::PropagationView model;
  Backends::SampleView sample;
  sample.binning.variableNames = {"X"};
  sample.binning.intervals = {{0, 1e30, std::nextafter(1e30, INFINITY), 0., 0.}};
  sample.binning.bins = {{0, 0, 1}};
  model.samples = {sample};
  Backends::DevicePropagationModel packed;
  EXPECT_THROW(packed.build(model), std::runtime_error);
}


TEST_P(BinningBackend, ResolvesNeighborsOfAnExactBoundary) {
  using namespace Backends;
  EngineView view;
  auto& model = view.propagation;
  model.variableValues = {.5};
  model.totalBins = 2;
  SampleView sample;
  sample.useDynamicBinning = sample.readHostVariables = true;
  sample.binCount = 2;
  sample.binning.variableNames = {"X"};
  sample.binning.intervals = {{0, 0., 1., 0., 0.}, {0, 1., 2., 0., 0.}};
  sample.binning.bins = {{0, 0, 1}, {1, 1, 1}};
  model.samples = {sample};
  EventView event;
  event.samplePosition = 0;
  event.variableCount = 1;
  model.events = {event};
  backend->build(view);
  PropagationInputs inputs;
  for( double value : {std::nextafter(1., 0.), 1., std::nextafter(1., 2.)} ){
    inputs.hostVariables = {value};
    auto token = backend->requestPropagation(inputs);
    EXPECT_EQ(backend->getEventStateHostView(token).bins[0], value < 1. ? 0 : 1);
    EXPECT_DOUBLE_EQ(backend->getHistogramSumsHostView(token)[value < 1. ? 0 : 1], 1.);
  }
}

TEST_P(BinningBackend, EmptyDynamicSampleProducesEmptyHistograms) {
  using namespace Backends;
  EngineView view;
  SampleView sample;
  sample.useDynamicBinning = true;
  sample.binCount = 2;
  sample.binning.variableNames = {"X"};
  sample.binning.intervals = {{0, 0., 1., 5., 5.}, {0, 1., 2., 5., 5.}};
  sample.binning.bins = {{0, 0, 1}, {1, 1, 1}};
  sample.binning.isSmoothed = true;
  view.propagation.samples = {sample};
  view.propagation.totalBins = 2;
  backend->build(view);
  auto token = backend->requestPropagation({});
  EXPECT_EQ(backend->getHistogramSumsHostView(token), std::vector<double>({0., 0.}));
  EXPECT_EQ(backend->getOutsideSumsHostView(token), std::vector<double>({0.}));
  EXPECT_TRUE(backend->getEventStateHostView(token).bins.empty());
}
