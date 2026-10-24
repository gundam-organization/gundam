#include "CpuBackend.h"
#ifdef __APPLE__
#include "MpsBackend.h"
#endif
#include "DeviceWeightModel.h"
#include "DevicePropagationModel.h"
#include "ResponseSmoothingPacking.h"
#include "DialResponseSupervisor.h"
#include "CompactSpline.h"
#include "UniformSpline.h"
#include "MonotonicSpline.h"
#include "GeneralSpline.h"
#include "Graph.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <tuple>

namespace {
  using namespace Backends;

  EngineView makeModel(const DialResponseSupervisor& supervisor_, bool variables_, int repeats_) {
    EngineView view;
    auto& model = view.propagation;
    model.parameterCount = 1;
    model.totalBins = 1;
    model.dialInputs = {{0, false, 0., 0.}};
    model.externalWeightBlocks = {{0, 1}};
    model.hasVariableResponses = variables_;
    SampleView sample;
    sample.sampleIndex = sample.binOffset = 0;
    sample.binCount = 1;
    sample.useDynamicBinning = variables_;
    sample.binning.variableNames = {"X"};
    sample.binning.intervals = {{0, -10., 10., 0., 0.}};
    sample.binning.bins = {{0, 0, 1}};
    model.samples = {sample};
    auto append = [&](BackendDialType type_, const std::vector<double>& payload_){
      BackendDialDescriptor dial;
      dial.type = type_;
      dial.inputCount = type_ == BackendDialType::Shift or type_ == BackendDialType::ExternalWeight ? 0 : 1;
      dial.allowExtrapolation = true;
      dial.payloadOffset = model.dialPayloads.size();
      dial.payloadSize = payload_.size();
      dial.hasMinResponse = not std::isnan(supervisor_.getMinResponse());
      dial.hasMaxResponse = not std::isnan(supervisor_.getMaxResponse());
      auto& smoothing = dial.smoothing;
      smoothing.minResponse = supervisor_.getMinResponse();
      smoothing.maxResponse = supervisor_.getMaxResponse();
      smoothing.minWidth = supervisor_.getMinSmoothingWidth();
      smoothing.maxWidth = supervisor_.getMaxSmoothingWidth();
      smoothing.minInverseWidth = smoothing.minWidth > 0. ? .5 / smoothing.minWidth : 0.;
      smoothing.maxInverseWidth = smoothing.maxWidth > 0. ? .5 / smoothing.maxWidth : 0.;
      dial.minResponse = smoothing.minResponse + smoothing.minWidth;
      dial.maxResponse = smoothing.maxResponse - smoothing.maxWidth;
      model.dialPayloads.insert(model.dialPayloads.end(), payload_.begin(), payload_.end());
      model.dials.push_back(dial);
    };
    append(BackendDialType::Norm, {});
    append(BackendDialType::Shift, {0.}); // Smoothing must also apply to folded constants.
    append(BackendDialType::ExternalWeight, {});
    std::vector<DialUtils::DialPoint> points;
    for( int x = -2; x <= 4; ++x ){ points.push_back({double(x), double(x), 1.}); }
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
    for( DeviceIndex dial = 0; dial < model.dials.size(); ++dial ){
      for( int repeat = 0; repeat < repeats_; ++repeat ){
        EventView event;
        BackendEventWeightDescriptor eventWeight;
        event.sampleIndex = event.binIndex = event.globalBinIndex = 0;
        event.samplePosition = 0;
        event.resultIndex = model.events.size();
        eventWeight.firstDial = model.eventDialIndices.size();
        eventWeight.dialCount = 1;
        model.eventDialIndices.push_back(dial);
        if( variables_ ){
          event.firstVariable = model.variableValues.size();
          event.variableCount = event.variableResponseCount = 1;
          event.firstVariableResponse = model.variableResponses.size();
          model.variableValues.push_back(.5);
          BackendVariableResponseDescriptor target;
          target.variableIndex = event.firstVariable;
          target.overrideDial = dial;
          model.variableResponses.push_back(target);
        }
        model.events.push_back(event);
        model.eventWeights.push_back(eventWeight);
      }
    }
    return view;
  }

  class BackendResponseSmoothing : public testing::TestWithParam<std::tuple<bool, bool, bool>> {};
}

TEST_P(BackendResponseSmoothing, MatchesOriginalSupervisor) {
  const auto [gpu, variables, cache] = GetParam();
  std::unique_ptr<Backend> backend;
  if( gpu ){
#ifdef __APPLE__
    backend = std::make_unique<MpsBackend>();
    if( not backend->getCapabilities().supportsGpu ){ GTEST_SKIP() << "Metal unavailable"; }
#else
    GTEST_SKIP() << "Metal requires macOS";
#endif
  }
  else{ backend = std::make_unique<CpuBackend>(); }
  if( gpu ){ backend->configure(ConfigReader(JsonType{{"propagationMode", cache ? "cachedResponses" : "recomputeResponses"}})); }
  const int repeats = kDeviceCachedDialReuseThreshold;
  for( int limits = 0; limits < 5; ++limits ){
    SCOPED_TRACE(limits);
    DialResponseSupervisor supervisor;
    if( limits != 2 ){ supervisor.setMinResponse(0.); }
    if( limits != 3 ){ supervisor.setMaxResponse(2.); }
    if( limits != 0 ){ supervisor.setSmoothingWidths(limits == 4 ? .5 : .4, limits == 4 ? 1.5 : .1); }
    auto view = makeModel(supervisor, variables, repeats);
    backend->build(view);
    double external{0};
    PropagationInputs inputs;
    inputs.externalWeights = {{&external, 1, 0, 1}};
    for( double value : {-1., -.4, -.2, 0., .2, .4, .8, 1., 1.6, 1.9, 2., 2.1, 2.5} ){
      SCOPED_TRACE(value);
      inputs.parameters.values = {value};
      external = value;
      ++inputs.externalWeights[0].generation;
      for( int repeat = 0; repeat < 2; ++repeat ){
        auto token = backend->requestPropagation(inputs);
        backend->wait(token);
        if( gpu ){ ASSERT_EQ(backend->getStatus(token).eventWeights, OutputState::ReadyOnDevice); }
        backend->materialize(token, OutputRequest::EventWeights);
        backend->materialize(token, OutputRequest::Histograms);
        if( variables ){ backend->materialize(token, OutputRequest::EventState); }
        const auto& weights = backend->getEventWeightsHostView(token);
        ASSERT_EQ(weights.size(), view.propagation.events.size());
        double sum{0}, sumSquares{0};
        for( std::size_t i = 0; i < weights.size(); ++i ){
          double expected = supervisor.process(i / repeats == 1 ? 0. : value);
          const double tolerance = gpu ? 3e-6 : 1e-12;
          EXPECT_NEAR(weights[i], expected, tolerance) << "event " << i;
          if( expected == 0. and value == -1. ){ EXPECT_EQ(weights[i], 0.); }
          if( expected == 2. and value == 2.5 ){ EXPECT_EQ(weights[i], 2.); }
          if( variables ){
            EXPECT_NEAR(backend->getEventStateHostView(token).variables[i], expected, tolerance);
          }
          sum += expected;
          sumSquares += expected * expected;
        }
        EXPECT_NEAR(backend->getHistogramSumsHostView(token)[0], sum, gpu ? 2e-4 : 1e-10);
        EXPECT_NEAR(backend->getHistogramSumSquaresHostView(token)[0], sumSquares, gpu ? 4e-4 : 1e-10);
        if( gpu and not variables and not cache ){
          EXPECT_EQ(backend->getLastTimingSummary().cachedDialCount, 5);
        }
      }
    }
  }
}

TEST(BackendResponseSmoothingPacking, RejectsCollapsedFloatTransition) {
  DialResponseSupervisor supervisor;
  supervisor.setMinResponse(1.);
  supervisor.setSmoothingWidths(1e-9, 1.);
  auto view = makeModel(supervisor, false, 1);
  DeviceNormDialOccurrence packed;
  EXPECT_THROW(packResponseLimits(view.propagation.dials[0], packed, [](double value_){ return float(value_); }), std::runtime_error);
  DeviceWeightModel weights;
  BackendTimingSummary timing;
  std::string reason;
  EXPECT_FALSE(packDeviceWeightModel(view.propagation, true, weights, timing, reason, "test"));
  EXPECT_NE(reason.find("smoothing"), std::string::npos);
  // Observable propagation retains this transition with its double-float scalars.
  DevicePropagationModel observable;
  EXPECT_NO_THROW(observable.build(view.propagation));
}

#ifdef __APPLE__
TEST(MpsPropagationMode, DefaultCachesScalarResponsesAndExplicitModeRecomputes) {
  DialResponseSupervisor supervisor;
  auto view = makeModel(supervisor, false, kDeviceCachedDialReuseThreshold);
  for( const std::string mode : {"", "cachedResponses", "recomputeResponses"} ){
    SCOPED_TRACE(mode);
    MpsBackend backend;
    if( backend.getCapabilities().deviceName == "Metal unavailable" ){ GTEST_SKIP(); }
    if( not mode.empty() ){ backend.configure(ConfigReader(JsonType{{"propagationMode", mode}})); }
    backend.build(view);
    auto diagnostic = JsonType::parse(backend.getDiagnostics());
    const bool cached = mode != "recomputeResponses";
    EXPECT_EQ(diagnostic["propagationMode"], cached ? "cachedResponses" : "recomputeResponses");
    EXPECT_EQ(diagnostic["path"], cached ? "eventPropagation" : "packedWeights");
    double external{.8};
    PropagationInputs inputs;
    inputs.parameters.values = {.8};
    inputs.externalWeights = {{&external, 1, 0, 1}};
    std::vector<double> previous;
    for( int repeat = 0; repeat < 2; ++repeat ){
      auto token = backend.requestPropagation(inputs);
      backend.materialize(token, OutputRequest::EventWeights);
      const auto& weights = backend.getEventWeightsHostView(token);
      if( repeat ){ EXPECT_EQ(weights, previous); }
      previous = weights;
    }
    if( cached ){
      auto state = JsonType::parse(backend.getDiagnosticEventPropagationState());
      EXPECT_EQ(state["evaluatedDials"], 0);
      EXPECT_EQ(state["updatedEventWeights"], false);
    }
  }
}

TEST(MpsPropagationMode, RejectsUnknownModes) {
  MpsBackend backend;
  EXPECT_ANY_THROW(backend.configure(ConfigReader(JsonType{{"propagationMode", "auto"}})));
}

namespace {
  void compareMpsWithCpu(MpsBackend& backend_, CpuBackend& reference_, const PropagationInputs& inputs_) {
    auto token = backend_.requestPropagation(inputs_);
    ASSERT_EQ(backend_.getStatus(token).eventWeights, OutputState::ReadyOnDevice);
    backend_.materialize(token, OutputRequest::EventWeights);
    backend_.materialize(token, OutputRequest::Histograms);
    auto referenceToken = reference_.requestPropagation(inputs_);
    auto compare = [](const std::vector<double>& actual_, const std::vector<double>& expected_){
      ASSERT_EQ(actual_.size(), expected_.size());
      for( std::size_t i = 0; i < actual_.size(); ++i ){
        EXPECT_NEAR(actual_[i], expected_[i], 3e-6 * std::max(1., std::abs(expected_[i]))) << "index " << i;
      }
    };
    compare(backend_.getEventWeightsHostView(token), reference_.getEventWeightsHostView(referenceToken));
    compare(backend_.getHistogramSumsHostView(token), reference_.getHistogramSumsHostView(referenceToken));
    compare(backend_.getHistogramSumSquaresHostView(token), reference_.getHistogramSumSquaresHostView(referenceToken));
  }
}

TEST(MpsOrderedSplinePayloads, PreservesRaggedEventsAndDistinctPayloads) {
  MpsBackend backend;
  CpuBackend reference;
  if( backend.getCapabilities().deviceName == "Metal unavailable" ){ GTEST_SKIP(); }
  backend.configure(ConfigReader(JsonType{{"propagationMode", "recomputeResponses"}}));
  DialResponseSupervisor supervisor;
  supervisor.setMinResponse(0.);
  supervisor.setMaxResponse(2.);
  supervisor.setSmoothingWidths(.4, .1);
  auto view = makeModel(supervisor, false, 9); // 72 events and a partial GPU tile.
  auto& model = view.propagation;
  for( std::size_t family = 3; family < 8; ++family ){
    auto descriptor = model.dials.at(family);
    auto originalOffset = descriptor.payloadOffset;
    descriptor.payloadOffset = model.dialPayloads.size();
    for( std::size_t i = 0; i < descriptor.payloadSize; ++i ){
      auto value = model.dialPayloads.at(originalOffset + i) + .15;
      model.dialPayloads.push_back(value);
    }
    model.dials.push_back(descriptor);
  }
  model.eventDialIndices.clear();
  for( std::size_t i = 0; i < model.eventWeights.size(); ++i ){
    auto& weight = model.eventWeights[i];
    weight.firstDial = model.eventDialIndices.size();
    weight.dialCount = i % 4;
    for( std::size_t j = 0; j < weight.dialCount; ++j ){
      model.eventDialIndices.push_back((i + j) % model.dials.size());
    }
  }
  for( int rebuild = 0; rebuild < 2; ++rebuild ){
    backend.build(view);
    reference.build(view);
    double external = 0;
    PropagationInputs inputs;
    inputs.externalWeights = {{&external, 1, 0, 1}};
    for( double value : {-.2, .5, 1.9, 2.5} ){
      inputs.parameters.values = {value};
      external = value;
      ++inputs.externalWeights[0].generation;
      for( int repeat = 0; repeat < 3; ++repeat ){ compareMpsWithCpu(backend, reference, inputs); }
    }
  }
}

TEST(MpsCompactDialDescriptors, MatchesCpuWithBoundsSmoothingAndExternalWeights) {
  for( int repeats : {1, int(kDeviceCachedDialReuseThreshold)} ){
    MpsBackend backend;
    CpuBackend reference;
    if( backend.getCapabilities().deviceName == "Metal unavailable" ){ GTEST_SKIP(); }
    backend.configure(ConfigReader(JsonType{{"propagationMode", "recomputeResponses"}}));
    for( int limits = 0; limits < 5; ++limits ){
      DialResponseSupervisor supervisor;
      if( limits != 2 ){ supervisor.setMinResponse(0.); }
      if( limits != 3 ){ supervisor.setMaxResponse(2.); }
      if( limits != 0 ){ supervisor.setSmoothingWidths(limits == 4 ? .5 : .4, limits == 4 ? 1.5 : .1); }
      auto view = makeModel(supervisor, false, repeats);
      if( limits == 4 ){
        // Distinct transitions exercise several out-of-line records, including
        // spline flags whose upper bits encode different smoothing indices.
        for( std::size_t i = 0; i < view.propagation.dials.size(); ++i ){
          auto& dial = view.propagation.dials[i];
          auto& smoothing = dial.smoothing;
          smoothing.minWidth *= 1. + .1 * double(i);
          smoothing.minInverseWidth = .5 / smoothing.minWidth;
          dial.minResponse = smoothing.minResponse + smoothing.minWidth;
        }
      }
      backend.build(view);
      reference.build(view);
      double external = 0;
      PropagationInputs inputs;
      inputs.externalWeights = {{&external, 1, 0, 1}};
      for( double value : {-1., -.4, -.2, 0., .2, .4, .8, 1., 1.6, 1.9, 2., 2.1, 2.5} ){
        SCOPED_TRACE(value);
        inputs.parameters.values = {value};
        external = value;
        ++inputs.externalWeights[0].generation;
        for( int repeat = 0; repeat < 3; ++repeat ){ compareMpsWithCpu(backend, reference, inputs); }
      }
    }
  }
}

TEST(MpsCachedDialDispatch, MixedCachedAndUncachedDescriptorsMatchCpu) {
  for( bool cache : {false, true} ){
    SCOPED_TRACE(cache);
    MpsBackend backend;
    CpuBackend reference;
    if( backend.getCapabilities().deviceName == "Metal unavailable" ){ GTEST_SKIP(); }
    backend.configure(ConfigReader(JsonType{{"propagationMode", "recomputeResponses"}, {"enableCachedDialResponses", cache}}));
    DialResponseSupervisor supervisor;
    auto view = makeModel(supervisor, false, 1);
    auto& model = view.propagation;
    model.events.clear();
    model.eventWeights.clear();
    model.eventDialIndices.clear();
    model.externalWeightBlocks.clear();
    auto appendEvent = [&](uint32_t dial_){
      EventView event;
      event.sampleIndex = event.binIndex = event.globalBinIndex = 0;
      event.resultIndex = model.events.size();
      BackendEventWeightDescriptor weight;
      weight.firstDial = model.eventDialIndices.size();
      weight.dialCount = 1;
      model.events.push_back(event);
      model.eventWeights.push_back(weight);
      model.eventDialIndices.push_back(dial_);
    };
    for( uint32_t family = 3; family < 8; ++family ){
      // First-use packing puts the uncached descriptor before the cached one.
      // This catches writing responses to the compact index instead of its slot.
      auto oneOff = model.dials.at(family);
      auto oneOffIndex = uint32_t(model.dials.size());
      model.dials.push_back(oneOff);
      appendEvent(oneOffIndex);
      for( uint32_t i = 0; i < kDeviceCachedDialReuseThreshold; ++i ){ appendEvent(family); }
    }
    for( int rebuild = 0; rebuild < 2; ++rebuild ){
      backend.build(view);
      reference.build(view);
      PropagationInputs inputs;
      for( double value : {-.2, .5, 1.7} ){
        inputs.parameters.values = {value};
        for( int repeat = 0; repeat < 3; ++repeat ){
          compareMpsWithCpu(backend, reference, inputs);
          EXPECT_EQ(backend.getLastTimingSummary().cachedDialCount, cache ? 5 : 0);
        }
      }
    }
  }
}
#endif

INSTANTIATE_TEST_SUITE_P(CpuAndMetal, BackendResponseSmoothing,
                        testing::Combine(testing::Bool(), testing::Bool(), testing::Bool()));
