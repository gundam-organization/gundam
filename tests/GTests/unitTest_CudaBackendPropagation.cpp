#include "CudaBackend.h"
#include "CpuBackend.h"
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
#include <stdexcept>
#include <vector>

namespace {
  using namespace Backends;

  EngineView makeModel(bool capped_, bool extrapolate_) {
    EngineView view;
    auto& model = view.propagation;
    model.parameterCount = 3;
    model.totalBins = 5;
    model.hasReweightCap = capped_;
    model.maxReweight = 1.25;
    const std::vector<DialUtils::DialPoint> points{{0., .8, 0.}, {1., 1., 0.}, {2., 1.1, 0.}, {3., .9, 0.}, {4., 1.2, 0.}};
    CompactSpline compact; compact.buildDial(points);
    UniformSpline uniform; uniform.buildDial(points);
    MonotonicSpline monotonic; monotonic.buildDial(points);
    GeneralSpline general; general.buildDial(points);
    Graph graph; graph.buildDial(points);
    const std::vector<BackendDialType> types{BackendDialType::Norm, BackendDialType::Shift,
        BackendDialType::CompactSpline, BackendDialType::UniformSpline, BackendDialType::MonotonicSpline,
        BackendDialType::GeneralSpline, BackendDialType::Graph};
    const std::vector<std::vector<double>> payloads{{}, {.9}, compact.getDialData(), uniform.getDialData(),
        monotonic.getDialData(), general.getDialData(), graph.getDialData()};
    for( std::size_t i = 0; i < types.size(); ++i ){
      BackendDialDescriptor dial;
      dial.type = types[i];
      dial.firstInput = model.dialInputs.size();
      dial.inputCount = dial.type == BackendDialType::Shift ? 0 : 1;
      if( dial.inputCount ){ model.dialInputs.push_back({i % 3, i % 2 == 0, 0., 4.}); }
      dial.payloadOffset = model.dialPayloads.size();
      dial.payloadSize = payloads[i].size();
      model.dialPayloads.insert(model.dialPayloads.end(), payloads[i].begin(), payloads[i].end());
      dial.allowExtrapolation = extrapolate_;
      dial.hasMinResponse = dial.hasMaxResponse = true;
      dial.minResponse = .6;
      dial.maxResponse = 1.4;
      dial.smoothing = {.5, 1.5, .1, .1, 5., 5.};
      model.dials.push_back(dial);
    }
    // Non-block-multiple count; permuted result slots; duplicated dial references;
    // negative/zero base weights, outside events and an entirely empty bin.
    model.events.resize(521);
    model.eventWeights.resize(521);
    for( std::size_t i = 0; i < model.events.size(); ++i ){
      const auto slot = (i * 37) % model.events.size();
      auto& event = model.events[i];
      event.resultIndex = slot;
      event.globalBinIndex = i % 13 == 0 ? -1 : int(i % 4);
      auto& weight = model.eventWeights[slot];
      weight.baseWeight = i % 11 == 0 ? 0. : (i % 3 == 0 ? -0.125 : 0.25);
      weight.firstDial = model.eventDialIndices.size();
      weight.dialCount = i % 10;
      for( std::size_t j = 0; j < weight.dialCount; ++j ){ model.eventDialIndices.push_back((i + j) % types.size()); }
    }
    LikelihoodSampleView likelihood;
    likelihood.dataSums = {2., 1., 0., 3., 0.};
    likelihood.ignoredBins = {false, false, true, false, false};
    likelihood.evalBin = [](double data_, double prediction_, double error_, int){
      const double difference = prediction_ - data_;
      return difference * difference / (1. + error_ * error_);
    };
    view.likelihood.samples.push_back(likelihood);
    return view;
  }

  void expectClose(const std::vector<double>& actual_, const std::vector<double>& reference_) {
    ASSERT_EQ(actual_.size(), reference_.size());
    for( std::size_t i = 0; i < actual_.size(); ++i ){
      EXPECT_NEAR(actual_[i], reference_[i], 2.e-12 * std::max(1., std::abs(reference_[i]))) << "index " << i;
    }
  }
}

TEST(CudaBackendPropagation, MatchesCpuForEveryScalarDialAndBothModes) {
  for( const auto* mode : {"cachedResponses", "recomputeResponses"} ){
    for( bool capped : {false, true} ){
      for( bool extrapolate : {false, true} ){
        SCOPED_TRACE(::testing::Message() << mode << " capped=" << capped << " extrapolate=" << extrapolate);
        auto view = std::make_shared<EngineView>(makeModel(capped, extrapolate));
        CpuBackend cpu;
        CudaBackend cuda;
        cuda.configure(ConfigReader(JsonType{{"propagationMode", mode}}));
        cpu.buildShared(view);
        cuda.buildShared(view);
        PropagationToken previous;
        const void* deviceWeights = nullptr;
        for( const auto& values : std::vector<std::vector<double>>{
            {.8, 1.2, 2.3}, {.8, 1.2, 2.3}, {1., 1.2, 2.3},
            {1., 2., 2.3}, {0., 1., 4.}, {-1., -2., -3.}, {9., 8., 7.}, {.45, 1.55, .55}} ){
          PropagationInputs inputs;
          inputs.parameters.values = values;
          const auto cpuToken = cpu.requestPropagation(inputs);
          const auto token = cuda.requestPropagation(inputs);
          EXPECT_TRUE(cuda.isReady(token));
          EXPECT_NO_THROW(cuda.wait(token));
          EXPECT_EQ(cuda.getStatus(token).eventWeights, OutputState::ReadyOnDevice);
          EXPECT_EQ(cuda.getStatus(token).histograms, OutputState::ReadyOnHost);
          EXPECT_FALSE(cuda.isReady(previous));
          EXPECT_THROW((void)cuda.getLikelihood(previous), std::runtime_error);
          const auto device = cuda.getDeviceView(token);
          ASSERT_NE(device.eventWeights, nullptr);
          EXPECT_EQ(device.eventWeightsBytes, view->propagation.events.size() * sizeof(double));
          EXPECT_EQ(device.histogramBytes, 5 * sizeof(double));
          if( deviceWeights ){ EXPECT_EQ(device.eventWeights, deviceWeights); }
          deviceWeights = device.eventWeights;
          expectClose(cuda.getEventWeightsHostView(token), cpu.getEventWeightsHostView(cpuToken));
          expectClose(cuda.getHistogramSumsHostView(token), cpu.getHistogramSumsHostView(cpuToken));
          expectClose(cuda.getHistogramSumSquaresHostView(token), cpu.getHistogramSumSquaresHostView(cpuToken));
          EXPECT_NEAR(cuda.getLikelihood(token), cpu.getLikelihood(cpuToken), 2.e-11 * std::max(1., std::abs(cpu.getLikelihood(cpuToken))));
          EXPECT_EQ(cuda.getHistogramSumsHostView(token).back(), 0.);
          EXPECT_EQ(cuda.getStatus(token).eventWeights, OutputState::ReadyOnHost);
          const auto bytes = cuda.getLastTimingSummary().eventWeightReadbackBytes;
          cuda.materialize(token, OutputRequest::EventWeights);
          EXPECT_EQ(cuda.getLastTimingSummary().eventWeightReadbackBytes, bytes);
          EXPECT_THROW(cuda.materialize(token, OutputRequest::SampleLikelihoods), std::runtime_error);
          previous = token;
        }
        EXPECT_EQ(cuda.getTimingTotals().propagationCount, 8);
      }
    }
  }
}

TEST(CudaBackendPropagation, EmptyModelsAndUndialedEventsNeedNoParameters) {
  for( const auto* mode : {"cachedResponses", "recomputeResponses"} ){
    CudaBackend backend;
    backend.configure(ConfigReader(JsonType{{"propagationMode", mode}}));
    EngineView view;
    backend.build(view);
    const auto empty = backend.requestPropagation({});
    EXPECT_TRUE(backend.isReady(empty));
    EXPECT_TRUE(backend.getEventWeightsHostView(empty).empty());
    EXPECT_TRUE(backend.getHistogramSumsHostView(empty).empty());
    EXPECT_THROW((void)backend.getLikelihood(empty), std::runtime_error);
    view.propagation.totalBins = 2;
    view.propagation.eventWeights = {{-2., 0, 0}, {3., 0, 0}};
    view.propagation.events.resize(2);
    view.propagation.events[0].resultIndex = 1;
    view.propagation.events[0].globalBinIndex = 0;
    view.propagation.events[1].resultIndex = 0;
    view.propagation.events[1].globalBinIndex = -1;
    backend.build(view);
    const auto token = backend.requestPropagation({});
    EXPECT_EQ(backend.getStatus(token).histograms, OutputState::ReadyOnDevice);
    EXPECT_EQ(backend.getEventWeightsHostView(token), (std::vector<double>{-2., 3.}));
    EXPECT_EQ(backend.getHistogramSumsHostView(token), (std::vector<double>{3., 0.}));
    EXPECT_EQ(backend.getHistogramSumSquaresHostView(token), (std::vector<double>{9., 0.}));
    EXPECT_FALSE(backend.isReady(empty));
  }
}

TEST(CudaBackendPropagation, SparseUpdatesHandleMultipleBlocksRangesAndUnusedParameters) {
  EngineView view;
  auto& model = view.propagation;
  model.parameterCount = 5;
  model.totalBins = 17;
  constexpr std::size_t scalarCount = 2052;
  for( std::size_t i = 0; i < scalarCount; ++i ){
    model.dialInputs.push_back({i % 4, i % 3 == 0, 0., 4.});
    BackendDialDescriptor dial;
    dial.firstInput = i;
    dial.inputCount = 1;
    model.dials.push_back(dial);
    model.eventWeights.push_back({.25, 2 * i, 2});
    EventView event;
    event.resultIndex = i;
    event.globalBinIndex = i % 17;
    model.events.push_back(event);
    model.eventDialIndices.push_back(i);
    model.eventDialIndices.push_back(scalarCount);
  }
  BackendDialDescriptor shift;
  shift.type = BackendDialType::Shift;
  shift.payloadSize = 1;
  model.dialPayloads = {.8};
  model.dials.push_back(shift);
  for( const auto* mode : {"cachedResponses", "recomputeResponses"} ){
    CudaBackend cuda;
    CpuBackend cpu;
    cuda.configure(ConfigReader(JsonType{{"propagationMode", mode}}));
    cuda.build(view);
    cpu.build(view);
    const std::vector<std::vector<double>> points{
        {1., 1., 1., 1., 0.}, {1., 1., 1., 1., 0.}, {1.2, 1., 1., 1., 0.},
        {1.2, 1.1, 1., 1.3, 0.}, {1.5, 1.4, 1.2, 1.3, 0.},
        {1.5, 1.4, 1.2, 1.3, 9.}, {6.5, 1.4, 1.2, 1.3, 9.},
        {1.5, 1.4, 1.2, 1.3, 9.}};
    const std::vector<std::size_t> candidates{2053, 0, 513, 1026, 2053, 0, 513, 513};
    for( std::size_t i = 0; i < points.size(); ++i ){
      SCOPED_TRACE(::testing::Message() << mode << " point=" << i);
      PropagationInputs inputs;
      inputs.parameters.values = points[i];
      const auto token = cuda.requestPropagation(inputs);
      const auto reference = cpu.requestPropagation(inputs);
      EXPECT_EQ(cuda.getLastTimingSummary().cacheCandidateCount,
                std::string(mode) == "cachedResponses" ? candidates[i] : model.dials.size());
      expectClose(cuda.getEventWeightsHostView(token), cpu.getEventWeightsHostView(reference));
      expectClose(cuda.getHistogramSumsHostView(token), cpu.getHistogramSumsHostView(reference));
      expectClose(cuda.getHistogramSumSquaresHostView(token), cpu.getHistogramSumSquaresHostView(reference));
    }
    cuda.build(view); // Rebuild must prime every response, including Shift.
    PropagationInputs inputs;
    inputs.parameters.values = points.back();
    const auto token = cuda.requestPropagation(inputs);
    const auto reference = cpu.requestPropagation(inputs);
    EXPECT_EQ(cuda.getLastTimingSummary().cacheCandidateCount, model.dials.size());
    expectClose(cuda.getEventWeightsHostView(token), cpu.getEventWeightsHostView(reference));
  }
}

TEST(CudaBackendPropagation, FailureInvalidatesTokensAndRetryRecomputesResponses) {
  for( const auto* mode : {"cachedResponses", "recomputeResponses"} ){
    CudaBackend backend;
    backend.configure(ConfigReader(JsonType{{"propagationMode", mode}}));
    auto view = makeModel(false, true);
    auto failLikelihood = std::make_shared<bool>(false);
    const auto evaluate = view.likelihood.samples[0].evalBin;
    view.likelihood.samples[0].evalBin = [evaluate, failLikelihood](double data_, double prediction_, double error_, int bin_){
      return *failLikelihood ? std::numeric_limits<double>::quiet_NaN() : evaluate(data_, prediction_, error_, bin_);
    };
    backend.build(view);
    PropagationInputs inputs;
    inputs.parameters.values = {.8, 1.2, 2.3};
    auto token = backend.requestPropagation(inputs);
    const auto reference = backend.getEventWeightsHostView(token);
    const auto originalToken = token;
    *failLikelihood = true;
    EXPECT_THROW((void)backend.requestPropagation(inputs), std::runtime_error);
    EXPECT_FALSE(backend.isReady(token));
    EXPECT_THROW((void)backend.getEventWeightsHostView(token), std::runtime_error);
    *failLikelihood = false;
    token = backend.requestPropagation(inputs);
    EXPECT_EQ(backend.getLastTimingSummary().cacheCandidateCount, view.propagation.dials.size());
    expectClose(backend.getEventWeightsHostView(token), reference);
    inputs.parameters.values.pop_back();
    EXPECT_THROW((void)backend.requestPropagation(inputs), std::runtime_error);
    EXPECT_FALSE(backend.isReady(token));
    inputs.parameters.values = {.8, 1.2, std::numeric_limits<double>::infinity()};
    EXPECT_THROW((void)backend.requestPropagation(inputs), std::runtime_error);
    inputs.parameters.values = {.8, 1.2, 2.3};
    inputs.forceBinningUpdate = true;
    EXPECT_THROW((void)backend.requestPropagation(inputs), std::runtime_error);
    inputs.forceBinningUpdate = false;
    token = backend.requestPropagation(inputs);
    expectClose(backend.getEventWeightsHostView(token), reference);
    backend.configure(ConfigReader(JsonType{{"propagationMode", mode}}));
    backend.build(view);
    token = backend.requestPropagation(inputs);
    EXPECT_NE(token.id, originalToken.id);
    EXPECT_FALSE(backend.isReady(originalToken));
    expectClose(backend.getEventWeightsHostView(token), reference);
  }
}

TEST(CudaBackendPropagation, DeviceOverflowCannotPublishResultsAndCanBeRetried) {
  EngineView view;
  auto& model = view.propagation;
  model.parameterCount = 1;
  model.totalBins = 1;
  model.dialInputs = {{0}};
  model.dials.resize(1);
  model.dials[0].inputCount = 1;
  model.eventDialIndices = {0, 0};
  model.eventWeights = {{1., 0, 2}};
  model.events.resize(1);
  model.events[0].globalBinIndex = 0;
  for( const auto* mode : {"cachedResponses", "recomputeResponses"} ){
    CudaBackend backend;
    backend.configure(ConfigReader(JsonType{{"propagationMode", mode}}));
    backend.build(view);
    PropagationInputs inputs;
    inputs.parameters.values = {2.};
    const auto token = backend.requestPropagation(inputs);
    inputs.parameters.values = {1.e308};
    EXPECT_THROW((void)backend.requestPropagation(inputs), std::runtime_error);
    EXPECT_FALSE(backend.isReady(token));
    inputs.parameters.values = {3.};
    const auto retry = backend.requestPropagation(inputs);
    EXPECT_EQ(backend.getEventWeightsHostView(retry), (std::vector<double>{9.}));
    EXPECT_EQ(backend.getHistogramSumSquaresHostView(retry), (std::vector<double>{81.}));
  }
}
