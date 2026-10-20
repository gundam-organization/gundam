#include "CpuBackend.h"
#include "GundamGlobals.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace {
  Backends::EngineView makeWeightModel(bool mixed_, bool capped_) {
    using namespace Backends;
    EngineView view;
    auto& model = view.propagation;
    model.parameterCount = 1;
    model.totalBins = 1;
    model.hasReweightCap = capped_;
    model.maxReweight = 3.;
    model.dialInputs = {{0}};
    BackendDialDescriptor norm;
    norm.inputCount = 1;
    model.dials = {norm};
    if( mixed_ ){
      BackendDialDescriptor shift;
      shift.type = BackendDialType::Shift;
      shift.payloadSize = 1;
      model.dialPayloads = {.5};
      model.dials.push_back(shift);
      BackendDialDescriptor external;
      external.type = BackendDialType::ExternalWeight;
      model.dials.push_back(external);
      model.externalWeightBlocks = {{0, 1}};
    }
    // Highly uneven ranges, including an event with no dials. The event
    // metadata order differs from result order; weights use result slots.
    for( std::size_t count : {std::size_t(0), std::size_t(1), std::size_t(7)} ){
      BackendEventWeightDescriptor weight;
      weight.baseWeight = -2.;
      weight.firstDial = model.eventDialIndices.size();
      weight.dialCount = count;
      for( std::size_t i = 0; i < count; ++i ){
        model.eventDialIndices.push_back(mixed_ ? std::uint32_t(i % 3) : 0);
      }
      EventView event;
      event.resultIndex = model.events.size();
      event.globalBinIndex = 0;
      model.events.push_back(event);
      model.eventWeights.push_back(weight);
    }
    std::reverse(model.events.begin(), model.events.end());
    return view;
  }
}

TEST(CpuWeightExecution, CachedAndMixedPathsPreserveProductsAndCapOrder) {
  for( int threads : {1, 2, 5} ){
    GundamGlobals::setNumberOfThreads(threads);
    for( bool mixed : {false, true} ){
      for( bool capped : {false, true} ){
        SCOPED_TRACE(::testing::Message() << threads << " threads, mixed=" << mixed << ", capped=" << capped);
        Backends::CpuBackend backend;
        backend.build(makeWeightModel(mixed, capped));
        Backends::PropagationInputs inputs;
        inputs.parameters.values = {2.};
        double external{2.};
        if( mixed ){ inputs.externalWeights = {{&external, 1, 0, 1}}; }
        auto token = backend.requestPropagation(inputs);
        const auto* storage = backend.getEventWeightsHostView(token).data();
        const double product = mixed ? 8. : 128.;
        const double expected = -2. * (capped ? std::min(product, 3.) : product);
        EXPECT_EQ(backend.getEventWeightsHostView(token), (std::vector<double>{-2., -4., expected}));
        EXPECT_DOUBLE_EQ(backend.getHistogramSumsHostView(token)[0], -6. + expected);
        EXPECT_DOUBLE_EQ(backend.getHistogramSumSquaresHostView(token)[0], 20. + expected * expected);
        // Repeated propagation must overwrite the retained storage and update
        // the cache when the parameter changes.
        inputs.parameters.values[0] = 0.;
        token = backend.requestPropagation(inputs);
        EXPECT_EQ(backend.getEventWeightsHostView(token).data(), storage);
        EXPECT_EQ(backend.getEventWeightsHostView(token), (std::vector<double>{-2., 0., 0.}));
        EXPECT_EQ(backend.getTimingTotals().propagationCount, 2);
        EXPECT_GT(backend.getTimingTotals().propagationSeconds, 0.);
        backend.build(makeWeightModel(mixed, capped));
        EXPECT_EQ(backend.getTimingTotals().propagationCount, 0);
      }
    }
  }
}

TEST(CpuWeightExecution, SharedModelLifetimeAndStandaloneSnapshot) {
  Backends::CpuBackend backend;
  auto model = std::make_shared<Backends::EngineView>(makeWeightModel(false, false));
  const auto* payload = model->propagation.eventDialIndices.data();
  std::weak_ptr<const Backends::EngineView> lifetime = model;
  backend.buildShared(model);
  EXPECT_EQ(backend.getEngineView().propagation.eventDialIndices.data(), payload);
  model.reset();
  EXPECT_FALSE(lifetime.expired());
  Backends::PropagationInputs inputs;
  inputs.parameters.values = {1.};
  auto token = backend.requestPropagation(inputs);
  EXPECT_EQ(backend.getEventWeightsHostView(token), (std::vector<double>{-2., -2., -2.}));
  auto standalone = makeWeightModel(false, false);
  backend.build(standalone);
  EXPECT_TRUE(lifetime.expired());
  standalone.propagation.eventWeights[0].baseWeight = 99.;
  token = backend.requestPropagation(inputs);
  EXPECT_DOUBLE_EQ(backend.getEventWeightsHostView(token)[0], -2.);
}

TEST(CpuWeightExecution, EmptyModelAndNonfiniteResponse) {
  Backends::CpuBackend backend;
  Backends::PropagationInputs inputs;
  backend.build(Backends::EngineView{});
  auto token = backend.requestPropagation(inputs);
  EXPECT_TRUE(backend.getEventWeightsHostView(token).empty());
  for( bool capped : {false, true} ){
    backend.build(makeWeightModel(false, capped));
    inputs.parameters.values = {std::numeric_limits<double>::quiet_NaN()};
    token = backend.requestPropagation(inputs);
    EXPECT_DOUBLE_EQ(backend.getEventWeightsHostView(token)[0], -2.);
    EXPECT_TRUE(std::isnan(backend.getEventWeightsHostView(token)[1]));
  }
}

TEST(CpuWeightExecution, ConstantDialsNeedNoParameterInputs) {
  using namespace Backends;
  EngineView view;
  auto& model = view.propagation;
  model.totalBins = 1;
  BackendDialDescriptor shift;
  shift.type = BackendDialType::Shift;
  shift.payloadSize = 1;
  shift.hasMinResponse = true;
  shift.minResponse = 1.;
  model.dials = {shift};
  model.dialPayloads = {.5};
  model.eventDialIndices = {0};
  model.events.resize(1);
  model.events[0].globalBinIndex = 0;
  model.eventWeights = {{-2., 0, 1}};
  CpuBackend backend;
  backend.build(view);
  PropagationInputs inputs;
  for( int repeat = 0; repeat < 3; ++repeat ){
    auto token = backend.requestPropagation(inputs);
    EXPECT_DOUBLE_EQ(backend.getEventWeightsHostView(token)[0], -2.);
    EXPECT_EQ(backend.getLastTimingSummary().cachedDialCount, 1);
  }
}
