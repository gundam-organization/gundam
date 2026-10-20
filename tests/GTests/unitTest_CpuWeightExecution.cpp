#include "CpuBackend.h"
#include "GundamGlobals.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
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

TEST(CpuWeightExecution, DynamicBlocksMatchFixedBitForBit) {
  const int previousThreads = GundamGlobals::getNbCpuThreads();
  for( int threads : {1, 2, 5} ){
    GundamGlobals::setNumberOfThreads(threads);
    for( bool mixed : {false, true} ){
      for( bool capped : {false, true} ){
        for( std::size_t events : {std::size_t(1), std::size_t(3), std::size_t(17), std::size_t(1031)} ){
          SCOPED_TRACE(::testing::Message() << threads << " threads, events=" << events
                       << ", mixed=" << mixed << ", capped=" << capped);
          auto view = std::make_shared<Backends::EngineView>(makeWeightModel(mixed, capped));
          auto& model = view->propagation;
          const auto source = model.eventWeights;
          model.events.resize(events);
          model.eventWeights.resize(events);
          model.totalBins = 3;
          for( std::size_t i = 0; i < events; ++i ){
            model.eventWeights[i] = source[i % source.size()];
            model.eventWeights[i].baseWeight = i % 11 == 0 ? 0. : (i % 2 ? -1.3 : .7);
            model.events[i].resultIndex = events - i - 1;
            model.events[i].globalBinIndex = int(i % 3);
          }
          // One event dominates estimated work. The other dynamic blocks must
          // remain nonempty even when this event crosses several work targets.
          model.eventWeights[0].firstDial = model.eventDialIndices.size();
          model.eventWeights[0].dialCount = 257;
          model.eventWeights[0].baseWeight = 1.;
          model.eventDialIndices.insert(model.eventDialIndices.end(), 257, 0);
          Backends::CpuBackend fixed;
          fixed.configure(ConfigUtils::ConfigReader(JsonType{{"weightScheduling", "Fixed"}}));
          fixed.buildShared(view);
          Backends::CpuBackend dynamic;
          dynamic.configure(ConfigUtils::ConfigReader(JsonType{{"weightScheduling", "Dynamic"}}));
          dynamic.buildShared(view);
          const auto* modelStorage = dynamic.getEngineView().propagation.eventWeights.data();
          Backends::PropagationInputs inputs;
          double external{1.3};
          if( mixed ){ inputs.externalWeights = {{&external, 1, 0, 1}}; }
          for( double value : {2., 0., -.5, std::numeric_limits<double>::quiet_NaN()} ){
            inputs.parameters.values = {value};
            const auto expected = fixed.requestPropagation(inputs);
            auto sameBits = [](const std::vector<double>& a_, const std::vector<double>& b_){
              ASSERT_EQ(a_.size(), b_.size());
              EXPECT_EQ(std::memcmp(a_.data(), b_.data(), a_.size() * sizeof(double)), 0);
            };
            for( std::size_t factor : {1, 2, 5, 10, 2} ){
              for( auto mode : {Backends::CpuWeightScheduling::Dynamic, Backends::CpuWeightScheduling::Fixed} ){
                dynamic.setDiagnosticWeightScheduling(mode, factor);
                const auto layout = dynamic.getWeightSchedulingDiagnostics();
                const auto dynamicExpected = mode == Backends::CpuWeightScheduling::Dynamic
                    and threads > 1 and events >= factor * threads;
                EXPECT_EQ(layout.at("weightBlockCount"), dynamicExpected ? factor * threads : 0);
                EXPECT_EQ(layout.at("weightBlocksPerThread"), factor);
                EXPECT_EQ(dynamic.getEngineView().propagation.eventWeights.data(), modelStorage);
                const auto actual = dynamic.requestPropagation(inputs);
                sameBits(fixed.getEventWeightsHostView(expected), dynamic.getEventWeightsHostView(actual));
                sameBits(fixed.getHistogramSumsHostView(expected), dynamic.getHistogramSumsHostView(actual));
                sameBits(fixed.getHistogramSumSquaresHostView(expected), dynamic.getHistogramSumSquaresHostView(actual));
              }
            }
            external += .1;
          }
        }
      }
    }
  }
  GundamGlobals::setNumberOfThreads(previousThreads);
}

TEST(CpuWeightExecution, DiagnosticSchedulingRejectsInvalidRequests) {
  Backends::CpuBackend backend;
  EXPECT_THROW(backend.setDiagnosticWeightScheduling(Backends::CpuWeightScheduling::Dynamic, 2), std::exception);
  GundamGlobals::setNumberOfThreads(2);
  backend.build(makeWeightModel(false, false));
  EXPECT_THROW(backend.setDiagnosticWeightScheduling(Backends::CpuWeightScheduling::Dynamic, 0), std::exception);
  EXPECT_THROW(backend.setDiagnosticWeightScheduling(Backends::CpuWeightScheduling::Dynamic,
                                                    std::numeric_limits<std::size_t>::max()), std::exception);
  // Validation must leave the previous usable layout intact.
  Backends::PropagationInputs inputs;
  inputs.parameters.values = {1.};
  auto token = backend.requestPropagation(inputs);
  EXPECT_EQ(backend.getEventWeightsHostView(token), (std::vector<double>{-2., -2., -2.}));
}
