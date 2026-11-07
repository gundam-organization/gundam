#include "CpuBackend.h"
#include "CpuWeightKernels.h"
#include "GundamGlobals.h"
#include "CompactSpline.h"
#include "UniformSpline.h"
#include "MonotonicSpline.h"
#include "GeneralSpline.h"
#include "Graph.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <type_traits>
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

TEST(CpuWeightExecution, InterleavedKernelPreservesEveryProductBit) {
  using namespace Backends;
  std::mt19937 random(1729);
  std::vector<double> responses(257);
  for( auto& response : responses ){ response = .8 + double(random() % 4000) / 10000; }
  responses[0] = 0.;
  responses[1] = -1.25;
  responses[2] = std::numeric_limits<double>::infinity();
  responses[3] = std::numeric_limits<double>::quiet_NaN();
  std::vector<BackendEventWeightDescriptor> events(131);
  std::vector<std::uint32_t> indices;
  for( std::size_t i = 0; i < events.size(); ++i ){
    events[i].firstDial = indices.size();
    events[i].dialCount = i % 33;
    events[i].baseWeight = i % 2 == 0 ? -.5 : 1.25;
    for( std::size_t j = 0; j < events[i].dialCount; ++j ){ indices.push_back(random() % responses.size()); }
  }
  auto check = [&](auto cached_, auto capped_){
    for( std::size_t begin : {std::size_t(0), std::size_t(1), events.size()} ){
      for( auto end : {begin, events.size() - (begin < events.size() ? 1 : 0), events.size()} ){
        std::vector<double> serial(events.size(), -123.), paired = serial;
        auto evaluate = [&](std::uint32_t dial_){ return responses[dial_]; };
        CpuWeightKernels::reweight<decltype(cached_)::value, decltype(capped_)::value>(
            events.data(), indices.data(), responses.data(), serial.data(), begin, end, 3., evaluate);
        CpuWeightKernels::reweightInterleaved<decltype(cached_)::value, decltype(capped_)::value>(
            events.data(), indices.data(), responses.data(), paired.data(), begin, end, 3., evaluate);
        EXPECT_EQ(std::memcmp(serial.data(), paired.data(), serial.size() * sizeof(double)), 0);
      }
    }
  };
  check(std::true_type{}, std::false_type{});
  check(std::true_type{}, std::true_type{});
  check(std::false_type{}, std::false_type{});
  check(std::false_type{}, std::true_type{});
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

TEST(CpuWeightExecution, LocalityProbeMatchesColdLruAndKeepsModelUnchanged) {
  GundamGlobals::setNumberOfThreads(1);
  auto view = std::make_shared<Backends::EngineView>(makeWeightModel(false, false));
  auto& model = view->propagation;
  model.events.resize(1);
  model.events[0].resultIndex = 0;
  model.eventWeights.resize(1);
  model.eventDialIndices.clear();
  model.dials.resize(4160, model.dials[0]);
  for( int repeat = 0; repeat < 2; ++repeat ){
    for( std::uint32_t line = 0; line < 520; ++line ){
      model.eventDialIndices.push_back(line * 8);
      model.eventDialIndices.push_back(line * 8 + 1);
    }
  }
  model.eventWeights[0].firstDial = 0;
  model.eventWeights[0].dialCount = model.eventDialIndices.size();
  Backends::CpuBackend backend;
  backend.buildShared(view);
  Backends::PropagationInputs inputs;
  inputs.parameters.values = {1.001};
  auto token = backend.requestPropagation(inputs);
  const auto weights = backend.getEventWeightsHostView(token);
  const auto indices = model.eventDialIndices;
  const auto report = backend.getWeightLocalityDiagnostics();
  ASSERT_TRUE(report.at("supported"));
  EXPECT_EQ(report.at("references"), indices.size());
  EXPECT_EQ(report.at("referencedResponses"), 1040);
  const std::size_t offset = report.at("responseBasePageOffset");
  for( const auto* layout : {"current", "firstUse"} ){
    std::vector<std::size_t> slots(4160, 4160);
    std::size_t next = 0;
    for( const auto& estimate : report.at(layout).at("idealLru") ){
      const std::size_t capacity = estimate.at("capacityBytes").get<std::size_t>() / 64;
      std::vector<std::size_t> lru;
      std::size_t hits = 0;
      for( const auto id : indices ){
        if( slots[id] == 4160 ){ slots[id] = next++; }
        const auto slot = std::string(layout) == "current" ? id : slots[id];
        const auto line = (offset + slot * sizeof(double)) / 64;
        auto found = std::find(lru.begin(), lru.end(), line);
        if( found != lru.end() ){ ++hits; lru.erase(found); }
        lru.insert(lru.begin(), line);
        if( lru.size() > capacity ){ lru.pop_back(); }
      }
      EXPECT_EQ(estimate.at("hits"), hits);
      EXPECT_EQ(estimate.at("misses"), indices.size() - hits);
    }
  }
  EXPECT_LT(report.at("firstUse").at("distinctLines").get<std::size_t>(),
            report.at("current").at("distinctLines").get<std::size_t>());
  EXPECT_EQ(model.eventDialIndices, indices);
  token = backend.requestPropagation(inputs);
  EXPECT_EQ(backend.getEventWeightsHostView(token), weights);
}

TEST(CpuWeightExecution, LocalityProbeHandlesNoReferencesAndMixedModels) {
  auto view = makeWeightModel(false, false);
  for( auto& event : view.propagation.eventWeights ){ event.dialCount = 0; event.firstDial = 0; }
  view.propagation.eventDialIndices.clear();
  Backends::CpuBackend backend;
  backend.build(view);
  const auto report = backend.getWeightLocalityDiagnostics();
  EXPECT_EQ(report.at("references"), 0);
  EXPECT_EQ(report.at("current").at("distinctLines"), 0);
  EXPECT_EQ(report.at("current").at("idealLru")[0].at("hitFraction"), 0.);
  // Full-stream counters continue after the bounded reuse-analysis prefix.
  view.propagation.eventDialIndices.assign(1000003, 0);
  view.propagation.eventWeights[0].dialCount = 1000003;
  backend.build(view);
  const auto bounded = backend.getWeightLocalityDiagnostics();
  EXPECT_EQ(bounded.at("references"), 1000003);
  EXPECT_EQ(bounded.at("reusePrefixReferences"), 1000000);
  EXPECT_EQ(bounded.at("current").at("sameLineTransitions"), 1000002);
  EXPECT_EQ(bounded.at("current").at("idealLru")[0].at("misses"), 1);
  EXPECT_EQ(bounded.at("current").at("idealLru")[0].at("hits"), 999999);
  backend.build(makeWeightModel(true, false));
  EXPECT_FALSE(backend.getWeightLocalityDiagnostics().at("supported"));
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

TEST(CpuCacheTraversal, LinearAndIndexedPreserveAllDialTypesAndSparseUpdates) {
  using namespace Backends;
  const int previousThreads = GundamGlobals::getNbCpuThreads();
  for( int threads : {1, 2, 5} ){
    GundamGlobals::setNumberOfThreads(threads);
    for( int constants : {1, 2000} ){
      auto view = std::make_shared<EngineView>();
      auto& model = view->propagation;
      model.parameterCount = 3;
      model.totalBins = 3;
      model.hasReweightCap = true;
      model.maxReweight = 2.;
      model.externalWeightBlocks = {{0, 1}};
      std::vector<DialUtils::DialPoint> points;
      for( int x = -2; x <= 4; ++x ){ points.push_back({double(x), 1. + .1 * x, .1}); }
      CompactSpline compact; compact.buildDial(points);
      UniformSpline uniform; uniform.buildDial(points);
      MonotonicSpline monotonic; monotonic.buildDial(points);
      GeneralSpline general; general.buildDial(points);
      Graph graph; graph.buildDial(points);
      const std::vector<std::vector<double>> payloads{{}, compact.getDialData(), uniform.getDialData(),
          monotonic.getDialData(), general.getDialData(), graph.getDialData()};
      const std::vector<BackendDialType> types{BackendDialType::Norm, BackendDialType::CompactSpline,
          BackendDialType::UniformSpline, BackendDialType::MonotonicSpline, BackendDialType::GeneralSpline,
          BackendDialType::Graph};
      auto append = [&](BackendDialDescriptor dial, const std::vector<double>& payload){
        dial.payloadOffset = model.dialPayloads.size();
        dial.payloadSize = payload.size();
        model.dialPayloads.insert(model.dialPayloads.end(), payload.begin(), payload.end());
        model.eventDialIndices.push_back(model.dials.size());
        model.dials.push_back(dial);
        EventView event;
        event.resultIndex = model.events.size();
        event.globalBinIndex = event.resultIndex % 3;
        model.events.push_back(event);
        model.eventWeights.push_back({1., model.eventDialIndices.size() - 1, 1});
      };
      for( std::size_t i = 0; i < 600; ++i ){
        BackendDialDescriptor dial;
        dial.type = types[i % types.size()];
        dial.firstInput = model.dialInputs.size();
        dial.inputCount = 1;
        dial.allowExtrapolation = i % 2;
        dial.hasMinResponse = dial.hasMaxResponse = true;
        dial.minResponse = .2;
        dial.maxResponse = 1.8;
        model.dialInputs.push_back({i % 3, i % 5 == 0, 0., 2.});
        append(dial, payloads[i % payloads.size()]);
      }
      BackendDialDescriptor shift;
      shift.type = BackendDialType::Shift;
      for( int i = 0; i < constants; ++i ){ append(shift, {.8}); }
      BackendDialDescriptor external;
      external.type = BackendDialType::ExternalWeight;
      append(external, {});
      CpuBackend indexed, automatic, perDial;
      indexed.buildShared(view);
      automatic.buildShared(view);
      perDial.buildShared(view);
      perDial.setDiagnosticInputEvaluation(CpuInputEvaluation::PerDial);
      indexed.setDiagnosticCacheTraversal(CpuCacheTraversal::Indexed);
      PropagationInputs inputs;
      double externalValue{1.};
      inputs.externalWeights = {{&externalValue, 1, 0, 1}};
      std::vector<std::vector<double>> values{{1., 1., 1.}, {1., 1., 1.}, {1.2, 1., 1.},
          {1.2, 1.4, 1.6}, {1.3, 1.5, 1.7}, {5.3, 5.5, 5.7}};
      for( std::size_t call = 0; call < values.size(); ++call ){
        inputs.parameters.values = values[call];
        const auto expected = indexed.requestPropagation(inputs);
        const auto actual = automatic.requestPropagation(inputs);
        const auto unshared = perDial.requestPropagation(inputs);
        const auto& timing = automatic.getLastTimingSummary();
        EXPECT_EQ(timing.linearCacheTraversal, constants == 1 and (call == 0 or call >= 3));
        EXPECT_EQ(timing.cacheCandidateCount, indexed.getLastTimingSummary().cacheCandidateCount);
        EXPECT_FALSE(indexed.getLastTimingSummary().linearCacheTraversal);
        EXPECT_TRUE(timing.sharedCacheInputs);
        EXPECT_FALSE(perDial.getLastTimingSummary().sharedCacheInputs);
        auto sameBits = [](const std::vector<double>& a, const std::vector<double>& b){
          ASSERT_EQ(a.size(), b.size());
          EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(double)), 0);
        };
        sameBits(indexed.getEventWeightsHostView(expected), automatic.getEventWeightsHostView(actual));
        sameBits(indexed.getHistogramSumsHostView(expected), automatic.getHistogramSumsHostView(actual));
        sameBits(indexed.getHistogramSumSquaresHostView(expected), automatic.getHistogramSumSquaresHostView(actual));
        sameBits(perDial.getEventWeightsHostView(unshared), automatic.getEventWeightsHostView(actual));
        sameBits(perDial.getHistogramSumsHostView(unshared), automatic.getHistogramSumsHostView(actual));
        sameBits(perDial.getHistogramSumSquaresHostView(unshared), automatic.getHistogramSumSquaresHostView(actual));
        externalValue += .1;
      }
    }
  }
  GundamGlobals::setNumberOfThreads(previousThreads);
}

TEST(CpuCacheInputs, MirrorKeysNaNsAndRuntimeSwitchesPreserveBits) {
  using namespace Backends;
  const int previousThreads = GundamGlobals::getNbCpuThreads();
  GundamGlobals::setNumberOfThreads(5);
  auto model = std::make_shared<EngineView>();
  auto& propagation = model->propagation;
  propagation.parameterCount = propagation.totalBins = 1;
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<BackendDialInputDescriptor> inputs{
      {0, false, nan, nan}, {0, true, 0., 2.}, {0, true, 1., 2.},
      {0, true, 0., 3.}, {0, true, nan, 2.}, {0, true, 0., nan}};
  for( std::size_t i = 0; i < 192; ++i ){
    BackendDialDescriptor dial;
    dial.inputCount = 1;
    dial.firstInput = i;
    propagation.dialInputs.push_back(inputs[i % inputs.size()]);
    propagation.dials.push_back(dial);
    propagation.eventDialIndices.push_back(i);
    EventView event;
    event.resultIndex = i;
    event.globalBinIndex = 0;
    propagation.events.push_back(event);
    propagation.eventWeights.push_back({1., i, 1});
  }
  CpuBackend shared, perDial;
  shared.buildShared(model);
  // Unreferenced, unique-input dials force the reference onto the original
  // per-dial input buffer without changing any event product. This makes the
  // comparison independent of the new shared-group bookkeeping.
  auto reference = *model;
  for( std::size_t i = 0; i < 64; ++i ){
    BackendDialDescriptor dial;
    dial.firstInput = reference.propagation.dialInputs.size();
    dial.inputCount = 1;
    reference.propagation.dialInputs.push_back({0, true, 100. + double(i), 1.});
    reference.propagation.dials.push_back(dial);
  }
  perDial.build(reference);
  EXPECT_EQ(perDial.getWeightSchedulingDiagnostics().at("cacheInputGroups"), 0);
  EXPECT_EQ(shared.getWeightSchedulingDiagnostics().at("cacheInputGroups"), 6);
  EXPECT_LT(shared.getWeightSchedulingDiagnostics().at("cacheInputStorageBytes").get<std::size_t>(),
            propagation.dials.size() * sizeof(double));
  perDial.setDiagnosticInputEvaluation(CpuInputEvaluation::PerDial);
  PropagationInputs runtime;
  for( double value : {0., 0., 1., 1. + 1e-12, 3., 7., -1., nan, nan, 1.} ){
    runtime.parameters.values = {value};
    const auto expected = perDial.requestPropagation(runtime);
    for( auto mode : {CpuInputEvaluation::Shared, CpuInputEvaluation::PerDial, CpuInputEvaluation::Shared} ){
      shared.setDiagnosticInputEvaluation(mode);
      const auto actual = shared.requestPropagation(runtime);
      const auto& a = perDial.getEventWeightsHostView(expected);
      const auto& b = shared.getEventWeightsHostView(actual);
      ASSERT_EQ(a.size(), b.size());
      EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(double)), 0);
    }
  }
  // Inputs without reuse must fall back instead of retaining larger metadata.
  auto unique = *model;
  for( std::size_t i = 0; i < unique.propagation.dialInputs.size(); ++i ){
    unique.propagation.dialInputs[i] = {0, true, double(i), 2.};
  }
  shared.build(unique);
  EXPECT_EQ(shared.getWeightSchedulingDiagnostics().at("cacheInputGroups"), 0);
  EXPECT_EQ(shared.getWeightSchedulingDiagnostics().at("cacheInputStorageBytes"),
            unique.propagation.dials.size() * sizeof(double));
  runtime.parameters.values = {1.};
  const auto token = shared.requestPropagation(runtime);
  EXPECT_FALSE(shared.getLastTimingSummary().sharedCacheInputs);
  EXPECT_EQ(shared.getLastTimingSummary().cacheInputTransformCount, 192);
  EXPECT_EQ(shared.getEventWeightsHostView(token).size(), 192);
  GundamGlobals::setNumberOfThreads(previousThreads);
}
