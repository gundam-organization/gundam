#include "CudaBackend.h"
#include "BackendFactory.h"
#include "BackendManager.h"
#include "CudaWeightModel.h"
#include "CompactSpline.h"
#include "UniformSpline.h"
#include "MonotonicSpline.h"
#include "GeneralSpline.h"
#include "Graph.h"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

TEST(CudaBackend, FactoryDoesNotRequireADevice) {
  for( const auto* type : {"CUDA", "cuda"} ){
    Backends::BackendManager manager;
    manager.configure(ConfigReader(JsonType{{"type", type}}));
    auto backend = Backends::makeBackend(manager);
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->getName(), "CUDA/cachedResponses");
    const auto capabilities = backend->getCapabilities();
    EXPECT_EQ(capabilities.supportsGpu, Backends::CudaBackend::isCompiled());
    EXPECT_FALSE(capabilities.supportsCpu);
    EXPECT_EQ(capabilities.supportsEventWeights, Backends::CudaBackend::isCompiled());
    EXPECT_EQ(capabilities.supportsHistograms, Backends::CudaBackend::isCompiled());
    EXPECT_EQ(capabilities.supportsLikelihood, Backends::CudaBackend::isCompiled());
    EXPECT_FALSE(capabilities.supportsDynamicBinning);
    EXPECT_FALSE(capabilities.supportsObservableTransforms);
  }
}

TEST(CudaBackend, ConfigurationValidatesDeviceIndexWithoutADevice) {
  Backends::CudaBackend backend;
  EXPECT_NO_THROW(backend.configure(ConfigReader(JsonType{{"name", "test"}, {"deviceIndex", 0}})));
  EXPECT_NO_THROW(backend.configure(ConfigReader(JsonType{{"propagationMode", "recomputeResponses"}})));
  EXPECT_EQ(backend.getName(), "CUDA/recomputeResponses");
  EXPECT_THROW(backend.configure(ConfigReader(JsonType{{"propagationMode", "invalid"}})), std::exception);
  EXPECT_THROW(backend.configure(ConfigReader(JsonType{{"propagationMode", 12}})), std::exception);
  EXPECT_THROW(backend.configure(ConfigReader(JsonType{{"deviceIndex", -1}})), std::runtime_error);
  EXPECT_THROW(backend.configure(ConfigReader(JsonType{{"deviceIndex", "invalid"}})), std::exception);
  EXPECT_THROW(backend.configure(ConfigReader(JsonType{{"deviceIndex", 0.5}})), std::runtime_error);
  EXPECT_NO_THROW(backend.configure(ConfigReader(JsonType{{"weightLayout", "warpInterleaved"}})));
  EXPECT_THROW(backend.configure(ConfigReader(JsonType{{"weightLayout", "invalid"}})), std::exception);
  EXPECT_THROW(backend.configure(ConfigReader(JsonType{{"weightLayout", 1}})), std::exception);
  EXPECT_THROW(backend.configure(ConfigReader(JsonType{{"deviceIndex",
      static_cast<unsigned long long>(std::numeric_limits<int>::max()) + 1}})), std::runtime_error);
}

TEST(CudaBackend, UnbuiltBackendRejectsResultsAndInventedTokens) {
  Backends::CudaBackend backend;
  EXPECT_THROW((void)backend.getEngineView(), std::runtime_error);
  EXPECT_THROW(backend.buildShared(nullptr), std::runtime_error);
  EXPECT_THROW(backend.requestPropagation({}), std::runtime_error);
  for( const auto& token : {Backends::PropagationToken{}, Backends::PropagationToken{1, true}} ){
    EXPECT_FALSE(backend.isReady(token));
    const auto status = backend.getStatus(token);
    EXPECT_EQ(status.backend, Backends::BackendStatus::Failed);
    EXPECT_EQ(status.eventWeights, Backends::OutputState::NotRequested);
    EXPECT_EQ(status.histograms, Backends::OutputState::NotRequested);
    EXPECT_EQ(status.statLikelihood, Backends::OutputState::NotRequested);
    EXPECT_THROW(backend.wait(token), std::runtime_error);
    EXPECT_THROW(backend.materialize(token, Backends::OutputRequest::Histograms), std::runtime_error);
    EXPECT_THROW((void)backend.getLikelihood(token), std::runtime_error);
    EXPECT_THROW((void)backend.getEventWeightsHostView(token), std::runtime_error);
    EXPECT_THROW((void)backend.getHistogramSumsHostView(token), std::runtime_error);
    EXPECT_THROW((void)backend.getHistogramSumSquaresHostView(token), std::runtime_error);
    EXPECT_THROW((void)backend.getEventStateHostView(token), std::runtime_error);
    EXPECT_THROW((void)backend.getOutsideSumsHostView(token), std::runtime_error);
    EXPECT_THROW((void)backend.getOutsideSumSquaresHostView(token), std::runtime_error);
    EXPECT_THROW((void)backend.getDeviceView(token), std::runtime_error);
  }
}

TEST(CudaBackend, MissingToolkitIsReportedWhenBuilding) {
  if( Backends::CudaBackend::isCompiled() ){ return; }
  Backends::CudaBackend backend;
  backend.configure(ConfigReader(JsonType::object()));
  EXPECT_THROW(backend.initialize(), std::runtime_error);
  try{
    backend.build(Backends::EngineView{});
    FAIL() << "Building without CUDA must fail.";
  }
  catch(const std::runtime_error& error){
    EXPECT_NE(std::string(error.what()).find("WITH_CUDA_LIB=ON"), std::string::npos);
  }
  EXPECT_THROW((void)backend.getEngineView(), std::runtime_error);
}

TEST(CudaWeightModel, PreservesResultOrderAndFixedBinMembership) {
  using namespace Backends;
  EngineView view;
  view.propagation.totalBins = 3;
  view.propagation.eventWeights = {{2., 0, 0}, {-3., 0, 0}, {4., 0, 0}};
  for( const auto& slot : {std::pair<int, int>{2, 0}, {0, 0}, {1, -1}} ){
    EventView event;
    event.resultIndex = slot.first;
    event.globalBinIndex = slot.second;
    view.propagation.events.push_back(event);
  }
  const auto packed = makeCudaWeightModel(view);
  EXPECT_EQ(packed.binOffsets, (std::vector<std::size_t>{0, 2, 2, 2}));
  EXPECT_EQ(packed.binEvents, (std::vector<std::size_t>{2, 0}));
  EXPECT_EQ(packed.events[0].baseWeight, 2.);
  EXPECT_EQ(packed.events[1].baseWeight, -3.);
  EXPECT_EQ(packed.events[2].baseWeight, 4.);
}

TEST(CudaWeightModel, RejectsUnsupportedFeaturesAndUnsafeLayouts) {
  using namespace Backends;
  EngineView view;
  EXPECT_NO_THROW((void)makeCudaWeightModel(view));
  view.propagation.totalBins = -1;
  EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
  view = {};
  view.propagation.hasVariableResponses = true;
  EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
  view = {};
  view.propagation.externalWeightBlocks.push_back({0, 1});
  EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
  view = {};
  view.propagation.samples.emplace_back().useDynamicBinning = true;
  EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
  view = {};
  view.propagation.samples.emplace_back().binning.isSmoothed = true;
  EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
  for( auto type : {BackendDialType::ExternalWeight, BackendDialType::Bilinear,
                    BackendDialType::Bicubic, BackendDialType::Tabulated, BackendDialType::Kriged} ){
    view = {};
    view.propagation.dials.emplace_back().type = type;
    EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
  }
  view = {};
  view.propagation.dials.emplace_back().inputCount = 1;
  EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
  view.propagation.parameterCount = 1;
  view.propagation.dialInputs = {{0}};
  EXPECT_NO_THROW((void)makeCudaWeightModel(view));
  view.propagation.dialInputs[0].useMirror = true;
  EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
  view.propagation.dialInputs[0].mirrorRange = 1.;
  EXPECT_NO_THROW((void)makeCudaWeightModel(view));
  view.propagation.dialInputs[0].source = DialInputSource::EventVariable;
  EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
  view = {};
  view.propagation.events.resize(1);
  view.propagation.eventWeights.resize(1);
  view.propagation.events[0].globalBinIndex = 0;
  EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
  view.propagation.totalBins = 1;
  EXPECT_NO_THROW((void)makeCudaWeightModel(view));
  view.propagation.events[0].resultIndex = 1;
  EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
  view.propagation.events[0].resultIndex = 0;
  view.propagation.eventWeights[0].dialCount = 1;
  EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
  view.propagation.eventDialIndices = {0};
  EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
  view = {};
  view.likelihood.samples.emplace_back().dataSums = {1.};
  EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
}

TEST(CudaWeightModel, RejectsMalformedSplineAndGraphPayloads) {
  using namespace Backends;
  EngineView view;
  auto& model = view.propagation;
  model.parameterCount = 1;
  model.dialInputs = {{0}};
  model.dials.resize(1);
  auto& dial = model.dials[0];
  dial.inputCount = 1;
  for( auto type : {BackendDialType::Shift, BackendDialType::CompactSpline, BackendDialType::UniformSpline,
                    BackendDialType::MonotonicSpline, BackendDialType::GeneralSpline, BackendDialType::Graph} ){
    dial.type = type;
    EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
  }
  dial.type = BackendDialType::Graph;
  model.dialPayloads = {1., 0., 2., 1.};
  dial.payloadSize = 4;
  EXPECT_NO_THROW((void)makeCudaWeightModel(view));
  model.dialPayloads[3] = 0.;
  EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
  dial.type = BackendDialType::CompactSpline;
  model.dialPayloads = {0., 1., 1., 2.};
  EXPECT_NO_THROW((void)makeCudaWeightModel(view));
  model.dialPayloads[1] = 0.;
  EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
}

TEST(CudaWeightModel, AcceptsPayloadsBuiltByNativeScalarDials) {
  using namespace Backends;
  const std::vector<DialUtils::DialPoint> points{{0., .8, .1}, {1., 1., .1}, {2., 1.2, .1}};
  CompactSpline compact; compact.buildDial(points);
  UniformSpline uniform; uniform.buildDial(points);
  MonotonicSpline monotonic; monotonic.buildDial(points);
  GeneralSpline general; general.buildDial({{0., .8, .1}, {.5, 1., .2}, {2., 1.2, .1}});
  Graph graph; graph.buildDial(points);
  const std::vector<BackendDialType> types{BackendDialType::CompactSpline, BackendDialType::UniformSpline,
      BackendDialType::MonotonicSpline, BackendDialType::GeneralSpline, BackendDialType::Graph};
  const std::vector<std::vector<double>> payloads{compact.getDialData(), uniform.getDialData(),
      monotonic.getDialData(), general.getDialData(), graph.getDialData()};
  for( std::size_t i = 0; i < types.size(); ++i ){
    SCOPED_TRACE(i);
    EngineView view;
    auto& model = view.propagation;
    model.parameterCount = 1;
    model.dialInputs = {{0}};
    model.dials.resize(1);
    model.dials[0].type = types[i];
    model.dials[0].inputCount = 1;
    model.dialPayloads = payloads[i];
    model.dials[0].payloadSize = payloads[i].size();
    EXPECT_NO_THROW((void)makeCudaWeightModel(view));
  }
}

TEST(CudaWeightModel, GroupsScalarDialsByOriginalParameterAndExcludesConstants) {
  using namespace Backends;
  EngineView view;
  auto& model = view.propagation;
  model.parameterCount = 4;
  model.dialInputs = {{2}, {0}, {2}, {1}};
  for( auto input : {0, 1, 2, 3, 0} ){
    BackendDialDescriptor dial;
    dial.firstInput = input;
    dial.inputCount = 1;
    model.dials.push_back(dial);
  }
  BackendDialDescriptor constant;
  constant.type = BackendDialType::Shift;
  constant.payloadSize = 1;
  model.dialPayloads = {.75};
  model.dials.insert(model.dials.begin() + 2, constant);
  const auto packed = makeCudaWeightModel(view);
  EXPECT_EQ(packed.parameterDialOffsets, (std::vector<std::size_t>{0, 1, 2, 5, 5}));
  EXPECT_EQ(packed.parameterDialIndices, (std::vector<std::uint32_t>{1, 4, 0, 3, 5}));
  CudaResponseUpdatePlan plan;
  plan.select(packed.parameterDialOffsets, {1., 2., 3., 4.}, {1., 2., 7., 9.});
  ASSERT_EQ(plan.ranges.size(), 1);
  EXPECT_EQ(plan.ranges[0].firstDial, 2);
  EXPECT_EQ(plan.ranges[0].endDial, 5);
  EXPECT_EQ(plan.dialCount, 3);
}

TEST(CudaWeightModel, SparseDispatchTilesMultipleRangesAndClearsPreviousWork) {
  using namespace Backends;
  CudaResponseUpdatePlan plan;
  // An unused parameter and ranges straddling the 256-thread block boundary.
  const std::vector<std::size_t> offsets{0, 0, 257, 513, 514, 1027};
  const std::vector<double> previous{0., 1., 2., 3., 4.};
  plan.select(offsets, previous, {9., 5., 2., 7., 8.});
  ASSERT_EQ(plan.ranges.size(), 3);
  EXPECT_EQ(plan.dialCount, 771);
  EXPECT_EQ(plan.blockCount, 6);
  EXPECT_EQ(plan.ranges[0].firstDial, 0);
  EXPECT_EQ(plan.ranges[0].endDial, 257);
  EXPECT_EQ(plan.ranges[0].firstBlock, 0);
  EXPECT_EQ(plan.ranges[1].firstDial, 513);
  EXPECT_EQ(plan.ranges[1].endDial, 514);
  EXPECT_EQ(plan.ranges[1].firstBlock, 2);
  EXPECT_EQ(plan.ranges[2].firstDial, 514);
  EXPECT_EQ(plan.ranges[2].endDial, 1027);
  EXPECT_EQ(plan.ranges[2].firstBlock, 3);
  plan.select(offsets, previous, {-0., 1., 2., 3., 4.});
  EXPECT_TRUE(plan.ranges.empty());
  EXPECT_EQ(plan.dialCount, 0);
  EXPECT_EQ(plan.blockCount, 0);
  EXPECT_THROW(plan.select(offsets, {}, previous), std::runtime_error);
  plan.select({0}, {}, {});
  EXPECT_TRUE(plan.ranges.empty());
}

TEST(CudaWeightModel, InterleavedIndicesPreservePerEventOrderAndBoundPadding) {
  using namespace Backends;
  EngineView view;
  auto& model = view.propagation;
  model.parameterCount = 1;
  model.dialInputs = {{0}};
  model.dials.resize(7);
  for( auto& dial : model.dials ){ dial.inputCount = 1; }
  model.events.resize(67);
  for( std::size_t i = 0; i < model.events.size(); ++i ){
    model.events[i].resultIndex = i;
    BackendEventWeightDescriptor event;
    event.baseWeight = .125 * i;
    event.firstDial = model.eventDialIndices.size();
    event.dialCount = i % 11 == 0 ? 0 : 5 + i % 4;
    for( std::size_t j = 0; j < event.dialCount; ++j ){ model.eventDialIndices.push_back((i + 3 * j) % 7); }
    model.eventWeights.push_back(event);
  }
  const auto packed = makeCudaWeightModel(view, true);
  ASSERT_EQ(packed.eventDialStride, 32);
  EXPECT_LE(packed.eventDialIndices.size(), 2 * model.eventDialIndices.size());
  for( std::size_t i = 0; i < packed.events.size(); ++i ){
    const auto& before = model.eventWeights[i];
    const auto& after = packed.events[i];
    EXPECT_EQ(before.baseWeight, after.baseWeight);
    EXPECT_EQ(before.dialCount, after.dialCount);
    for( std::size_t j = 0; j < before.dialCount; ++j ){
      EXPECT_EQ(model.eventDialIndices[before.firstDial + j], packed.eventDialIndices[after.firstDial + 32 * j]);
    }
  }
  // A single long event per warp must not inflate the index storage 32-fold.
  for( auto& event : model.eventWeights ){ event.dialCount = 0; }
  model.eventWeights[0].dialCount = model.eventDialIndices.size();
  model.eventWeights[0].firstDial = 0;
  const auto fallback = makeCudaWeightModel(view, true);
  EXPECT_EQ(fallback.eventDialStride, 1);
  EXPECT_EQ(fallback.eventDialIndices, model.eventDialIndices);
}
