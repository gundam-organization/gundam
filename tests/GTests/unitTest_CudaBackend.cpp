#include "CudaBackend.h"
#include "BackendFactory.h"
#include "BackendManager.h"
#include "CudaWeightModel.h"
#include "EventStateCache.h"
#include "CompactSpline.h"
#include "UniformSpline.h"
#include "MonotonicSpline.h"
#include "GeneralSpline.h"
#include "Graph.h"

#include <gtest/gtest.h>

#include <algorithm>
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
    EXPECT_EQ(capabilities.supportsDynamicBinning, Backends::CudaBackend::isCompiled());
    EXPECT_EQ(capabilities.supportsObservableTransforms, Backends::CudaBackend::isCompiled());
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
  EXPECT_NO_THROW((void)makeCudaWeightModel(view));
  view = {};
  view.propagation.externalWeightBlocks.push_back({0, 1});
  EXPECT_NO_THROW((void)makeCudaWeightModel(view));
  view = {};
  view.propagation.samples.emplace_back().useDynamicBinning = true;
  EXPECT_NO_THROW((void)makeCudaWeightModel(view));
  view = {};
  view.propagation.samples.emplace_back().binning.isSmoothed = true;
  EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
  for( auto type : {BackendDialType::ExternalWeight, BackendDialType::Tabulated, BackendDialType::Kriged} ){
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

TEST(CudaWeightModel, SurfaceDependenciesAreDisjointAndRemappedWithoutChangingProducts) {
  using namespace Backends;
  EngineView view;
  auto& model = view.propagation;
  model.parameterCount = 3;
  model.dialInputs = {{0}, {0}, {1}, {1}, {0}, {1}, {1}, {2}};
  model.dialPayloads = {2., 3., -1., 2., -2., 0., 3., .6, .8, 1.1, 1.2, 1.4, 1.7, 1.25};
  for( std::size_t i = 0; i < 6; ++i ){
    BackendDialDescriptor dial;
    if( i == 0 or i == 2 or i == 5 ){
      dial.type = i == 2 ? BackendDialType::Bicubic : BackendDialType::Bilinear;
      dial.firstInput = i == 0 ? 1 : (i == 2 ? 3 : 5);
      dial.inputCount = 2;
      dial.payloadSize = 13;
    }
    else if( i == 4 ){
      dial.type = BackendDialType::Shift;
      dial.payloadOffset = 13;
      dial.payloadSize = 1;
    }
    else{ dial.firstInput = i == 1 ? 0 : 7; dial.inputCount = 1; }
    model.dials.push_back(dial);
  }
  const std::vector<std::uint32_t> references{5, 0, 2, 1, 4, 3, 0, 0};
  const std::vector<std::uint32_t> mapped{5, 3, 4, 0, 2, 1, 3, 3};
  model.events.resize(67);
  for( std::size_t i = 0; i < model.events.size(); ++i ){
    model.events[i].resultIndex = i;
    model.eventWeights.push_back({.75, model.eventDialIndices.size(), references.size()});
    model.eventDialIndices.insert(model.eventDialIndices.end(), references.begin(), references.end());
  }
  for( bool interleave : {false, true} ){
    const auto packed = makeCudaWeightModel(view, interleave);
    EXPECT_EQ(packed.surfaceOffset, 3);
    EXPECT_EQ(packed.dials[0].type, BackendDialType::Norm);
    EXPECT_EQ(packed.dials[2].type, BackendDialType::Shift);
    EXPECT_EQ(packed.dials[3].type, BackendDialType::Bilinear);
    EXPECT_EQ(packed.dials[4].type, BackendDialType::Bicubic);
    EXPECT_EQ(packed.parameterDialOffsets, (std::vector<std::size_t>{0, 1, 1, 2}));
    EXPECT_EQ(packed.parameterDialIndices, (std::vector<std::uint32_t>{0, 1, 3, 4, 5}));
    ASSERT_EQ(packed.surfaceGroups.size(), 2);
    EXPECT_EQ(packed.eventDialStride, interleave ? 32 : 1);
    for( const auto& event : packed.events ){
      for( std::size_t j = 0; j < event.dialCount; ++j ){
        EXPECT_EQ(packed.eventDialIndices[event.firstDial + j * packed.eventDialStride], mapped[j]);
      }
    }
    CudaResponseUpdatePlan plan;
    const auto selected = [&](const std::vector<double>& values_){
      plan.select(packed.parameterDialOffsets, {1., 1., 1.}, values_, packed.surfaceGroups);
      std::vector<std::uint32_t> indices;
      for( const auto& range : plan.ranges ){
        indices.insert(indices.end(), packed.parameterDialIndices.begin() + range.firstDial,
                       packed.parameterDialIndices.begin() + range.endDial);
      }
      std::sort(indices.begin(), indices.end());
      EXPECT_EQ(indices.size(), plan.dialCount);
      return indices;
    };
    EXPECT_EQ(selected({2., 1., 1.}), (std::vector<std::uint32_t>{0, 3, 4}));
    EXPECT_EQ(selected({1., 2., 1.}), (std::vector<std::uint32_t>{3, 4, 5}));
    EXPECT_EQ(selected({2., 2., 1.}), (std::vector<std::uint32_t>{0, 3, 4, 5}));
    EXPECT_EQ(selected({1., 1., 2.}), (std::vector<std::uint32_t>{1}));
    EXPECT_TRUE(selected({1., 1., 1.}).empty());
  }
}

TEST(CudaWeightModel, RejectsMalformedSurfacesAndInvalidSecondInputs) {
  using namespace Backends;
  EngineView valid;
  auto& model = valid.propagation;
  model.parameterCount = 2;
  model.dialInputs = {{0}, {1}};
  model.dialPayloads = {2., 2., -1., 1., -2., 2., .5, 1., 1.5, 2.};
  model.dials.resize(1);
  model.dials[0].inputCount = 2;
  model.dials[0].payloadSize = model.dialPayloads.size();
  for( auto type : {BackendDialType::Bilinear, BackendDialType::Bicubic} ){
    model.dials[0].type = type;
    EXPECT_NO_THROW((void)makeCudaWeightModel(valid));
    for( const auto& bad : std::vector<std::pair<std::size_t, double>>{
        {0, 1.}, {0, 2.5}, {1, 3.}, {3, -1.}, {5, -2.}, {6, std::numeric_limits<double>::infinity()}} ){
      auto view = valid;
      view.propagation.dialPayloads[bad.first] = bad.second;
      EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
    }
    auto view = valid;
    view.propagation.dialInputs[1].parameterIndex = 2;
    EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
    view = valid;
    view.propagation.dialInputs[1].source = DialInputSource::EventVariable;
    EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
    view = valid;
    view.propagation.dialInputs[1].useMirror = true;
    EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
    view = valid;
    view.propagation.dials[0].firstInput = 1;
    EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
    view = valid;
    view.propagation.dials[0].inputCount = 1;
    EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
  }
}

TEST(CudaWeightModel, ExternalAndTableGroupsUseProducerGenerations) {
  using namespace Backends;
  EngineView view;
  auto& model = view.propagation;
  model.parameterCount = 1;
  model.externalWeightBlocks = {{8, 3}, {0, 2}, {42, 0}}; // Gaps, reordered blocks, empty producer.
  model.dialInputs = {{0}};
  BackendDialDescriptor norm;
  norm.inputCount = 1;
  BackendDialDescriptor external;
  external.type = BackendDialType::ExternalWeight;
  external.externalWeightIndex = 2;
  external.inputCount = 99; // Producer dependencies are not device parameter inputs.
  BackendDialDescriptor tabulated;
  tabulated.type = BackendDialType::Tabulated;
  tabulated.externalBlockIndex = 1;
  tabulated.tableTermCount = 2;
  BackendDialDescriptor kriged;
  kriged.type = BackendDialType::Kriged;
  kriged.firstTableTerm = 2;
  kriged.tableTermCount = 3;
  BackendDialDescriptor empty = kriged;
  empty.externalBlockIndex = 2;
  empty.tableTermCount = 0;
  model.tableTerms = {{0, .25}, {1, .75}, {2, -.25}, {0, .5}, {2, .75}};
  model.dials = {external, norm, tabulated, kriged, empty};
  model.events.resize(1);
  model.eventWeights = {{2., 0, 6}};
  model.eventDialIndices = {0, 1, 2, 3, 4, 0};
  const auto packed = makeCudaWeightModel(view);
  EXPECT_EQ(packed.externalWeightCount, 11);
  EXPECT_EQ(packed.externalDialCount, 4);
  EXPECT_EQ(packed.parameterDialOffsets, (std::vector<std::size_t>{0, 1}));
  EXPECT_EQ(packed.parameterDialIndices, (std::vector<std::uint32_t>{1, 0, 3, 2, 4}));
  EXPECT_EQ(packed.eventDialIndices, model.eventDialIndices);
  ASSERT_EQ(packed.externalGroups.size(), 3);
  EXPECT_EQ(packed.externalGroups[0].blockIndex, 0);
  EXPECT_EQ(packed.externalGroups[0].firstDial, 1);
  EXPECT_EQ(packed.externalGroups[0].endDial, 3);
  CudaResponseUpdatePlan plan;
  plan.selectExternal(packed.externalGroups, {0, 0, 0}, {1, 1, 1});
  EXPECT_EQ(plan.dialCount, 4);
  EXPECT_EQ(plan.blockCount, 3);
  plan.selectExternal(packed.externalGroups, {1, 1, 1}, {2, 1, 1});
  ASSERT_EQ(plan.ranges.size(), 1);
  EXPECT_EQ(plan.dialCount, 2);
  EXPECT_EQ(plan.ranges[0].firstDial, 1);
  plan.selectExternal(packed.externalGroups, {2, 1, 1}, {2, 1, 1});
  EXPECT_EQ(plan.dialCount, 0);
  EXPECT_TRUE(plan.ranges.empty());
  plan.selectExternal(packed.externalGroups, {2, 1, 1}, {2, 1, 1}, true);
  EXPECT_EQ(plan.dialCount, 4);
  // A decreasing but different revision still invalidates its producer block.
  plan.selectExternal(packed.externalGroups, {2, 1, 1}, {1, 1, 1});
  EXPECT_EQ(plan.dialCount, 2);
  EXPECT_THROW(plan.selectExternal(packed.externalGroups, {1}, {1}), std::runtime_error);
  EXPECT_THROW(plan.selectExternal(packed.externalGroups, {}, {1}), std::runtime_error);
}

TEST(CudaWeightModel, RejectsMalformedExternalBlocksAndTableTerms) {
  using namespace Backends;
  EngineView good;
  auto& model = good.propagation;
  model.externalWeightBlocks = {{0, 3}, {3, 2}};
  BackendDialDescriptor external;
  external.type = BackendDialType::ExternalWeight;
  BackendDialDescriptor table;
  table.type = BackendDialType::Tabulated;
  table.externalBlockIndex = 1;
  table.tableTermCount = 2;
  model.dials = {external, table};
  model.tableTerms = {{0, .25}, {1, .75}};
  EXPECT_NO_THROW((void)makeCudaWeightModel(good));
  auto bad = good;
  bad.propagation.dials[0].externalBlockIndex = 2;
  EXPECT_THROW((void)makeCudaWeightModel(bad), std::runtime_error);
  bad = good;
  bad.propagation.dials[0].externalWeightIndex = 3;
  EXPECT_THROW((void)makeCudaWeightModel(bad), std::runtime_error);
  bad = good;
  bad.propagation.dials[1].externalBlockIndex = 2;
  EXPECT_THROW((void)makeCudaWeightModel(bad), std::runtime_error);
  bad = good;
  bad.propagation.tableTerms.pop_back();
  EXPECT_THROW((void)makeCudaWeightModel(bad), std::runtime_error);
  bad = good;
  bad.propagation.dials[1].tableTermCount = 1;
  EXPECT_THROW((void)makeCudaWeightModel(bad), std::runtime_error);
  bad = good;
  bad.propagation.tableTerms[0].index = 2;
  EXPECT_THROW((void)makeCudaWeightModel(bad), std::runtime_error);
  for( auto invalid : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()} ){
    bad = good;
    bad.propagation.tableTerms[1].coefficient = invalid;
    EXPECT_THROW((void)makeCudaWeightModel(bad), std::runtime_error);
  }
  bad = good;
  bad.propagation.externalWeightBlocks[1].offset = 2;
  EXPECT_THROW((void)makeCudaWeightModel(bad), std::runtime_error);
  bad = good;
  bad.propagation.externalWeightBlocks[1].offset = std::numeric_limits<std::size_t>::max();
  EXPECT_THROW((void)makeCudaWeightModel(bad), std::runtime_error);
  bad = good;
  bad.propagation.externalWeightBlocks = {{0, std::numeric_limits<std::size_t>::max()}};
  EXPECT_THROW((void)makeCudaWeightModel(bad), std::runtime_error);
}

TEST(CudaWeightModel, ObservableReferencesAreRemappedAndUnsafeLayoutsRejected) {
  using namespace Backends;
  EngineView good;
  auto& model = good.propagation;
  model.parameterCount = 2;
  model.totalBins = 1;
  model.hasVariableResponses = true;
  model.dialInputs = {{0}, {1}};
  model.dialPayloads = {2., 2., -1., 1., -1., 1., 1., 2., 3., 4.};
  BackendDialDescriptor surface;
  surface.type = BackendDialType::Bilinear;
  surface.inputCount = 2;
  surface.payloadSize = 10;
  BackendDialDescriptor norm;
  norm.inputCount = 1;
  model.dials = {surface, norm};
  model.variableValues = {.2, .3, .4, .5};
  model.variableDialIndices = {0, 1};
  BackendVariableResponseDescriptor target;
  target.overrideDial = 0;
  target.firstAddDial = 1;
  target.addDialCount = 1;
  model.variableResponses = {target, target};
  model.variableResponses[1].variableIndex = 2;
  SampleView sample;
  sample.useDynamicBinning = true;
  sample.binCount = 1;
  sample.binning.variableNames = {"X"};
  sample.binning.bins = {{0, 0, 1}};
  sample.binning.intervals = {{0, 0., 1.}};
  model.samples = {sample};
  model.events.resize(2);
  model.eventWeights.resize(2);
  for( std::size_t i = 0; i < 2; ++i ){
    auto& event = model.events[i];
    event.resultIndex = i;
    event.binIndex = event.globalBinIndex = 0;
    event.samplePosition = 0;
    event.firstVariable = 2 * i;
    event.variableCount = 2;
    event.firstVariableResponse = i;
    event.variableResponseCount = 1;
  }
  for( bool interleaved : {false, true} ){
    const auto packed = makeCudaWeightModel(good, interleaved);
    EXPECT_TRUE(packed.hasEventState);
    EXPECT_EQ(packed.variableDialIndices, (std::vector<std::uint32_t>{1, 0}));
    EXPECT_EQ(packed.variableResponses[0].overrideDial, 1);
    EXPECT_EQ(packed.variableResponses[1].overrideDial, 1);
    EXPECT_EQ(packed.variableValues, model.variableValues);
    EXPECT_EQ(packed.samples.size(), 1);
    EXPECT_EQ(packed.stateEvents.size(), 2);
  }
  for( int error = 0; error < 16; ++error ){
    auto bad = good;
    auto& m = bad.propagation;
    switch( error ){
      case 0: m.variableResponses[0].variableIndex = 2; break;
      case 1: m.variableResponses[0].referenceValue = std::numeric_limits<double>::quiet_NaN(); break;
      case 2: m.variableResponses[0].overrideDial = 2; break;
      case 3: m.variableResponses[0].firstAddDial = 3; break;
      case 4: m.variableResponses[0].multiplyDialCount = 3; break;
      case 5: m.variableDialIndices[0] = 2; break;
      case 6: m.events[0].variableCount = 5; break;
      case 7: m.events[0].variableResponseCount = 3; break;
      case 8: m.events[0].samplePosition = 2; break;
      case 9: m.events[1].firstVariable = 0; m.variableResponses[1].variableIndex = 0; break;
      case 10: m.samples[0].binning.intervals[0].variableIndex = 1; break;
      case 11: m.samples[0].binning.intervals[0].lowerBound = std::numeric_limits<double>::quiet_NaN(); break;
      case 12: m.samples[0].binning.isSmoothed = true; break;
      case 13: m.samples[0].binning.bins[0].binIndex = 1; break;
      case 14: m.samples[0].binOffset = -1; break;
      case 15: m.variableValues[0] = std::numeric_limits<double>::infinity(); break;
    }
    SCOPED_TRACE(error);
    EXPECT_THROW((void)makeCudaWeightModel(bad), std::runtime_error);
  }
}

TEST(CudaWeightModel, ExternalObservableCacheUsesPublicationRatherThanProducerInputs) {
  using namespace Backends;
  PropagationView model;
  model.externalWeightBlocks = {{0, 1}};
  BackendDialDescriptor external;
  external.type = BackendDialType::ExternalWeight;
  external.inputCount = 99; // No evaluator inputs; producer metadata is ignored.
  model.dials = {external};
  BackendVariableResponseDescriptor target;
  target.overrideDial = 0;
  model.variableResponses = {target};
  EventStateCache cache;
  EXPECT_NO_THROW(cache.build(model));
  double value = 1.;
  PropagationInputs inputs;
  inputs.externalWeights = {{&value, 1, 0, 1}};
  cache.prepare(inputs);
  EXPECT_TRUE(cache.variableInputsChanged());
  cache.commit(inputs, {});
  cache.prepare(inputs);
  EXPECT_FALSE(cache.needsUpdate(inputs));
  ++inputs.externalWeights[0].generation;
  cache.prepare(inputs);
  EXPECT_TRUE(cache.variableInputsChanged());
}
