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
