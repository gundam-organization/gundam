#include "CudaBackend.h"
#include "BackendFactory.h"
#include "BackendManager.h"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

TEST(CudaBackend, FactoryDoesNotRequireADevice) {
  for( const auto* type : {"CUDA", "cuda"} ){
    Backends::BackendManager manager;
    manager.configure(ConfigReader(JsonType{{"type", type}}));
    auto backend = Backends::makeBackend(manager);
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->getName(), "CUDA");
    const auto capabilities = backend->getCapabilities();
    EXPECT_EQ(capabilities.supportsGpu, Backends::CudaBackend::isCompiled());
    EXPECT_FALSE(capabilities.supportsCpu);
    EXPECT_FALSE(capabilities.supportsEventWeights);
    EXPECT_FALSE(capabilities.supportsHistograms);
    EXPECT_FALSE(capabilities.supportsLikelihood);
    EXPECT_FALSE(capabilities.supportsDynamicBinning);
    EXPECT_FALSE(capabilities.supportsObservableTransforms);
  }
}

TEST(CudaBackend, ConfigurationValidatesDeviceIndexWithoutADevice) {
  Backends::CudaBackend backend;
  EXPECT_NO_THROW(backend.configure(ConfigReader(JsonType{{"name", "test"}, {"deviceIndex", 0}})));
  EXPECT_THROW(backend.configure(ConfigReader(JsonType{{"deviceIndex", -1}})), std::runtime_error);
  EXPECT_THROW(backend.configure(ConfigReader(JsonType{{"deviceIndex", "invalid"}})), std::exception);
  EXPECT_THROW(backend.configure(ConfigReader(JsonType{{"deviceIndex", 0.5}})), std::runtime_error);
  EXPECT_THROW(backend.configure(ConfigReader(JsonType{{"deviceIndex",
      static_cast<unsigned long long>(std::numeric_limits<int>::max()) + 1}})), std::runtime_error);
}

TEST(CudaBackend, NeverPublishesResultsOrAcceptsInventedTokens) {
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
