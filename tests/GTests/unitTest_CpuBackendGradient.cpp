#include "CpuBackend.h"
#ifdef __APPLE__
#include "MpsBackend.h"
#endif
#ifdef GUNDAM_TEST_CUDA_DEVICE
#include "CudaBackend.h"
#endif
#include "GradientModel.h"
#include "Semantics/BackendDeviceGradient.h"
#include "Semantics/BackendDialGradient.h"
#include "Graph.h"
#include "CompactSpline.h"
#include "UniformSpline.h"
#include "MonotonicSpline.h"
#include "GeneralSpline.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>

namespace {
  using namespace Backends;

  EngineView gradientModel(double rate_, int count_ = 128) {
    EngineView view;
    auto& model = view.propagation;
    model.parameterCount = 2;
    model.totalBins = 2;
    model.dialInputs = {{0}, {1}};
    BackendDialDescriptor norm;
    norm.inputCount = 1;
    model.dials.push_back(norm);
    norm.firstInput = 1;
    model.dials.push_back(norm);
    // Repeated input and zero responses exercise product rules without division.
    for( int i = 0; i < count_; ++i ){
      EventView event;
      event.sampleIndex = event.binIndex = event.globalBinIndex = 0;
      event.resultIndex = model.events.size();
      model.events.push_back(event);
      model.eventWeights.push_back({rate_ * (.1 + .03*(i%3)), model.eventDialIndices.size(), std::size_t(2+i%2)});
      model.eventDialIndices.insert(model.eventDialIndices.end(), {0, 1});
      if( i%2 ){ model.eventDialIndices.push_back(0); }
    }
    for( int bin = 0; bin < 2; ++bin ){
      EventView event;
      event.sampleIndex = 0;
      event.binIndex = event.globalBinIndex = bin;
      event.resultIndex = model.events.size();
      model.events.push_back(event);
      model.eventWeights.push_back({rate_, 0, 0});
    }
    LikelihoodSampleView sample;
    sample.dataSums = {rate_ * 12., rate_ * 100.};
    sample.ignoredBins = {false, true};
    sample.portableOptions.kind = LikelihoodKind::BarlowBeestonLiteGaussian;
    view.likelihood.samples.push_back(sample);
    return view;
  }

  void compareGradient(Backend& backend_, const std::vector<double>& point_, double absoluteTolerance_ = 1e-14) {
    PropagationInputs inputs;
    inputs.parameters.values = point_;
    const auto token = backend_.requestPropagation(inputs);
    const auto gradient = backend_.getLikelihoodGradient(token);
    ASSERT_EQ(gradient.size(), point_.size());
    EXPECT_EQ(backend_.getStatus(token).histograms, OutputState::ReadyOnDevice);
    for( std::size_t i = 0; i < point_.size(); ++i ){
      const double h = 2e-5;
      inputs.parameters.values = point_;
      inputs.parameters.values[i] += h;
      const double up = backend_.getLikelihood(backend_.requestPropagation(inputs));
      inputs.parameters.values[i] -= 2*h;
      const double down = backend_.getLikelihood(backend_.requestPropagation(inputs));
      const double numeric = (up-down)/(2*h);
      EXPECT_NEAR(gradient[i], numeric, 3e-5 * std::max(1e-20, std::abs(numeric)) + absoluteTolerance_);
    }
    EXPECT_THROW(static_cast<void>(backend_.getLikelihoodGradient(token)), std::exception);
  }
}

TEST(CpuBackendGradient, FiniteDifferencesForLowRatesZeroResponsesRepeatedInputsAndCap) {
  for( const double rate : {1e-12, 1., 1e8} ){
    for( const int count : {1, 128} ){
      SCOPED_TRACE(::testing::Message() << rate << ", events=" << count);
      CpuBackend backend;
      auto view = gradientModel(rate, count);
      backend.build(view);
      EXPECT_TRUE(backend.getCapabilities().supportsLikelihoodGradient);
      for( const auto& point : {std::vector<double>{.8, 1.1}, {0., .9}, {0., 0.}} ){
        compareGradient(backend, point);
      }
      view.propagation.hasReweightCap = true;
      view.propagation.maxReweight = .5;
      backend.build(view);
      compareGradient(backend, {1.2, 1.3});
      PropagationInputs inputs;
      inputs.parameters.values = {1.2, 1.3};
      EXPECT_EQ(backend.getLikelihoodGradient(backend.requestPropagation(inputs)), (std::vector<double>{0., 0.}));
    }
  }
}

TEST(CpuBackendGradient, SimpleDialDerivativesMatchResponsesAndMirrorClamps) {
  std::vector<DialUtils::DialPoint> points{{-2., .7, .15}, {-1., .9, .15}, {0., 1., .15}, {1., 1.25, .15}, {2., 1.4, .15}};
  CompactSpline compact; compact.buildDial(points);
  UniformSpline uniform; uniform.buildDial(points);
  MonotonicSpline monotonic; monotonic.buildDial(points);
  GeneralSpline general; general.buildDial(points);
  Graph graph; graph.buildDial(points);
  const std::vector<std::pair<BackendDialType, std::vector<double>>> cases{
      {BackendDialType::CompactSpline, compact.getDialData()},
      {BackendDialType::UniformSpline, uniform.getDialData()},
      {BackendDialType::MonotonicSpline, monotonic.getDialData()},
      {BackendDialType::GeneralSpline, general.getDialData()},
      {BackendDialType::Graph, graph.getDialData()}, {BackendDialType::Norm, {}}};
  for( const auto& item : cases ){
    BackendDialDescriptor dial;
    dial.type = item.first;
    dial.inputCount = 1;
    dial.payloadSize = item.second.size();
    BackendDialInputDescriptor input{0};
    for( bool mirror : {false, true} ){
      input.useMirror = mirror;
      input.mirrorMin = -2.; input.mirrorRange = 4.;
      for( bool extrapolate : {false, true} ){
        dial.allowExtrapolation = extrapolate;
        for( double x : {-5.7, -2.5, -.27, .31, 1.61, 2.7, 5.3} ){
          for( bool clamp : {false, true} ){
            dial.hasMinResponse = dial.hasMaxResponse = clamp;
            dial.minResponse = .95; dial.maxResponse = 1.2;
            auto eval = [&](double raw_){ return Semantics::evalDialResponse(dial, &input, item.second.data(), &raw_); };
            const double numeric = (eval(x+1e-6)-eval(x-1e-6))/2e-6;
            const double analytic = Semantics::evalWeightDialGradient(dial, input, item.second.data(), x);
            EXPECT_NEAR(analytic, numeric, 2e-7);
          }
        }
      }
    }
    if( dial.type != BackendDialType::Norm ){
      dial.allowExtrapolation = false;
      input.useMirror = false;
      EXPECT_EQ(Semantics::evalWeightDialGradient(dial, input, item.second.data(), -2.), 0.);
      EXPECT_EQ(Semantics::evalWeightDialGradient(dial, input, item.second.data(), 2.), 0.);
    }
  }
}

TEST(CpuBackendGradient, RejectsUnsupportedLikelihoodAndSmoothing) {
  CpuBackend backend;
  auto view = gradientModel(1.);
  view.likelihood.samples[0].portableOptions.kind = LikelihoodKind::Poisson;
  backend.build(view);
  EXPECT_THROW(backend.validateLikelihoodGradient(), std::exception);
  view.likelihood.samples[0].portableOptions.kind = LikelihoodKind::BarlowBeestonLiteGaussian;
  view.propagation.dials[0].smoothing.minWidth = .1;
  backend.build(view);
  EXPECT_THROW(backend.validateLikelihoodGradient(), std::exception);

}

namespace {
  struct HostGradientMath : Backends::LikelihoodSemantics::DoubleMath {
    static double remainder(double a_, double b_) { return std::fmod(a_, b_); }
    static int integer(double x_) { return int(x_); }
  };
}
TEST(DeviceGradient, PortableDialDerivativesMatchCpuReference) {
  std::vector<DialUtils::DialPoint> points{{-2., .7, .15}, {-1., .9, .15}, {0., 1., .15}, {1., 1.25, .15}, {2., 1.4, .15}};
  CompactSpline compact; compact.buildDial(points);
  UniformSpline uniform; uniform.buildDial(points);
  MonotonicSpline monotonic; monotonic.buildDial(points);
  GeneralSpline general; general.buildDial(points);
  Graph graph; graph.buildDial(points);
  const std::vector<std::pair<BackendDialType, std::vector<double>>> cases{
      {BackendDialType::CompactSpline, compact.getDialData()}, {BackendDialType::UniformSpline, uniform.getDialData()},
      {BackendDialType::MonotonicSpline, monotonic.getDialData()}, {BackendDialType::GeneralSpline, general.getDialData()},
      {BackendDialType::Graph, graph.getDialData()}, {BackendDialType::Norm, {}}, {BackendDialType::Shift, {.83}}};
  for( const auto& item : cases ){
    SCOPED_TRACE(unsigned(item.first));
    for( bool mirror : {false, true} ) for( bool extrapolate : {false, true} ){
      auto view = gradientModel(1.);
      auto& model = view.propagation;
      model.dials.resize(1);
      auto& dial = model.dials[0];
      dial.type = item.first; dial.payloadSize = item.second.size(); dial.allowExtrapolation = extrapolate;
      model.dialPayloads = item.second;
      model.dialInputs[0] = {0, mirror, -2., 4.};
      model.events.clear(); model.eventWeights.clear(); model.eventDialIndices.clear();
      GradientModel<double> packed; packed.build(view);
      for( double x : {-6.7, -2.5, -2., -.27, .31, 1.61, 2., 2.7, 6.3} ){
        const auto actual = GradientSemantics::evaluateDial<HostGradientMath>(packed.dials[0], item.second.data(), x);
        const double expected = Semantics::evalDialResponse(dial, model.dialInputs.data(), item.second.data(), &x);
        const double slope = item.first == BackendDialType::Shift ? 0. : Semantics::evalWeightDialGradient(dial, model.dialInputs[0], item.second.data(), x);
        EXPECT_NEAR(actual.value, expected, 3e-13*std::max(1., std::abs(expected)));
        EXPECT_NEAR(actual.derivative, slope, 3e-13*std::max(1., std::abs(slope)));
      }
    }
  }
}

#ifdef __APPLE__
TEST(DeviceGradient, MetalMatchesCpuWithResidentBinsAndRejectsStaleTokens) {
  for( const char* mode : {"cachedResponses", "recomputeResponses"} ){
    MpsBackend backend;
    if( !backend.getCapabilities().supportsGpu ) GTEST_SKIP() << "Metal unavailable";
    backend.configure(ConfigReader(JsonType{{"propagationMode", mode}}));
    for( double rate : {1e-8, 1., 1e8} ){
      auto view = gradientModel(rate);
      backend.build(view);
      for( auto point : {std::vector<double>{.8, 1.1}, {0., .9}, {0., 0.}} ){
        SCOPED_TRACE(::testing::Message() << mode << ", rate=" << rate << ", p=" << point[0]);
        compareGradient(backend, point, 2e-7);
        CpuBackend cpu; cpu.build(view);
        PropagationInputs inputs; inputs.parameters.values = point;
        const auto reference = cpu.getLikelihoodGradient(cpu.requestPropagation(inputs));
        const auto actual = backend.getLikelihoodGradient(backend.requestPropagation(inputs));
        for( std::size_t i = 0; i < actual.size(); ++i ) EXPECT_NEAR(actual[i], reference[i], 1e-14 + 3e-8*std::abs(reference[i]));
      }
      view.propagation.hasReweightCap = true; view.propagation.maxReweight = .5;
      backend.build(view);
      compareGradient(backend, {1.2, 1.3}, 2e-7);
    }
  }
}
#endif
#ifdef GUNDAM_TEST_CUDA_DEVICE
TEST(DeviceGradient, CudaMatchesCpuWithResidentBinsAndRejectsStaleTokens) {
  for( const char* mode : {"cachedResponses", "recomputeResponses"} ) for( const char* layout : {"eventMajor", "warpInterleaved"} ){
    CudaBackend backend;
    backend.configure(ConfigReader(JsonType{{"propagationMode", mode}, {"weightLayout", layout}}));
    for( double rate : {1e-12, 1., 1e8} ){
      auto view = gradientModel(rate); backend.build(view);
      for( auto point : {std::vector<double>{.8, 1.1}, {0., .9}, {0., 0.}} ){
        compareGradient(backend, point);
        CpuBackend cpu; cpu.build(view);
        PropagationInputs inputs; inputs.parameters.values = point;
        auto reference = cpu.getLikelihoodGradient(cpu.requestPropagation(inputs));
        auto actual = backend.getLikelihoodGradient(backend.requestPropagation(inputs));
        for( std::size_t i = 0; i < actual.size(); ++i ) EXPECT_NEAR(actual[i], reference[i], 1e-14 + 3e-12*std::abs(reference[i]));
      }
      view.propagation.hasReweightCap = true; view.propagation.maxReweight = .5;
      backend.build(view); compareGradient(backend, {1.2, 1.3});
    }
  }
}
#endif

#ifdef __APPLE__
TEST(DeviceGradient, MetalGraphSplinesAndResponseClampsMatchCpu) {
  std::vector<DialUtils::DialPoint> points{{-2., .7, .15}, {-1., .9, .15}, {0., 1., .15}, {1., 1.25, .15}, {2., 1.4, .15}};
  CompactSpline compact; compact.buildDial(points);
  UniformSpline uniform; uniform.buildDial(points);
  MonotonicSpline monotonic; monotonic.buildDial(points);
  GeneralSpline general; general.buildDial(points);
  Graph graph; graph.buildDial(points);
  const std::vector<std::pair<BackendDialType, std::vector<double>>> cases{
      {BackendDialType::CompactSpline, compact.getDialData()}, {BackendDialType::UniformSpline, uniform.getDialData()},
      {BackendDialType::MonotonicSpline, monotonic.getDialData()}, {BackendDialType::GeneralSpline, general.getDialData()},
      {BackendDialType::Graph, graph.getDialData()}, {BackendDialType::Shift, {.83}}};
  MpsBackend gpu;
  if( !gpu.getCapabilities().supportsGpu ) GTEST_SKIP() << "Metal unavailable";
  for( const auto& item : cases ) for( bool extrapolate : {false, true} ) for( bool clamp : {false, true} ) for( bool mirror : {false, true} ){
    auto view = gradientModel(1.);
    auto& model = view.propagation;
    model.dials.resize(1); model.dialPayloads = item.second;
    model.dialInputs[0] = {0, mirror, -2., 4.};
    auto& dial = model.dials[0];
    dial.type = item.first; dial.payloadSize = item.second.size(); dial.allowExtrapolation = extrapolate;
    dial.hasMinResponse = dial.hasMaxResponse = clamp; dial.minResponse = .95; dial.maxResponse = 1.2;
    for( auto& index : model.eventDialIndices ) index = 0;
    gpu.build(view);
    CpuBackend cpu; cpu.build(view);
    for( double point : {.31, 2.7, -2.7} ){
      SCOPED_TRACE(::testing::Message() << unsigned(item.first) << ", extrapolate=" << extrapolate << ", clamp=" << clamp << ", point=" << point);
      PropagationInputs inputs; inputs.parameters.values = {point, 1.};
      auto expected = cpu.getLikelihoodGradient(cpu.requestPropagation(inputs));
      auto actual = gpu.getLikelihoodGradient(gpu.requestPropagation(inputs));
      for( std::size_t i = 0; i < actual.size(); ++i ) EXPECT_NEAR(actual[i], expected[i], 1e-13 + 3e-8*std::abs(expected[i]));
      compareGradient(gpu, inputs.parameters.values, 2e-7);
    }
  }
}
#endif
