#include "CpuBackend.h"
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

  void compareGradient(CpuBackend& backend_, const std::vector<double>& point_) {
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
      EXPECT_NEAR(gradient[i], numeric, 3e-5 * std::max(1e-20, std::abs(numeric)) + 1e-14);
    }
    EXPECT_THROW(backend_.getLikelihoodGradient(token), std::exception);
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
