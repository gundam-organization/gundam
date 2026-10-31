#include "CudaBackend.h"
#include "CpuBackend.h"
#include "Bilinear.h"
#include "Bicubic.h"
#include "Semantics/BackendDialSemantics.h"

#include <TH2D.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
  using namespace Backends;

  TH2D makeSurface() {
    const double x[]{-3., -1.9, -.7, .2, 1.8, 3., 5.};
    const double y[]{-2., -.8, .1, 1.4, 2.9, 5.};
    TH2D histogram("cudaSurface", "", 6, x, 5, y);
    histogram.SetDirectory(nullptr);
    for( int i = 1; i <= 6; ++i ){
      for( int j = 1; j <= 5; ++j ){
        const double a = histogram.GetXaxis()->GetBinCenter(i), b = histogram.GetYaxis()->GetBinCenter(j);
        histogram.SetBinContent(i, j, 1. + .1 * a + .2 * b + .03 * a * b + .04 * std::sin(a - b));
      }
    }
    return histogram;
  }

  EngineView makeModel(const DialBase& native_, bool cubic_, bool extrapolate_, bool mirror_, bool capped_) {
    EngineView view;
    auto& model = view.propagation;
    model.parameterCount = 4; // Last parameter is deliberately unused.
    model.totalBins = 5;
    model.hasReweightCap = capped_;
    model.maxReweight = 1.5;
    model.dialPayloads = native_.getDialData();
    // Interleave surfaces, constants and scalar dials to exercise remapping.
    for( int i = 0; i < 5; ++i ){
      BackendDialDescriptor dial;
      if( i == 0 or i == 3 or i == 4 ){
        dial.type = cubic_ ? BackendDialType::Bicubic : BackendDialType::Bilinear;
        dial.firstInput = model.dialInputs.size();
        dial.inputCount = 2;
        dial.payloadSize = native_.getDialData().size();
        dial.allowExtrapolation = extrapolate_;
        dial.hasMinResponse = dial.hasMaxResponse = true;
        dial.minResponse = .2;
        dial.maxResponse = 1.8;
        model.dialInputs.push_back({i == 3 ? 1u : 0u, mirror_, -1.5, 4.});
        model.dialInputs.push_back({i == 4 ? 0u : (i == 3 ? 0u : 1u), mirror_, -1., 3.});
      }
      else if( i == 1 ){
        dial.type = BackendDialType::Shift;
        dial.payloadOffset = model.dialPayloads.size();
        dial.payloadSize = 1;
        model.dialPayloads.push_back(.9);
      }
      else{
        dial.firstInput = model.dialInputs.size();
        dial.inputCount = 1;
        model.dialInputs.push_back({2});
      }
      model.dials.push_back(dial);
    }
    for( int i = 0; i < 64; ++i ){ model.dials.push_back(model.dials[2]); }
    model.events.resize(521);
    model.eventWeights.resize(521);
    for( std::size_t i = 0; i < model.events.size(); ++i ){
      const auto slot = (37 * i) % model.events.size();
      model.events[i].resultIndex = slot;
      model.events[i].globalBinIndex = i % 11 == 0 ? -1 : int(i % 4);
      auto& event = model.eventWeights[slot];
      event.baseWeight = i % 17 == 0 ? 0. : (i % 13 == 0 ? -.125 : .25);
      event.firstDial = model.eventDialIndices.size();
      for( auto index : {0u, 2u, 3u, 0u, 4u, 1u, unsigned(5 + i % 64)} ){
        model.eventDialIndices.push_back(index);
      }
      event.dialCount = 7;
    }
    LikelihoodSampleView sample;
    sample.dataSums = {2., 1., 0., 3., 0.};
    sample.ignoredBins = {false, false, true, false, false};
    sample.evalBin = [](double data_, double prediction_, double error_, int){
      const double delta = prediction_ - data_;
      return delta * delta / (1. + error_ * error_);
    };
    view.likelihood.samples.push_back(sample);
    return view;
  }

  void expectClose(const std::vector<double>& actual_, const std::vector<double>& expected_) {
    ASSERT_EQ(actual_.size(), expected_.size());
    for( std::size_t i = 0; i < actual_.size(); ++i ){
      EXPECT_NEAR(actual_[i], expected_[i], 3.e-12 * std::max(1., std::abs(expected_[i]))) << "slot " << i;
    }
  }

  std::vector<double> nativeWeights(const EngineView& view_, DialBase& native_, const std::vector<double>& values_) {
    const auto& model = view_.propagation;
    std::vector<double> responses;
    DialInputBuffer buffer;
    for( const auto& dial : model.dials ){
      double response;
      if( dial.inputCount == 2 ){
        buffer.getInputBuffer().clear();
        for( std::size_t j = 0; j < 2; ++j ){
          const auto& input = model.dialInputs[dial.firstInput + j];
          buffer.getInputBuffer().push_back(Semantics::transformDialInput(input, values_[input.parameterIndex]));
        }
        response = std::max(.2, std::min(1.8, native_.evalResponse(buffer)));
      }
      else{ response = dial.type == BackendDialType::Shift ? .9 : values_[2]; }
      responses.push_back(response);
    }
    std::vector<double> result;
    for( const auto& event : model.eventWeights ){
      double response = 1.;
      for( std::size_t i = 0; i < event.dialCount; ++i ){
        response *= responses[model.eventDialIndices[event.firstDial + i]];
      }
      if( model.hasReweightCap ){ response = std::min(response, model.maxReweight); }
      result.push_back(event.baseWeight * response);
    }
    return result;
  }
}

TEST(CudaBackendSurface, MatchesNativeAndCpuForBothModesAndLayouts) {
  auto histogram = makeSurface();
  Bilinear bilinear; bilinear.buildDial(histogram);
  Bicubic bicubic; bicubic.buildDial(histogram);
  for( bool cubic : {false, true} ){
    auto& native = cubic ? static_cast<DialBase&>(bicubic) : static_cast<DialBase&>(bilinear);
    for( const auto* mode : {"cachedResponses", "recomputeResponses"} ){
      for( const auto* layout : {"eventMajor", "warpInterleaved"} ){
        for( bool extrapolate : {false, true} ){
          native.setAllowExtrapolation(extrapolate);
          for( bool mirror : {false, true} ){
            for( bool capped : {false, true} ){
              SCOPED_TRACE(::testing::Message() << "cubic=" << cubic << " " << mode << " " << layout
                  << " extrapolate=" << extrapolate << " mirror=" << mirror << " capped=" << capped);
              const auto view = makeModel(native, cubic, extrapolate, mirror, capped);
              CudaBackend cuda;
              CpuBackend cpu;
              cuda.configure(ConfigReader(JsonType{{"propagationMode", mode}, {"weightLayout", layout}}));
              cuda.build(view);
              cpu.build(view);
              EXPECT_EQ(JsonType::parse(cuda.getDiagnostics())["weightLayout"], layout);
              const std::vector<std::vector<double>> points{
                  {.1, .2, 1., 0.}, {.1, .2, 1., 0.}, {.1, .8, 1., 0.}, {.7, .8, 1., 0.},
                  {1.1, -.6, 1., 0.}, {1.1, -.6, 1., 0.}, {1.1, -.6, 1., 2.},
                  {-4., 6., 1., 2.}, {6., -4., 1., 2.}, {.4, .5, 1.2, 2.}, {.4, .7, 1.2, 2.},
                  {-2.45, 3.95, 1.2, 2.}};
              const std::vector<std::size_t> candidates{69, 0, 2, 3, 3, 0, 0, 3, 3, 69, 2, 3};
              PropagationToken previous;
              for( std::size_t i = 0; i < points.size(); ++i ){
                SCOPED_TRACE(i);
                PropagationInputs inputs;
                inputs.parameters.values = points[i];
                const auto token = cuda.requestPropagation(inputs);
                const auto reference = cpu.requestPropagation(inputs);
                EXPECT_FALSE(cuda.isReady(previous));
                EXPECT_EQ(cuda.getLastTimingSummary().cacheCandidateCount,
                          std::string(mode) == "cachedResponses" ? candidates[i] : 69);
                expectClose(cuda.getEventWeightsHostView(token), nativeWeights(view, native, points[i]));
                expectClose(cuda.getEventWeightsHostView(token), cpu.getEventWeightsHostView(reference));
                expectClose(cuda.getHistogramSumsHostView(token), cpu.getHistogramSumsHostView(reference));
                expectClose(cuda.getHistogramSumSquaresHostView(token), cpu.getHistogramSumSquaresHostView(reference));
                EXPECT_NEAR(cuda.getLikelihood(token), cpu.getLikelihood(reference),
                            3.e-11 * std::max(1., std::abs(cpu.getLikelihood(reference))));
                previous = token;
              }
            }
          }
        }
      }
    }
  }
}

TEST(CudaBackendSurface, ShortAndLargeAxesMatchAnalyticSurface) {
  for( auto shape : {std::pair<int, int>{2, 2}, {3, 5}, {17, 19}} ){
    for( bool cubic : {false, true} ){
      for( const auto* mode : {"cachedResponses", "recomputeResponses"} ){
        EngineView view;
        auto& model = view.propagation;
        model.parameterCount = 2;
        model.totalBins = 1;
        model.dialInputs = {{0}, {1}};
        model.dials.resize(1);
        auto& dial = model.dials[0];
        dial.type = cubic ? BackendDialType::Bicubic : BackendDialType::Bilinear;
        dial.inputCount = 2;
        dial.allowExtrapolation = true;
        model.dialPayloads = {double(shape.first), double(shape.second)};
        for( int i = 0; i < shape.first; ++i ){ model.dialPayloads.push_back(.25 * i); }
        for( int i = 0; i < shape.second; ++i ){ model.dialPayloads.push_back(.5 * i); }
        for( int i = 0; i < shape.first; ++i ){
          for( int j = 0; j < shape.second; ++j ){ model.dialPayloads.push_back(1. + .25 * i + j); }
        }
        dial.payloadSize = model.dialPayloads.size();
        model.eventWeights = {{1., 0, 1}};
        model.eventDialIndices = {0};
        model.events.resize(1);
        model.events[0].globalBinIndex = 0;
        CudaBackend cuda;
        cuda.configure(ConfigReader(JsonType{{"propagationMode", mode}}));
        cuda.build(view);
        for( const auto& point : std::vector<std::vector<double>>{
            {0., 0.}, {.1, .3}, {.1, 10.}, {-1., 10.}, {10., -1.}, {.25 * (shape.first - 1), .5 * (shape.second - 1)}} ){
          PropagationInputs inputs;
          inputs.parameters.values = point;
          const auto token = cuda.requestPropagation(inputs);
          EXPECT_NEAR(cuda.getEventWeightsHostView(token)[0], 1. + point[0] + 2 * point[1], 1.e-10);
        }
      }
    }
  }
}

TEST(CudaBackendSurface, ErrorsAndRebuildInvalidateBothCachedInputs) {
  auto histogram = makeSurface();
  Bilinear native; native.buildDial(histogram);
  for( const auto* mode : {"cachedResponses", "recomputeResponses"} ){
    auto view = makeModel(native, false, true, false, false);
    // Finite raw parameters can overflow the arithmetic of the second mirror.
    auto& y = view.propagation.dialInputs[1];
    y.useMirror = true;
    y.mirrorMin = -std::numeric_limits<double>::max();
    y.mirrorRange = std::numeric_limits<double>::max();
    auto fail = std::make_shared<bool>(false);
    const auto evaluate = view.likelihood.samples[0].evalBin;
    view.likelihood.samples[0].evalBin = [fail, evaluate](double a_, double b_, double c_, int d_){
      return *fail ? std::numeric_limits<double>::quiet_NaN() : evaluate(a_, b_, c_, d_);
    };
    CudaBackend cuda;
    cuda.configure(ConfigReader(JsonType{{"propagationMode", mode}}));
    cuda.build(view);
    PropagationInputs inputs;
    inputs.parameters.values = {.1, 0., 1., 0.};
    auto token = cuda.requestPropagation(inputs);
    const auto expected = cuda.getEventWeightsHostView(token);
    inputs.parameters.values[1] = std::numeric_limits<double>::max();
    EXPECT_THROW((void)cuda.requestPropagation(inputs), std::runtime_error);
    EXPECT_FALSE(cuda.isReady(token));
    inputs.parameters.values[1] = 0.;
    token = cuda.requestPropagation(inputs);
    EXPECT_EQ(cuda.getLastTimingSummary().cacheCandidateCount, view.propagation.dials.size());
    expectClose(cuda.getEventWeightsHostView(token), expected);
    *fail = true;
    inputs.parameters.values[0] = .7;
    EXPECT_THROW((void)cuda.requestPropagation(inputs), std::runtime_error);
    *fail = false;
    token = cuda.requestPropagation(inputs);
    EXPECT_EQ(cuda.getLastTimingSummary().cacheCandidateCount, view.propagation.dials.size());
    CpuBackend cpu; cpu.build(view);
    expectClose(cuda.getEventWeightsHostView(token), cpu.getEventWeightsHostView(cpu.requestPropagation(inputs)));
    const auto stale = token;
    cuda.build(view);
    token = cuda.requestPropagation(inputs);
    EXPECT_FALSE(cuda.isReady(stale));
    EXPECT_EQ(cuda.getLastTimingSummary().cacheCandidateCount, view.propagation.dials.size());
  }
}
