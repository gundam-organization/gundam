#include "CpuBackend.h"
#ifdef __APPLE__
#include "MpsBackend.h"
#endif
#include "Semantics/BackendDialSemantics.h"
#include "Bilinear.h"
#include "Bicubic.h"

#include <TH2D.h>
#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <tuple>

namespace {
  using namespace Backends;

  TH2D makeSurface() {
    const double edges[]{-3., -1.9, -.7, .2, 1.8, 3., 5.};
    TH2D surface("backendSurface", "", 6, edges, 6, edges);
    surface.SetDirectory(nullptr);
    for( int x = 1; x <= 6; ++x ){
      for( int y = 1; y <= 6; ++y ){
        const auto a = surface.GetXaxis()->GetBinCenter(x);
        const auto b = surface.GetYaxis()->GetBinCenter(y);
        surface.SetBinContent(x, y, 1. + .1 * a + .2 * b + .03 * a * b + .04 * std::sin(a - b));
      }
    }
    return surface;
  }

  EngineView makeModel(const DialBase& surface_, bool cubic_, bool extrapolate_, bool mirror_, bool observable_) {
    EngineView view;
    auto& model = view.propagation;
    model.parameterCount = 3;
    model.totalBins = 1;
    model.dialInputs = {{0}, {1, mirror_, -1., 3.}, {2}};
    BackendDialDescriptor surface;
    surface.type = cubic_ ? BackendDialType::Bicubic : BackendDialType::Bilinear;
    surface.inputCount = 2;
    surface.allowExtrapolation = extrapolate_;
    surface.payloadSize = surface_.getDialData().size();
    surface.hasMinResponse = surface.hasMaxResponse = true;
    surface.minResponse = .2;
    surface.maxResponse = 2.;
    model.dialPayloads = surface_.getDialData();
    model.dials.push_back(surface);
    // Enough one-input dials to exercise shared CPU input groups and sparse
    // MPS updates. All events reuse the same surface response slot.
    for( int i = 0; i < 64; ++i ){
      BackendDialDescriptor norm;
      norm.type = BackendDialType::Norm;
      norm.firstInput = 2;
      norm.inputCount = 1;
      model.dials.push_back(norm);
      EventView event;
      event.resultIndex = i;
      event.globalBinIndex = event.binIndex = 0;
      event.samplePosition = 0;
      model.events.push_back(event);
      model.eventWeights.push_back({1. + .01 * i, model.eventDialIndices.size(), 2});
      model.eventDialIndices.push_back(0);
      model.eventDialIndices.push_back(i + 1);
    }
    SampleView sample;
    sample.binCount = 1;
    sample.binning.variableNames = {"X"};
    sample.binning.intervals = {{0, -10., 10., 0., 0.}};
    sample.binning.bins = {{0, 0, 1}};
    model.samples.push_back(sample);
    if( observable_ ){
      model.hasVariableResponses = true;
      model.variableValues = {.5};
      model.events[0].variableCount = model.events[0].variableResponseCount = 1;
      BackendVariableResponseDescriptor response;
      response.overrideDial = 0;
      model.variableResponses.push_back(response);
    }
    return view;
  }

  class BackendSurface : public testing::TestWithParam<std::tuple<bool, bool, bool>> {};
}

TEST_P(BackendSurface, MatchesNativeAndInvalidatesEveryInput) {
  const auto [gpu, cubic, cache] = GetParam();
  std::unique_ptr<Backend> backend;
  if( gpu ){
#ifdef __APPLE__
    backend = std::make_unique<MpsBackend>();
    ASSERT_NE(backend->getCapabilities().deviceName, "Metal unavailable");
    backend->configure(ConfigReader(JsonType{{"enableCachedDialResponses", cache}}));
#else
    GTEST_SKIP();
#endif
  }
  else{ backend = std::make_unique<CpuBackend>(); }
  auto histogram = makeSurface();
  Bilinear bilinear; bilinear.buildDial(histogram);
  Bicubic bicubic; bicubic.buildDial(histogram);
  DialBase& native = cubic ? static_cast<DialBase&>(bicubic) : static_cast<DialBase&>(bilinear);
  DialInputBuffer buffer;
  for( bool observable : {false, true} ){
    for( bool extrapolate : {false, true} ){
      native.setAllowExtrapolation(extrapolate);
      for( bool mirror : {false, true} ){
        auto view = makeModel(native, cubic, extrapolate, mirror, observable);
        backend->build(view);
        PropagationInputs inputs;
        // Includes y-only, x-only, simultaneous, repeated, out-of-domain and
        // unrelated-parameter changes. Rebuilds above also invalidate caches.
        for( const auto& point : {std::vector<double>{.1, .2, 1.}, {.1, .8, 1.},
                                 {.7, .8, 1.}, {1.1, -.6, 1.}, {1.1, -.6, 1.},
                                 {-4., 6., 1.}, {6., -4., 1.}, {.4, .5, 1.2}} ){
          inputs.parameters.values = point;
          auto token = backend->requestPropagation(inputs);
          ASSERT_TRUE(token.isValid);
          backend->materialize(token, OutputRequest::EventWeights);
          buffer.getInputBuffer() = {point[0], Semantics::transformDialInput(view.propagation.dialInputs[1], point[1])};
          const double response = std::max(.2, std::min(2., native.evalResponse(buffer)));
          const auto& weights = backend->getEventWeightsHostView(token);
          double sum = 0., squares = 0.;
          for( std::size_t i = 0; i < weights.size(); ++i ){
            const double expected = (1. + .01 * i) * response * point[2];
            EXPECT_NEAR(weights[i], expected, gpu ? 2e-6 : 1e-12);
            sum += expected;
            squares += expected * expected;
          }
          EXPECT_NEAR(backend->getHistogramSumsHostView(token)[0], sum, gpu ? 5e-5 : 1e-10);
          EXPECT_NEAR(backend->getHistogramSumSquaresHostView(token)[0], squares, gpu ? 1e-4 : 1e-10);
          if( observable ){
            EXPECT_NEAR(backend->getEventStateHostView(token).variables[0], response, gpu ? 2e-6 : 1e-12);
          }
#ifdef __APPLE__
          if( gpu and cache ){
            const auto state = JsonType::parse(static_cast<MpsBackend&>(*backend).getDiagnosticEventPropagationState());
            EXPECT_LE(state.at("evaluatedDials").get<std::size_t>(), view.propagation.dials.size());
          }
#endif
        }
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(CpuAndMetal, BackendSurface,
    testing::Combine(testing::Bool(), testing::Bool(), testing::Bool()));

TEST(BackendSurfaceInputs, RejectsUnsupportedEventSourcesAndMalformedSurfaces) {
  auto histogram = makeSurface();
  Bilinear native; native.buildDial(histogram);
  auto good = makeModel(native, false, false, false, false);
  std::vector<std::unique_ptr<Backend>> backends;
  backends.push_back(std::make_unique<CpuBackend>());
#ifdef __APPLE__
  backends.push_back(std::make_unique<MpsBackend>());
#endif
  for( auto& backend : backends ){
    auto bad = good;
    bad.propagation.dialInputs[1].source = DialInputSource::EventVariable;
    EXPECT_ANY_THROW(backend->build(bad));
    bad = good;
    bad.propagation.dials[0].inputCount = 1;
    EXPECT_ANY_THROW(backend->build(bad));
    bad = good;
    bad.propagation.dialPayloads[0] = 1.;
    EXPECT_ANY_THROW(backend->build(bad));
    bad = good;
    bad.propagation.dialPayloads.pop_back();
    EXPECT_ANY_THROW(backend->build(bad));
    bad = good;
    bad.propagation.dialPayloads[3] = bad.propagation.dialPayloads[2];
    EXPECT_ANY_THROW(backend->build(bad));
    EXPECT_NO_THROW(backend->build(good));
  }
}

TEST_P(BackendSurface, ReorderedAndRepeatedParametersUseDisjointCacheSlots) {
  const auto [gpu, cubic, cache] = GetParam();
  std::unique_ptr<Backend> backend;
  if( gpu ){
#ifdef __APPLE__
    backend = std::make_unique<MpsBackend>();
    ASSERT_NE(backend->getCapabilities().deviceName, "Metal unavailable");
    backend->configure(ConfigReader(JsonType{{"enableCachedDialResponses", cache}}));
#else
    GTEST_SKIP();
#endif
  }
  else{ backend = std::make_unique<CpuBackend>(); }
  auto histogram = makeSurface();
  Bilinear bilinear; bilinear.buildDial(histogram);
  Bicubic bicubic; bicubic.buildDial(histogram);
  DialBase& native = cubic ? static_cast<DialBase&>(bicubic) : static_cast<DialBase&>(bilinear);
  auto view = makeModel(native, cubic, false, false, false);
  auto& model = view.propagation;
  model.dialInputs.insert(model.dialInputs.end(), {{1}, {0}, {0}, {0}});
  auto surface = model.dials[0];
  const auto reversed = model.dials.size();
  surface.firstInput = 3;
  model.dials.push_back(surface); // Same payload, different ordered inputs.
  const auto repeated = model.dials.size();
  surface.firstInput = 5;
  model.dials.push_back(surface); // One parameter appears twice.
  for( std::size_t i = 0; i < model.events.size(); ++i ){
    model.eventDialIndices[2 * i] = i % 3 == 0 ? 0 : (i % 3 == 1 ? reversed : repeated);
  }
  backend->build(view);
  PropagationInputs inputs;
  DialInputBuffer buffer;
  const std::size_t expectedUpdates[]{model.dials.size(), 2, 3, 3, 0};
  std::size_t pointIndex{0};
  for( const auto& point : {std::vector<double>{.1, .2, 1.}, {.1, .8, 1.},
                           {.7, .8, 1.}, {1.1, -.6, 1.}, {1.1, -.6, 1.}} ){
    inputs.parameters.values = point;
    auto token = backend->requestPropagation(inputs);
    ASSERT_TRUE(token.isValid);
    backend->materialize(token, OutputRequest::EventWeights);
    const auto& weights = backend->getEventWeightsHostView(token);
    for( std::size_t i = 0; i < weights.size(); ++i ){
      buffer.getInputBuffer() = i % 3 == 0 ? std::vector<double>{point[0], point[1]}
          : i % 3 == 1 ? std::vector<double>{point[1], point[0]} : std::vector<double>{point[0], point[0]};
      const double expected = (1. + .01 * i) * std::max(.2, std::min(2., native.evalResponse(buffer)));
      EXPECT_NEAR(weights[i], expected, gpu ? 2e-6 : 1e-12);
    }
#ifdef __APPLE__
    if( gpu and cache ){
      const auto state = JsonType::parse(static_cast<MpsBackend&>(*backend).getDiagnosticEventPropagationState());
      EXPECT_EQ(state.at("evaluatedDials").get<std::size_t>(), expectedUpdates[pointIndex]);
    }
#endif
    if( not gpu ){
      EXPECT_EQ(backend->getLastTimingSummary().cacheCandidateCount, expectedUpdates[pointIndex]);
    }
    ++pointIndex;
  }
}

TEST(BackendSurfaceMath, ShortAndLargeAxesAreBounded) {
  for( int nx : {2, 3, 17} ){
    for( int ny : {2, 5, 19} ){
      std::vector<double> payload{double(nx), double(ny)};
      for( int x = 0; x < nx; ++x ){ payload.push_back(.25 * x); }
      for( int y = 0; y < ny; ++y ){ payload.push_back(.5 * y); }
      for( int x = 0; x < nx; ++x ){
        for( int y = 0; y < ny; ++y ){ payload.push_back(1. + .25 * x + y); }
      }
      for( bool cubic : {false, true} ){
        for( double x : {-1., 0., .1, .25 * (nx - 1), 10.} ){
          for( double y : {-1., 0., .3, .5 * (ny - 1), 10.} ){
            EXPECT_NEAR(GundamDeviceMath::EvaluateSurface(x, y, payload.data(), cubic, true),
                        1. + x + 2 * y, 1e-10);
          }
        }
      }
    }
  }
}
