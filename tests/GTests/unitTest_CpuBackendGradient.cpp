#include "CpuBackend.h"
#ifdef __APPLE__
#include "MpsBackend.h"
#endif
#ifdef GUNDAM_TEST_CUDA_DEVICE
#include "CudaBackend.h"
#endif
#include "GradientModel.h"
#include "FormulaCompiler.h"
#include "DialResponseSupervisor.h"
#include "Semantics/BackendDeviceGradient.h"
#include "Semantics/BackendDialGradient.h"
#include "Bilinear.h"
#include "Bicubic.h"
#include "Graph.h"
#include "CompactSpline.h"
#include "UniformSpline.h"
#include "MonotonicSpline.h"
#include "GeneralSpline.h"

#include <TH2D.h>
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

  void smoothLimits(BackendDialDescriptor& dial_, double minWidth_ = .1, double maxWidth_ = .15) {
    auto& smoothing = dial_.smoothing;
    smoothing.minResponse = dial_.minResponse; smoothing.maxResponse = dial_.maxResponse;
    smoothing.minWidth = dial_.hasMinResponse ? minWidth_ : 0.;
    smoothing.maxWidth = dial_.hasMaxResponse ? maxWidth_ : 0.;
    smoothing.minInverseWidth = smoothing.minWidth > 0. ? .5/smoothing.minWidth : 0.;
    smoothing.maxInverseWidth = smoothing.maxWidth > 0. ? .5/smoothing.maxWidth : 0.;
    dial_.minResponse += smoothing.minWidth; dial_.maxResponse -= smoothing.maxWidth;
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
          for( bool clamp : {false, true} ) for( bool smooth : {false, true} ){
            dial.hasMinResponse = dial.hasMaxResponse = clamp;
            dial.minResponse = .95; dial.maxResponse = 1.2;
            dial.smoothing = {};
            if( smooth ){ smoothLimits(dial); }
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

TEST(CpuBackendGradient, RejectsUnsupportedLikelihood) {
  CpuBackend backend;
  auto view = gradientModel(1.);
  view.likelihood.samples[0].portableOptions.kind = LikelihoodKind::Poisson;
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
  for( const auto& item : cases ) for( bool extrapolate : {false, true} ) for( bool clamp : {false, true} ) for( bool mirror : {false, true} ) for( bool smooth : {false, true} ){
    auto view = gradientModel(1.);
    auto& model = view.propagation;
    model.dials.resize(1); model.dialPayloads = item.second;
    model.dialInputs[0] = {0, mirror, -2., 4.};
    auto& dial = model.dials[0];
    dial.type = item.first; dial.payloadSize = item.second.size(); dial.allowExtrapolation = extrapolate;
    dial.hasMinResponse = dial.hasMaxResponse = clamp; dial.minResponse = .95; dial.maxResponse = 1.2;
    if( smooth ) smoothLimits(dial);
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

namespace {
  void checkSurfaceInputsMatchFiniteDifferences(Backend& backend_) {
    const double edges[]{-3., -1.9, -.7, .2, 1.8, 3., 5.};
    TH2D histogram("gradientSurface", "", 6, edges, 6, edges);
    histogram.SetDirectory(nullptr);
    for( int x = 1; x <= 6; ++x ) for( int y = 1; y <= 6; ++y ){
      const double a = histogram.GetXaxis()->GetBinCenter(x), b = histogram.GetYaxis()->GetBinCenter(y);
      histogram.SetBinContent(x, y, 1. + .1*a + .2*b + .03*a*b + .04*std::sin(a-b));
    }
    Bilinear bilinear; bilinear.buildDial(histogram);
    Bicubic bicubic; bicubic.buildDial(histogram);
    for( bool cubic : {false, true} ) for( bool mirror : {false, true} )
    for( bool extrapolate : {false, true} ) for( bool shared : {false, true} )
    for( bool clamp : {false, true} ) for( bool smooth : {false, true} ){
      SCOPED_TRACE(::testing::Message() << cubic << mirror << extrapolate << shared << clamp);
      auto view = gradientModel(1.);
      auto& model = view.propagation;
      model.dials.resize(1);
      auto& dial = model.dials[0];
      dial.type = cubic ? BackendDialType::Bicubic : BackendDialType::Bilinear;
      dial.inputCount = 2;
      dial.allowExtrapolation = extrapolate;
      model.dialPayloads = cubic ? bicubic.getDialData() : bilinear.getDialData();
      dial.payloadSize = model.dialPayloads.size();
      dial.hasMinResponse = dial.hasMaxResponse = clamp;
      dial.minResponse = .95; dial.maxResponse = 1.2;
      if( smooth ){ smoothLimits(dial); }
      model.dialInputs = {{0, mirror, -2., 4.}, {shared ? 0u : 1u, mirror, -2., 4.}};
      for( auto& index : model.eventDialIndices ){ index = 0; }
      backend_.build(view);
      for( const auto& point : {std::vector<double>{.31, -.27}, {-3.1, .31}, {.31, 5.37}, {6.3, -6.7}} ){
        compareGradient(backend_, point, 2e-7);
      }
    }
  }
}
TEST(CpuBackendGradient, SurfaceInputsMatchFiniteDifferences) {
  CpuBackend backend;
  checkSurfaceInputsMatchFiniteDifferences(backend);
}
#ifdef __APPLE__
TEST(MpsBackendGradient, SurfaceInputsMatchFiniteDifferences) {
  MpsBackend backend;
  if( !backend.getCapabilities().supportsGpu ) GTEST_SKIP() << "Metal unavailable";
  checkSurfaceInputsMatchFiniteDifferences(backend);
}
#endif

namespace {
  void checkCompositeBranchesAndFormulaNodes(Backend& backend_) {
    for( bool mirror : {false, true} ) for( bool clamp : {false, true} ) for( bool smooth : {false, true} ){
      auto view = gradientModel(1.);
      auto& model = view.propagation;
      model.dials.resize(1);
      auto& root = model.dials[0];
      root.type = BackendDialType::Composite; root.inputCount = 2;
      root.hasMaxResponse = clamp; root.maxResponse = 1.4;
      if( smooth ){ smoothLimits(root); }
      model.dialInputs = {{0, mirror, -2., 4.}, {1}};
      // source = p^2 + q; output = source*p + source. Repeated graph
      // edges and a direct root input must all contribute to the adjoint.
      model.formulaPrograms = {FormulaCompiler::compile("[p]^2+[q]", {"p", "q"}),
                              FormulaCompiler::compile("[a]*[p]+[b]", {"a", "p", "b"})};
      model.formulaInputs = {{0}, {1}, {0}, {1}, {2}};
      model.compositeInputs = {{0, mirror, -2., 4.}, {1}, {2}, {0}, {2}};
      CompositeInstruction first, second;
      first.dial.type = second.dial.type = BackendDialType::Formula;
      first.dial.inputCount = 2;
      first.dial.hasMinResponse = clamp; first.dial.minResponse = .5;
      if( smooth ){ smoothLimits(first.dial); }
      second.dial.firstInput = 2; second.dial.inputCount = 3;
      second.dial.firstFormulaInput = 2; second.dial.formulaProgramIndex = 1;
      model.compositeInstructions = {first, second};
      model.compositePrograms = {{0, 2, 2, 3}};
      for( auto& index : model.eventDialIndices ){ index = 0; }
      backend_.build(view);
      for( const auto& point : {std::vector<double>{.31, .27}, {-.7, .31}, {5.37, .8}} ){
        compareGradient(backend_, point, 2e-7);
      }
    }
  }
}
TEST(CpuBackendGradient, CompositeBranchesAndFormulaNodes) {
  CpuBackend backend;
  checkCompositeBranchesAndFormulaNodes(backend);
}
#ifdef __APPLE__
TEST(MpsBackendGradient, CompositeBranchesAndFormulaNodes) {
  MpsBackend backend;
  if( !backend.getCapabilities().supportsGpu ) GTEST_SKIP() << "Metal unavailable";
  checkCompositeBranchesAndFormulaNodes(backend);
}
#endif

namespace {
  void checkExternalAndTableSnapshotsMatchFiniteDifferences(Backend& backend_) {
    for( auto type : {BackendDialType::ExternalWeight, BackendDialType::Tabulated, BackendDialType::Kriged} )
    for( bool mirror : {false, true} ) for( bool clamp : {false, true} ) for( bool smooth : {false, true} ){
      auto view = gradientModel(1.);
      auto& model = view.propagation;
      model.dials.resize(1);
      auto& dial = model.dials[0];
      dial.type = type; dial.inputCount = 2;
      dial.hasMinResponse = dial.hasMaxResponse = clamp; dial.minResponse = .95; dial.maxResponse = 1.3;
      if( smooth ){ smoothLimits(dial); }
      model.dialInputs = {{0, mirror, -2., 4.}, {1}};
      model.externalWeightBlocks = {{0, 2}};
      model.tableTerms = {{0, .25}, {1, .75}}; dial.tableTermCount = 2;
      for( auto& index : model.eventDialIndices ){ index = 0; }
      backend_.build(view);
      PropagationInputs inputs;
      std::uint64_t generation = 0;
    std::vector<double> table(2);
      auto publish = [&](const std::vector<double>& point_) {
        inputs.parameters.values = point_;
        const double p = Semantics::transformDialInput(model.dialInputs[0], point_[0]), q = point_[1];
        table = {1.+.2*p*p+.1*q, 1.+.1*p+.3*q*q};
        inputs.externalWeights = {{table.data(), table.size(), 0, ++generation}};
        inputs.externalDialGradients = {type == BackendDialType::ExternalWeight ? std::vector<double>{.4*p, .1}
            : std::vector<double>{.1*p+.075, .025+.45*q}};
        return backend_.requestPropagation(inputs);
      };
      for( const auto& point : {std::vector<double>{.31, .27}, {5.37, .8}} ){
        const auto token = publish(point);
        // Jacobians and values are owned by the backend_ after publication.
        inputs.externalDialGradients[0].assign(2, 123.);
        table.assign(2, 456.);
        const auto analytic = backend_.getLikelihoodGradient(token);
        for( std::size_t i = 0; i < point.size(); ++i ){
          auto up = point, down = point; up[i] += 1e-5; down[i] -= 1e-5;
          const double upper = backend_.getLikelihood(publish(up));
          const double lower = backend_.getLikelihood(publish(down));
          EXPECT_NEAR(analytic[i], (upper-lower)/2e-5, 2e-6);
        }
      }
      publish({.31, .27});
      inputs.externalDialGradients.clear();
      auto token = backend_.requestPropagation(inputs);
      EXPECT_THROW(static_cast<void>(backend_.getLikelihoodGradient(token)), std::exception);
      inputs.externalDialGradients = {{1.}};
      token = backend_.requestPropagation(inputs);
      EXPECT_THROW(static_cast<void>(backend_.getLikelihoodGradient(token)), std::exception);
      inputs.externalDialGradients = {{NAN, 1.}};
      token = backend_.requestPropagation(inputs);
      EXPECT_THROW(static_cast<void>(backend_.getLikelihoodGradient(token)), std::exception);
    }
  }
}
TEST(CpuBackendGradient, ExternalAndTableSnapshotsMatchFiniteDifferences) {
  CpuBackend backend;
  checkExternalAndTableSnapshotsMatchFiniteDifferences(backend);
}
#ifdef __APPLE__
TEST(MpsBackendGradient, ExternalAndTableSnapshotsMatchFiniteDifferences) {
  MpsBackend backend;
  if( !backend.getCapabilities().supportsGpu ) GTEST_SKIP() << "Metal unavailable";
  checkExternalAndTableSnapshotsMatchFiniteDifferences(backend);
}
#endif

namespace {
  void checkCompositeFormulaOperationsMatchFiniteDifferences(Backend& backend_) {
    for( const auto* expression : {
        "exp([p])+log([q])+log10([q])+sqrt([q])+abs([p])",
        "sin([p])+cos([p])+tan([p])+asin([p])+acos([p])+atan([p])",
        "sinh([p])+cosh([p])+tanh([p])+erf([p])+erfc([p])",
        "pow([p],[q])+atan2([p],[q])+[p]/[q]-[p]*[q]",
        "min([p],[q])+max([p],[q])+floor([p])+ceil([q])",
        "(-[p])^2+0*[p]+2"} ){
      SCOPED_TRACE(expression);
      auto view = gradientModel(1.);
      auto& model = view.propagation;
      model.dials.resize(1);
      model.dials[0].type = BackendDialType::Composite;
      model.dials[0].inputCount = 2;
      model.formulaPrograms = {FormulaCompiler::compile(expression, {"p", "q"})};
      model.formulaInputs = {{0}, {1}};
      model.compositeInputs = {{0}, {1}};
      CompositeInstruction instruction;
      instruction.dial.type = BackendDialType::Formula; instruction.dial.inputCount = 2;
      model.compositeInstructions = {instruction};
      model.compositePrograms = {{0, 1, 2, 2}};
      for( auto& index : model.eventDialIndices ){ index = 0; }
      backend_.build(view);
      compareGradient(backend_, {.31, .73}, 2e-7);
    }
  }
}
TEST(CpuBackendGradient, CompositeFormulaOperationsMatchFiniteDifferences) {
  CpuBackend backend;
  checkCompositeFormulaOperationsMatchFiniteDifferences(backend);
}
#ifdef __APPLE__
TEST(MpsBackendGradient, CompositeFormulaOperationsMatchFiniteDifferences) {
  MpsBackend backend;
  if( !backend.getCapabilities().supportsGpu ) GTEST_SKIP() << "Metal unavailable";
  checkCompositeFormulaOperationsMatchFiniteDifferences(backend);
}
#endif

namespace {
  void checkStandaloneFormulaBindingsMirrorsAndResponseLimits(Backend& backend_) {
    for( bool mirror : {false, true} ) for( bool clamp : {false, true} ) for( bool smooth : {false, true} ){
      auto view = gradientModel(1.);
      auto& model = view.propagation;
      model.dials.resize(1);
      auto& dial = model.dials[0];
      dial.type = BackendDialType::Formula;
      dial.inputCount = 3;
      dial.hasMinResponse = dial.hasMaxResponse = clamp;
      dial.minResponse = 1.05; dial.maxResponse = 1.15;
      if( smooth ){ smoothLimits(dial, .02, .03); }
      model.dialInputs = {{1, mirror, -2., 4.}, {0}, {1}};
      model.formulaPrograms = {FormulaCompiler::compile("1+.1*[a]+.02*[b]^2+.03*[repeated]+.001*[event]",
                                                       {"event", "b", "a", "repeated"})};
      model.formulaInputs = {{-1, 17.}, {1}, {0}, {2}};
      for( auto& index : model.eventDialIndices ){ index = 0; }
      backend_.build(view);
      for( const auto& point : {std::vector<double>{.31, .73}, {-.7, 5.37}, {.2, -5.71}} ){
        compareGradient(backend_, point, 2e-7);
      }
      // Event-only formulas and constants have no parameter derivatives.
      dial.inputCount = 0;
      model.formulaPrograms = {FormulaCompiler::compile("1+.01*[event]", {"event"})};
      model.formulaInputs = {{-1, 7.}};
      backend_.build(view);
      compareGradient(backend_, {.31, .73}, 2e-7);
    }
  }
}
TEST(CpuBackendGradient, StandaloneFormulaBindingsMirrorsAndResponseLimits) {
  CpuBackend backend;
  checkStandaloneFormulaBindingsMirrorsAndResponseLimits(backend);
}
#ifdef __APPLE__
TEST(MpsBackendGradient, StandaloneFormulaBindingsMirrorsAndResponseLimits) {
  MpsBackend backend;
  if( !backend.getCapabilities().supportsGpu ) GTEST_SKIP() << "Metal unavailable";
  checkStandaloneFormulaBindingsMirrorsAndResponseLimits(backend);
}
#endif

TEST(CpuBackendGradient, ResponseSmoothingMatchesSupervisorAtTransitionBoundaries) {
  for( double minWidth : {0., .1} ) for( double maxWidth : {0., .2} ){
    DialResponseSupervisor supervisor;
    if( minWidth > 0. ){ supervisor.setMinResponse(.5); }
    if( maxWidth > 0. ){ supervisor.setMaxResponse(1.5); }
    if( minWidth > 0. or maxWidth > 0. ){ supervisor.setSmoothingWidths(minWidth, maxWidth); }
    supervisor.setMinResponse(.5); supervisor.setMaxResponse(1.5);
    BackendDialDescriptor dial;
    dial.inputCount = 1;
    dial.hasMinResponse = dial.hasMaxResponse = true;
    dial.minResponse = .5; dial.maxResponse = 1.5;
    smoothLimits(dial, minWidth, maxWidth);
    BackendDialInputDescriptor input{0};
    for( double x : {.3, .4, .45, .5, .55, .6, .8, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8} ){
      SCOPED_TRACE(::testing::Message() << minWidth << ", " << maxWidth << ", " << x);
      const double actual = Semantics::evalWeightDialGradient(dial, input, nullptr, x);
      EXPECT_NEAR(actual, supervisor.evalDerivative(x), 1e-14);
      const double h = 1e-6;
      if( (minWidth == 0. and x == .5) or (maxWidth == 0. and x == 1.5) ){ continue; }
      const double up = Semantics::clampDialResponse(dial, x+h), down = Semantics::clampDialResponse(dial, x-h);
      EXPECT_NEAR(actual, (up-down)/(2*h), 1e-8);
    }
  }
}

namespace {
  EngineView observableGradientModel() {
    EngineView view;
    auto& model = view.propagation;
    model.parameterCount = 3;
    model.totalBins = 4;
    model.hasVariableResponses = true;
    model.dialInputs = {{0}, {1}, {2}};
    for( std::size_t i = 0; i < 3; ++i ){
      BackendDialDescriptor dial;
      dial.firstInput = i; dial.inputCount = 1;
      model.dials.push_back(dial);
    }
    SampleView sample;
    sample.sampleIndex = 0; sample.binCount = 4;
    sample.useDynamicBinning = true;
    sample.variableNames = sample.binning.variableNames = {"x", "y", "selection"};
    sample.binning.isSmoothed = true;
    for( int x = 0; x < 2; ++x ) for( int y = 0; y < 2; ++y ){
      sample.binning.bins.push_back({2*x+y, sample.binning.intervals.size(), 3});
      sample.binning.intervals.push_back({0, double(x), double(x+1), 1./.3, 1./.3});
      sample.binning.intervals.push_back({1, double(y), double(y+1), 1./.4, 1./.4});
      sample.binning.intervals.push_back({2, 0., 1., 0., 0.});
    }
    model.samples.push_back(sample);
    for( int i = 0; i < 10; ++i ){
      EventView event;
      event.resultIndex = model.events.size();
      event.samplePosition = 0; event.sampleIndex = 0;
      // The initial strict bin is deliberately unavailable, including for
      // events that later contribute fractions from outside the hard support.
      event.binIndex = event.globalBinIndex = -1;
      event.firstVariable = model.variableValues.size(); event.variableCount = 3;
      model.variableValues.insert(model.variableValues.end(), {.5, .5, i == 5 ? 2. : .5});
      event.firstVariableResponse = model.variableResponses.size();
      if( i < 6 ){
        BackendVariableResponseDescriptor x;
        x.variableIndex = event.firstVariable;
        x.overrideDial = 0;
        x.firstMultiplyDial = model.variableDialIndices.size(); x.multiplyDialCount = 2;
        model.variableDialIndices.insert(model.variableDialIndices.end(), {1, 1});
        x.firstAddDial = model.variableDialIndices.size(); x.addDialCount = 1;
        model.variableDialIndices.push_back(2);
        model.variableResponses.push_back(x);
        BackendVariableResponseDescriptor y;
        y.variableIndex = event.firstVariable+1; y.referenceValue = .5+.1*i;
        y.firstMultiplyDial = model.variableDialIndices.size(); y.multiplyDialCount = 1;
        model.variableDialIndices.push_back(1);
        y.firstAddDial = model.variableDialIndices.size(); y.addDialCount = 1;
        model.variableDialIndices.push_back(0);
        model.variableResponses.push_back(y);
        event.variableResponseCount = 2;
        model.eventWeights.push_back({.2+.1*i, model.eventDialIndices.size(), 3});
        model.eventDialIndices.insert(model.eventDialIndices.end(), {0, 1, 0});
      }
      else {
        // Parameter-independent support keeps every populated data bin finite.
        const int bin = i-6;
        model.variableValues[event.firstVariable] = (bin/2)+.5;
        model.variableValues[event.firstVariable+1] = (bin%2)+.5;
        model.variableValues[event.firstVariable+2] = .5;
        model.eventWeights.push_back({2., 0, 0});
      }
      model.events.push_back(event);
    }
    LikelihoodSampleView likelihood;
    likelihood.portableOptions.kind = LikelihoodKind::BarlowBeestonLiteGaussian;
    likelihood.dataSums = {3., 4., 3.5, 5.};
    likelihood.ignoredBins = {false, true, false, false};
    view.likelihood.samples.push_back(likelihood);
    return view;
  }
}

TEST(CpuBackendGradient, MultidimensionalMigrationFractionsAndOutsideSupport) {
  for( bool mirror : {false, true} ) for( bool cap : {false, true} ){
    auto view = observableGradientModel();
    auto& model = view.propagation;
    model.dialInputs[0] = {0, mirror, 0., 2.};
    model.hasReweightCap = cap; model.maxReweight = .1;
    CpuBackend backend;
    backend.build(view);
    for( const auto& point : {std::vector<double>{.8, .9, .2}, {.8, 0., .2},
                            {.8, .9, -.8}, {.8, .9, 1.5}, {3.2, .9, .2}} ){
      compareGradient(backend, point, 2e-7);
    }
    // Host-only fractional bins remain differentiable for event weights.
    model.hasVariableResponses = false;
    model.variableResponses.clear();
    model.samples[0].useDynamicBinning = false;
    for( auto& event : model.events ){ event.variableResponseCount = 0; }
    backend.build(view);
    compareGradient(backend, {.8, .9, .2});
  }
}

TEST(CpuBackendGradient, RejectsMigrationWithoutSmoothingOnEachMovingAxis) {
  auto view = observableGradientModel();
  CpuBackend backend;
  view.propagation.samples[0].useDynamicBinning = false;
  backend.build(view);
  EXPECT_THROW(backend.validateLikelihoodGradient(), std::exception);
  view.propagation.samples[0].useDynamicBinning = true;
  view.propagation.samples[0].binning.isSmoothed = false;
  backend.build(view);
  EXPECT_THROW(backend.validateLikelihoodGradient(), std::exception);
  view.propagation.samples[0].binning.isSmoothed = true;
  view.propagation.samples[0].binning.intervals[1].lowerInverseWidth = 0.;
  view.propagation.samples[0].binning.intervals[1].upperInverseWidth = 0.;
  backend.build(view);
  EXPECT_THROW(backend.validateLikelihoodGradient(), std::exception);
}

TEST(CpuBackendGradient, ObservableFormulaCompositeAndProducerJacobians) {
  for( auto type : {BackendDialType::Formula, BackendDialType::Composite,
                   BackendDialType::ExternalWeight, BackendDialType::Tabulated, BackendDialType::Kriged} ){
    auto view = observableGradientModel();
    auto& model = view.propagation;
    auto& dial = model.dials[0];
    dial.type = type; dial.inputCount = 2;
    dial.hasMinResponse = dial.hasMaxResponse = true;
    dial.minResponse = .7; dial.maxResponse = 1.;
    smoothLimits(dial, .1, .1);
    const bool producer = type == BackendDialType::ExternalWeight or isTableDial(type);
    if( producer ){
      model.externalWeightBlocks = {{0, 2}};
      model.tableTerms = {{0, .25}, {1, .75}};
      dial.tableTermCount = 2;
    }
    else {
      model.formulaPrograms = {FormulaCompiler::compile("[p]^2+.2*[q]", {"p", "q"})};
      model.formulaInputs = {{0}, {1}};
      if( type == BackendDialType::Composite ){
        CompositeInstruction node;
        node.dial.type = BackendDialType::Formula; node.dial.inputCount = 2;
        model.compositeInputs = {{0}, {1}};
        model.compositeInstructions = {node};
        model.compositePrograms = {{0, 1, 2, 2}};
      }
    }
    CpuBackend backend;
    backend.build(view);
    PropagationInputs inputs;
    std::vector<double> table(2);
    std::uint64_t generation = 0;
    auto publish = [&](const std::vector<double>& point_){
      inputs.parameters.values = point_;
      if( producer ){
        table.assign(2, point_[0]*point_[0]+.2*point_[1]);
        inputs.externalWeights = {{table.data(), table.size(), 0, ++generation}};
        inputs.externalDialGradients.resize(model.dials.size());
        inputs.externalDialGradients[0] = {2.*point_[0], .2};
      }
      return backend.requestPropagation(inputs);
    };
    for( const auto& point : {std::vector<double>{.8, .9, .2}, {.75, .8, -.6}, {.9, .8, 1.3}} ){
      const auto analytic = backend.getLikelihoodGradient(publish(point));
      for( std::size_t i = 0; i < point.size(); ++i ){
        auto up = point, down = point;
        up[i] += 1e-5; down[i] -= 1e-5;
        const double upper = backend.getLikelihood(publish(up));
        const double lower = backend.getLikelihood(publish(down));
        EXPECT_NEAR(analytic[i], (upper-lower)/2e-5, 2e-6);
      }
    }
  }
}

namespace {
  void checkSparseProducerRowsMatchDenseSnapshotsAndRejectStaleRows(Backend& backend_) {
    auto view = gradientModel(1., 8);
    auto& model = view.propagation;
    model.dials.resize(6, model.dials[0]);
    model.dials[1].type = BackendDialType::ExternalWeight;
    model.dials[1].firstInput = 0;
    model.dials[4].type = BackendDialType::Tabulated;
    model.dials[4].firstInput = 1;
    model.dials[4].externalWeightIndex = 1;
    model.externalWeightBlocks = {{0, 3}};
    model.dials[4].tableTermCount = 2;
    model.tableTerms = {{1, .25}, {2, .75}};
    for( auto& index : model.eventDialIndices ){ index = index == 0 ? 1 : 4; }
    backend_.build(view);
    PropagationInputs inputs;
    std::uint64_t generation = 0;
    std::vector<double> table(3);
    const auto publish = [&](const std::vector<double>& point_, bool sparse_){
      inputs.parameters.values = point_;
      table = {1.+.2*point_[0]*point_[0], 1.+.1*point_[1], 1.+.3*point_[1]*point_[1]};
      inputs.externalWeights = {{table.data(), table.size(), 0, ++generation}};
      inputs.externalDialGradientIndices = sparse_ ? std::vector<std::size_t>{1, 4} : std::vector<std::size_t>{};
      inputs.externalDialGradients.assign(sparse_ ? 2 : model.dials.size(), {});
      inputs.externalDialGradients[sparse_ ? 0 : 1] = {.4*point_[0]};
      inputs.externalDialGradients[sparse_ ? 1 : 4] = {.025+.45*point_[1]};
      return backend_.requestPropagation(inputs);
    };
    for( const auto& point : {std::vector<double>{.31, .27}, {.73, -.4}} ){
      const auto dense = backend_.getLikelihoodGradient(publish(point, false));
      const auto token = publish(point, true);
      // Sparse publications must also own their Jacobian/value snapshots.
      inputs.externalDialGradients[0][0] = 123.;
      table.assign(3, 456.);
      const auto sparse = backend_.getLikelihoodGradient(token);
      EXPECT_EQ(sparse, dense);
      for( std::size_t i = 0; i < point.size(); ++i ){
        auto up = point, down = point;
        up[i] += 1e-5; down[i] -= 1e-5;
        const double upper = backend_.getLikelihood(publish(up, true));
        const double lower = backend_.getLikelihood(publish(down, true));
        EXPECT_NEAR(sparse[i], (upper-lower)/2e-5, 2e-6);
      }
    }
    publish({.31, .27}, true);
    inputs.externalDialGradientIndices = {1};
    inputs.externalDialGradients.resize(1);
    auto token = backend_.requestPropagation(inputs);
    EXPECT_THROW(static_cast<void>(backend_.getLikelihoodGradient(token)), std::exception);
    inputs.externalDialGradientIndices.clear();
    inputs.externalDialGradients.clear();
    token = backend_.requestPropagation(inputs);
    EXPECT_THROW(static_cast<void>(backend_.getLikelihoodGradient(token)), std::exception);

    for( const auto& indices : {std::vector<std::size_t>{1, 1}, {4, 1}, {1, 6}, {1, 3}} ){
      inputs.externalDialGradientIndices = indices;
      inputs.externalDialGradients.assign(2, {1.});
      EXPECT_THROW(static_cast<void>(backend_.requestPropagation(inputs)), std::exception);
    }
    inputs.externalDialGradientIndices = {1, 4};
    inputs.externalDialGradients.resize(1);
    EXPECT_THROW(static_cast<void>(backend_.requestPropagation(inputs)), std::exception);
    // A rebuild changes the compact snapshot's index map and clears publication.
    model.dials[1].type = model.dials[4].type = BackendDialType::Norm;
    model.externalWeightBlocks.clear();
    backend_.build(view);
    compareGradient(backend_, {.7, .9});
  }
}
TEST(CpuBackendGradient, SparseProducerRowsMatchDenseSnapshotsAndRejectStaleRows) {
  CpuBackend backend;
  checkSparseProducerRowsMatchDenseSnapshotsAndRejectStaleRows(backend);
}
#ifdef __APPLE__
TEST(MpsBackendGradient, SparseProducerRowsMatchDenseSnapshotsAndRejectStaleRows) {
  MpsBackend backend;
  if( !backend.getCapabilities().supportsGpu ) GTEST_SKIP() << "Metal unavailable";
  checkSparseProducerRowsMatchDenseSnapshotsAndRejectStaleRows(backend);
}
#endif
