#include "CpuBackend.h"
#include "FormulaCompiler.h"
#include "DevicePropagationModel.h"
#include "DeviceMath/FormulaMath.h"
#ifdef __APPLE__
#include "MpsBackend.h"
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

TEST(DeviceFormulaMath, CompensatedFunctionsMatchDoubleReference) {
  using namespace Backends::FormulaMath;
  auto close = [](Scalar value_, double expected_) {
    const double value = double(value_.high) + value_.low;
    EXPECT_NEAR(value, expected_, 2e-11 * std::max(1., std::abs(expected_)));
  };
  for( double x : {-10., -4., -2., -.7, -1e-5, 0., .3, 1., 2., 4., 10.} ){
    SCOPED_TRACE(x);
    Scalar value(x);
    close(exp(value), std::exp(x));
    close(sin(value), std::sin(x));
    close(cos(value), std::cos(x));
    close(sinh(value), std::sinh(x));
    close(cosh(value), std::cosh(x));
    close(atan2(value, Scalar(1)), std::atan(x));
    close(erf(value), std::erf(x));
    close(erfc(value), std::erfc(x));
    close(pow(value, Scalar(3)), std::pow(x, 3));
    if( x > 0 ){
      close(log(value), std::log(x));
      close(sqrt(value), std::sqrt(x));
      close(pow(value, Scalar(.25)), std::pow(x, .25));
    }
  }
  for( double x : {1.-1e-12, 1.+1e-12, -1.-1e-12, -1.+1e-12, 1e8+.25} ){
    close(floor(Scalar(x)), std::floor(x));
    close(ceil(Scalar(x)), std::ceil(x));
  }
  EXPECT_TRUE(std::isnan(sqrt(Scalar(-1)).high));
  EXPECT_TRUE(std::isnan(log(Scalar(-1)).high));
  EXPECT_EQ(log(Scalar(0)).high, -std::numeric_limits<float>::infinity());
  EXPECT_EQ(exp(Scalar(1000)).high, std::numeric_limits<float>::infinity());
}

namespace {
  Backends::EngineView formulaView(const std::string& expression_) {
    using namespace Backends;
    EngineView view;
    auto& model = view.propagation;
    model.parameterCount = 4;
    model.totalBins = 1;
    model.dialInputs = {{0}, {1}, {2}, {3}};
    model.formulaPrograms.push_back(FormulaCompiler::compile(expression_, {"delta", "alpha", "q3", "third", "cap"}));
    model.formulaInputs = {{0, 0.}, {1, 0.}, {-1, 750.125}, {2, 0.}, {-1, 20.}};
    BackendDialDescriptor formula;
    formula.type = BackendDialType::Formula;
    formula.inputCount = 3;
    model.dials.push_back(formula);
    // Enough controls to exercise sparse invalidation of a three-input dial.
    for( int i = 1; i < 10; ++i ){
      BackendDialDescriptor norm;
      norm.inputCount = 1;
      norm.firstInput = 3;
      model.dials.push_back(norm);
    }
    for( std::size_t i = 0; i < model.dials.size(); ++i ){
      EventView event;
      event.resultIndex = i;
      event.globalBinIndex = 0;
      model.events.push_back(event);
      model.eventWeights.push_back({1., i, 1});
      model.eventDialIndices.push_back(i);
    }
    return view;
  }
}

TEST(DeviceFormulaPacking, ValidatesOperandsAndPreservesLoadedConstants) {
  using namespace Backends;
  auto view = formulaView("[delta]+[alpha]*(.056*[q3]-41.6)+[third]");
  DevicePropagationModel packed;
  packed.build(view.propagation);
  ASSERT_EQ(packed.formulaPrograms.size(), 1);
  ASSERT_EQ(packed.programInputs.size(), 5);
  EXPECT_EQ(packed.programInputs[2].source, ProgramInputSource::Constant);
  EXPECT_DOUBLE_EQ(double(packed.programInputs[2].value.high) + packed.programInputs[2].value.low, 750.125);
  EXPECT_GT(packed.programScratchSize, 0);
  view.propagation.formulaInputs[0].parameterIndex = 3;
  EXPECT_THROW(packed.build(view.propagation), std::runtime_error);
  view = formulaView("[delta]");
  view.propagation.formulaPrograms[0].instructions[0].operand = 99;
  EXPECT_THROW(packed.build(view.propagation), std::runtime_error);
}

#ifdef __APPLE__
TEST(MpsFormula, AllOpcodesAndThreeParameterCacheMatchCpu) {
  using namespace Backends;
  const std::vector<std::string> expressions{
      "[delta]+[alpha]*(.056*[q3]-41.6)+[third]", "min([delta],[cap])+max([alpha],[third])",
      "exp([delta])+log([q3])+log10([q3])+sqrt([q3])+abs([alpha])",
      "sin([delta])+cos([alpha])+tan([third])", "asin([delta])+acos([alpha])+atan([third])",
      "atan2([delta],[alpha])+pow([q3],.25)+[third]^3", "-[delta]^2+[alpha]/[q3]",
      "sinh([delta])+cosh([alpha])+tanh([third])", "erf([delta])+erfc([alpha])+floor([third])+ceil([delta])",
      "2.5", "[q3]"};
  for( const auto* mode : {"cachedResponses", "recomputeResponses"} ){
    MpsBackend gpu;
    ASSERT_NE(gpu.getCapabilities().deviceName, "Metal unavailable");
    gpu.configure(ConfigReader(JsonType{{"propagationMode", mode}}));
    for( const auto& expression : expressions ){
      SCOPED_TRACE(expression);
      auto view = formulaView(expression);
      CpuBackend cpu;
      cpu.build(view);
      gpu.build(view);
      EXPECT_EQ(JsonType::parse(gpu.getDiagnostics()).at("uniqueDialsByType").at("Formula"), 1);
      const std::vector<std::vector<double>> points{{.3, -.2, .4, 1.}, {.3, .5, .4, 1.},
          {.3, .5, -.1, 1.}, {-.6, .5, -.1, 1.}, {-.6, .5, -.1, 1.}, {-.6, .5, -.1, 1.2},
          {.2, .3, .4, 1.2}, {.2, .3, .4, 1.2}};
      for( std::size_t i = 0; i < points.size(); ++i ){
        PropagationInputs inputs;
        inputs.parameters.values = points[i];
        auto reference = cpu.requestPropagation(inputs), token = gpu.requestPropagation(inputs);
        gpu.materialize(token, OutputRequest::EventWeights);
        const auto& expected = cpu.getEventWeightsHostView(reference);
        const auto& actual = gpu.getEventWeightsHostView(token);
        ASSERT_EQ(actual.size(), expected.size());
        for( std::size_t j = 0; j < expected.size(); ++j ){
          EXPECT_NEAR(actual[j], expected[j], 2e-10 * std::max(1., std::abs(expected[j])));
        }
        auto state = JsonType::parse(gpu.getDiagnosticEventPropagationState());
        if( std::string(mode) == "cachedResponses" ){
          if( i == 4 or i == 7 ){ EXPECT_EQ(state.at("evaluatedDials"), 0); }
          if( (i > 0 and i < 4) or i == 6 ){ EXPECT_EQ(state.at("evaluatedDials"), 1); }
        }
      }
    }
  }
}

TEST(MpsFormula, EventOnlyFormulaNeedsNoParameterBuffer) {
  using namespace Backends;
  EngineView view;
  auto& model = view.propagation;
  model.totalBins = 1;
  model.formulaPrograms.push_back(FormulaCompiler::compile("1+.001*[q3]", {"q3"}));
  model.formulaInputs.push_back({-1, 750.125});
  BackendDialDescriptor dial;
  dial.type = BackendDialType::Formula;
  model.dials.push_back(dial);
  EventView event;
  event.globalBinIndex = 0;
  model.events.push_back(event);
  model.eventWeights.push_back({1., 0, 1});
  model.eventDialIndices.push_back(0);
  MpsBackend gpu;
  gpu.build(view);
  for( int i = 0; i < 2; ++i ){
    const auto token = gpu.requestPropagation(PropagationInputs{});
    gpu.materialize(token, OutputRequest::EventWeights);
    EXPECT_NEAR(gpu.getEventWeightsHostView(token)[0], 1.750125, 1e-12);
    if( i ){ EXPECT_EQ(JsonType::parse(gpu.getDiagnosticEventPropagationState()).at("evaluatedDials"), 0); }
  }
}

TEST(MpsFormula, InfiniteIntermediatesCanBeLimitedByTheNodeSupervisor) {
  using namespace Backends;
  for( const auto& expression : {"log10([delta])", "exp([delta])+cosh([delta])"} ){
    auto view = formulaView(expression);
    auto& dial = view.propagation.dials[0];
    dial.hasMinResponse = dial.hasMaxResponse = true;
    dial.minResponse = -5.;
    dial.maxResponse = 5.;
    MpsBackend gpu;
    gpu.build(view);
    CpuBackend cpu;
    cpu.build(view);
    for( double parameter : {0., 100.} ){
      PropagationInputs inputs;
      inputs.parameters.values = {parameter, .2, .3, 1.};
      auto reference = cpu.requestPropagation(inputs), token = gpu.requestPropagation(inputs);
      gpu.materialize(token, OutputRequest::EventWeights);
      EXPECT_NEAR(gpu.getEventWeightsHostView(token)[0], cpu.getEventWeightsHostView(reference)[0], 1e-12);
    }
  }
}
#endif
