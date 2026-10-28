#include "CpuBackend.h"
#include "CpuFormulaProgram.h"
#include "FormulaCompiler.h"
#include "EngineLayout.h"
#include "LikelihoodInterface.h"
#include "RootFormula.h"
#include "GundamGlobals.h"

#include <TFormula.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace {
  double evaluate(const Backends::FormulaProgram& program_, const std::vector<double>& inputs_) {
    std::vector<double> stack(program_.stackSize);
    return Backends::evaluateFormula(program_, [&](std::size_t index_){ return inputs_.at(index_); }, stack.data());
  }

  void compare(double actual_, double expected_) {
    if( std::isnan(expected_) ){ EXPECT_TRUE(std::isnan(actual_)); }
    else if( std::isinf(expected_) ){ EXPECT_EQ(actual_, expected_); }
    else{ EXPECT_NEAR(actual_, expected_, 2e-13 * std::max(1., std::abs(expected_))); }
  }
}

TEST(BackendFormula, ArithmeticFunctionsAndPrecedenceMatchRoot) {
  const std::vector<std::string> expressions{
      "[delta]+[alpha]*(0.056*[q3]-41.6)",
      "-2^2", "(-2)^2", "2^3^2", "2**3", "2^-2", "-[delta]^2+[alpha]/[q3]",
      "+[delta]--[alpha]", "1.5e-3+.2E+2*[delta]", "pow([delta],3)",
      "exp([delta])+log([q3])+log10([q3])+sqrt([q3])+abs([alpha])",
      "sin([delta])+cos([delta])+tan([delta])",
      "asin([delta])+acos([delta])+atan([delta])+atan2([delta],[alpha])",
      "sinh([delta])+cosh([delta])+tanh([delta])",
      "erf([delta])+erfc([delta])+floor([alpha])+ceil([alpha])",
      "min([delta],[alpha])+max([delta],[alpha])",
      "TMath::Exp([delta])+TMath::Power([q3],2)+TMath::Pi()",
      "std::sin([delta])+TMath::TwoPi()", "pi", "2.5"};
  for( const auto& expression : expressions ){
    SCOPED_TRACE(expression);
    TFormula root("backendFormulaReference", expression.c_str(), false);
    ASSERT_TRUE(root.IsValid());
    const auto program = Backends::FormulaCompiler::compile(expression, {"q3", "alpha", "delta"});
    for( double delta : {-.7, 0., .3, .8} ){
      std::vector<double> values{7.5, -.25, delta};
      std::vector<double> parameters(root.GetNpar());
      for( int i = 0; i < root.GetNpar(); ++i ){
        const std::string name = root.GetParName(i);
        parameters[i] = name == "delta" ? delta : name == "alpha" ? values[1] : values[0];
      }
      compare(evaluate(program, values), root.EvalPar(nullptr, parameters.data()));
    }
  }
}

TEST(BackendFormula, ConstantsAndDomainErrors) {
  const auto program = Backends::FormulaCompiler::compile("[a]+[bin]", {"unused", "a"}, {{"bin", 4.5}});
  EXPECT_DOUBLE_EQ(evaluate(program, {99., 2.}), 6.5);
  for( const auto& expression : {"sqrt([a])", "log([a])", "1/[a]", "asin([a])", "exp([a])"} ){
    TFormula root("formulaDomain", expression, false);
    ASSERT_TRUE(root.IsValid());
    const auto code = Backends::FormulaCompiler::compile(expression, {"a"});
    for( double value : {-2., 0., 2., 1000., std::numeric_limits<double>::quiet_NaN()} ){
      compare(evaluate(code, {value}), root.EvalPar(nullptr, &value));
    }
  }
}

TEST(BackendFormula, RejectsUnsupportedExpressionsAndInvalidBytecode) {
  for( const auto& expression : {"", "[missing]", "[a", "[a]+", "[a][a]", "sin()", "sin(1,2)",
      "pow(2)", "max(1,2,3)", "(1+2", "x[0]", "gaus(0)", "[a]>0?1:0", "[a]=1", "1;2", "1e+"} ){
    SCOPED_TRACE(expression);
    EXPECT_THROW(Backends::FormulaCompiler::compile(expression, {"a"}), std::runtime_error);
  }
  EXPECT_THROW(Backends::FormulaCompiler::compile("[a]", {"a", "a"}), std::runtime_error);
  EXPECT_THROW(Backends::FormulaCompiler::compile(std::string(200, '-') + "1", {}), std::runtime_error);
  auto code = Backends::FormulaCompiler::compile("[a]+1", {"a"});
  code.instructions[0].operand = 1;
  EXPECT_THROW(code.validate(), std::runtime_error);
  code.instructions[0] = {Backends::FormulaOpcode::Add, 0};
  EXPECT_THROW(code.validate(), std::runtime_error);
  code.instructions[0] = {static_cast<Backends::FormulaOpcode>(999), 0};
  EXPECT_THROW(code.validate(), std::runtime_error);
  code = Backends::FormulaCompiler::compile("1", {});
  code.stackSize = 0;
  EXPECT_THROW(code.validate(), std::runtime_error);
}

TEST(BackendFormula, CpuCacheHandlesConstantsMirroringAndParameterChanges) {
  using namespace Backends;
  EngineView view;
  auto& model = view.propagation;
  model.parameterCount = 2;
  model.totalBins = 1;
  model.dialInputs = {{0, true, -1., 2.}, {1}};
  model.formulaPrograms.push_back(FormulaCompiler::compile("2+sin([a])+[b]^2", {"a", "b"}));
  model.formulaPrograms.push_back(FormulaCompiler::compile("2+cos([a])", {"a"}));
  model.formulaPrograms.push_back(FormulaCompiler::compile("2.5", {}));
  // More than eight dials exercise the mixed shared-input cache path.
  for( int i = 0; i < 12; ++i ){
    BackendDialDescriptor dial;
    dial.type = i < 3 ? BackendDialType::Formula : BackendDialType::Norm;
    dial.inputCount = i == 0 ? 2 : i == 2 ? 0 : 1;
    dial.formulaProgramIndex = std::min(i, 2);
    dial.firstFormulaInput = model.formulaInputs.size();
    for( std::size_t j = 0; j < dial.inputCount; ++j ){ model.formulaInputs.push_back({int(j), 0.}); }
    model.dials.push_back(dial);
    EventView event;
    event.resultIndex = i;
    event.globalBinIndex = 0;
    model.events.push_back(event);
    model.eventDialIndices.push_back(i);
    model.eventWeights.push_back({1.25, std::size_t(i), 1});
  }
  model.dials[0].hasMinResponse = model.dials[0].hasMaxResponse = true;
  model.dials[0].minResponse = 1.5;
  model.dials[0].maxResponse = 3.;
  const auto previousThreads = GundamGlobals::getNbCpuThreads();
  for( int threads : {1, 3} ){
    GundamGlobals::setNumberOfThreads(threads);
    for( auto traversal : {CpuCacheTraversal::Auto, CpuCacheTraversal::Indexed} ){
      CpuBackend backend;
      backend.build(view);
      backend.setDiagnosticCacheTraversal(traversal);
      for( const auto& point : {std::array<double, 2>{.2, .3}, {.2, .3}, {.2, 2.}, {3., 2.}, {-.5, .1}} ){
        PropagationInputs inputs;
        inputs.parameters.values = {point[0], point[1]};
        const auto token = backend.requestPropagation(inputs);
        const auto& weights = backend.getEventWeightsHostView(token);
        const double a = Semantics::transformDialInput(model.dialInputs[0], point[0]);
        compare(weights[0], 1.25 * std::clamp(2 + std::sin(a) + point[1] * point[1], 1.5, 3.));
        compare(weights[1], 1.25 * (2 + std::cos(a)));
        compare(weights[2], 1.25 * 2.5);
        compare(weights[3], 1.25 * a);
        const auto repeat = backend.requestPropagation(inputs);
        EXPECT_EQ(backend.getLastTimingSummary().cacheCandidateCount, 0);
        compare(backend.getEventWeightsHostView(repeat)[0], weights[0]);
      }
    }
  }
  GundamGlobals::setNumberOfThreads(previousThreads);
  auto invalid = view;
  invalid.propagation.formulaInputs[0].parameterIndex = 2;
  CpuBackend backend;
  EXPECT_THROW(backend.build(invalid), std::runtime_error);
}

TEST(BackendFormula, EngineLayoutPreservesNamedOrderAndBinningConstants) {
  LikelihoodInterface likelihood;
  auto& propagator = likelihood.getModelPropagator();
  auto& samples = propagator.getSampleSet().getSampleList();
  samples.resize(1);
  samples[0].setIndex(0);
  samples[0].configure(ConfigReader(JsonType{
      {"name", "sample"}, {"binning", {{"binningDefinition", {
          {{"name", "X"}, {"edges", {0., 1.}}}}}}}}));
  std::vector<ParameterSet> parameters(1);
  for( int i = 0; i < 3; ++i ){ parameters[0].getParameterList().emplace_back(&parameters[0]); }
  DialInputBuffer buffer;
  buffer.setParSetRef(&parameters);
  for( int i = 0; i < 3; ++i ){
    DialInputBuffer::ParameterReference ref;
    ref.parSetIndex = 0; ref.parIndex = i;
    buffer.addParameterReference(ref);
  }
  buffer.initialise();
  RootFormula formula;
  formula.setFormulaStr("[delta]+[alpha]*(0.056*[q3]-41.6)+[bin]");
  formula.setInputNames({"q3", "delta", "alpha"}, {"bin"});
  formula.getFormula().SetParameter("bin", 2.5);
  DialInterface interface;
  interface.setDial(DialBaseObject(formula));
  interface.setInputBufferRef(&buffer);
  Event event;
  event.getIndices().sample = event.getIndices().bin = 0;
  event.getWeights().base = 1.5;
  auto& cache = propagator.getEventDialCache().getCache();
  cache.push_back({&event});
  cache.back().dialResponseCacheList.emplace_back(interface);
  Backends::EngineLayout layout;
  for( int rebuild = 0; rebuild < 2; ++rebuild ){
    layout.build(likelihood);
    ASSERT_EQ(layout.view.propagation.formulaPrograms.size(), 1);
    Backends::CpuBackend backend;
    backend.build(layout.view);
    for( const auto& point : {std::array<double, 3>{750., .5, .25}, {900., -.3, .7}} ){
      for( int i = 0; i < 3; ++i ){ parameters[0].getParameterList()[i].setParameterValue(point[i]); }
      buffer.update();
      Backends::PropagationInputs inputs;
      for( const auto& binding : layout.bindings.parameters ){ inputs.parameters.values.push_back(binding.parameter->getParameterValue()); }
      const double expected = 1.5 * formula.evalResponse(buffer);
      const auto token = backend.requestPropagation(inputs);
      compare(backend.getEventWeightsHostView(token)[0], expected);
    }
  }
}

TEST(BackendFormula, CompositeUsesScratchInputsAndValidatesFormulaBindings) {
  using namespace Backends;
  EngineView view;
  auto& model = view.propagation;
  model.parameterCount = 1;
  model.totalBins = 1;
  model.dialInputs.emplace_back(0);
  model.compositeInputs.emplace_back(0);
  model.formulaPrograms.push_back(FormulaCompiler::compile("[parameter]+[event]", {"parameter", "event"}));
  model.formulaInputs = {{0, 0.}, {-1, 3.}};
  CompositeInstruction instruction;
  instruction.dial.type = BackendDialType::Formula;
  instruction.dial.inputCount = 1;
  model.compositeInstructions.push_back(instruction);
  model.compositePrograms.push_back({0, 1, 1, 1});
  BackendDialDescriptor root;
  root.type = BackendDialType::Composite;
  root.inputCount = 1;
  model.dials.push_back(root);
  EventView event;
  event.resultIndex = 0;
  event.globalBinIndex = 0;
  model.events.push_back(event);
  model.eventWeights.push_back({1., 0, 1});
  model.eventDialIndices.push_back(0);
  CpuBackend backend;
  backend.build(view);
  PropagationInputs inputs;
  for( double parameter : {1., 1., 2., -1.} ){
    inputs.parameters.values = {parameter};
    const auto token = backend.requestPropagation(inputs);
    EXPECT_DOUBLE_EQ(backend.getEventWeightsHostView(token)[0], parameter + 3.);
  }
  // The instruction's inputs live in compositeInputs, not dialInputs.
  model.dialInputs.emplace_back(0);
  model.compositeInstructions[0].dial.firstInput = 1;
  EXPECT_THROW(backend.build(view), std::runtime_error);
  model.compositeInstructions[0].dial.firstInput = 0;
  model.formulaInputs[0].parameterIndex = 1;
  EXPECT_THROW(backend.build(view), std::runtime_error);
  model.formulaInputs[0].parameterIndex = 0;
  model.compositeInstructions[0].dial.formulaProgramIndex = 1;
  EXPECT_THROW(backend.build(view), std::runtime_error);
}

TEST(BackendFormula, CacheDependenciesDeduplicateRepeatedAndSimultaneouslyChangedParameters) {
  using namespace Backends;
  EngineView view;
  auto& model = view.propagation;
  model.parameterCount = 3;
  model.totalBins = 1;
  model.dialInputs = {{0}, {0}, {1}};
  model.formulaPrograms.push_back(FormulaCompiler::compile("[a]+[b]+[c]", {"a", "b", "c"}));
  model.formulaInputs = {{0, 0.}, {1, 0.}, {2, 0.}};
  BackendDialDescriptor dial;
  dial.type = BackendDialType::Formula;
  dial.inputCount = 3;
  model.dials.push_back(dial);
  EventView event;
  event.globalBinIndex = 0;
  model.events.push_back(event);
  model.eventWeights.push_back({1., 0, 1});
  model.eventDialIndices.push_back(0);
  for( auto traversal : {CpuCacheTraversal::Auto, CpuCacheTraversal::Indexed} ){
    CpuBackend backend;
    backend.build(view);
    backend.setDiagnosticCacheTraversal(traversal);
    const std::vector<std::vector<double>> points{{1., 2., 0.}, {2., 2., 0.}, {3., 4., 0.},
                                                  {3., 4., 1.}, {3., 4., 1.}, {3., 5., 1.}};
    const std::vector<std::size_t> candidates{1, 1, 1, 0, 0, 1};
    for( std::size_t i = 0; i < points.size(); ++i ){
      PropagationInputs inputs;
      inputs.parameters.values = points[i];
      const auto token = backend.requestPropagation(inputs);
      EXPECT_DOUBLE_EQ(backend.getEventWeightsHostView(token)[0], 2 * points[i][0] + points[i][1]);
      EXPECT_EQ(backend.getLastTimingSummary().cacheCandidateCount, candidates[i]);
      EXPECT_EQ(backend.getLastTimingSummary().cacheInputTransformCount, 3 * candidates[i]);
    }
    // A rebuild must reset dependency generations and response validity.
    backend.build(view);
    PropagationInputs inputs;
    inputs.parameters.values = points.back();
    const auto token = backend.requestPropagation(inputs);
    EXPECT_EQ(backend.getLastTimingSummary().cacheCandidateCount, 1);
    EXPECT_DOUBLE_EQ(backend.getEventWeightsHostView(token)[0], 11.);
  }
}
