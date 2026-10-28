#include "CudaBackend.h"
#include "CpuBackend.h"
#include "CudaDialPrograms.h"
#include "CpuCompositeProgram.h"
#include "FormulaCompiler.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {
  using namespace Backends;

  EngineView programModel(const std::string& expression_) {
    EngineView view;
    auto& model = view.propagation;
    model.parameterCount = 4;
    model.totalBins = 1;
    model.dialInputs = {{0, true, -1., 2.}, {1}, {2}, {3}};
    model.formulaPrograms.push_back(FormulaCompiler::compile(expression_, {"delta", "alpha", "q3", "third"}));
    model.formulaInputs = {{0, 0.}, {1, 0.}, {-1, 750.125}, {2, 0.}};
    BackendDialDescriptor formula;
    formula.type = BackendDialType::Formula;
    formula.inputCount = 3;
    model.dials.push_back(formula);
    // A formula followed by a graph, with a distinct mirror on its incoming edge.
    model.compositeInputs = {{0}, {1}, {2}, {3, true, -10., 20.}};
    model.compositeInstructions.push_back({formula, -1e20, 1e20});
    BackendDialDescriptor graph;
    graph.type = BackendDialType::Graph;
    graph.firstInput = 3;
    graph.inputCount = 1;
    graph.payloadSize = 6;
    graph.hasMaxResponse = true;
    graph.maxResponse = 1.4;
    model.dialPayloads = {.25, -10., 1., 0., 2., 10.};
    model.compositeInstructions.push_back({graph, -10., 10.});
    model.compositePrograms.push_back({0, 2, 3, 4});
    BackendDialDescriptor composite;
    composite.type = BackendDialType::Composite;
    composite.inputCount = 3;
    model.dials.push_back(composite);
    // Interleave a surface to exercise root remapping and workspace ownership.
    BackendDialDescriptor surface;
    surface.type = BackendDialType::Bilinear;
    surface.inputCount = 2;
    surface.payloadOffset = model.dialPayloads.size();
    surface.payloadSize = 10;
    model.dialPayloads.insert(model.dialPayloads.end(), {2., 2., -1., 1., -1., 1., .5, 1., 1.5, 2.});
    model.dials.insert(model.dials.begin(), surface);
    for( int i = 0; i < 20; ++i ){
      BackendDialDescriptor norm;
      norm.firstInput = 3;
      norm.inputCount = 1;
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

  EngineView constantModel(const std::string& expression_) {
    EngineView view;
    auto& model = view.propagation;
    model.totalBins = 1;
    model.formulaPrograms.push_back(FormulaCompiler::compile(expression_, {"q3"}));
    model.formulaInputs = {{-1, 750.125}};
    BackendDialDescriptor formula;
    formula.type = BackendDialType::Formula;
    formula.hasMinResponse = formula.hasMaxResponse = true;
    formula.minResponse = -7.;
    formula.maxResponse = 7.;
    model.compositeInstructions.push_back({formula, 0., 0.});
    model.compositePrograms.push_back({0, 1, 0, 0});
    auto composite = formula;
    composite.type = BackendDialType::Composite;
    model.dials = {formula, composite};
    model.events.resize(2);
    model.events[0].globalBinIndex = model.events[1].globalBinIndex = 0;
    model.events[1].resultIndex = 1;
    model.eventWeights = {{1., 0, 1}, {1., 1, 1}};
    model.eventDialIndices = {0, 1};
    return view;
  }

  const std::vector<std::string> expressions{
      "[delta]+[alpha]*(.056*[q3]-41.6)+[third]", "min([delta],2)+max([alpha],[third])",
      "exp([delta])+log([q3])+log10([q3])+sqrt([q3])+abs([alpha])",
      "sin([delta])+cos([alpha])+tan([third])", "asin([delta])+acos([alpha])+atan([third])",
      "atan2([delta],[alpha])+pow([q3],.25)+[third]^3", "-[delta]^2+[alpha]/[q3]",
      "sinh([delta])+cosh([alpha])+tanh([third])", "erf([delta])+erfc([alpha])+floor([third])+ceil([delta])",
      "2.5", "[q3]"};
  const std::vector<std::vector<double>> points{
      {.3, -.2, .4, 1.}, {.3, .5, .4, 1.}, {.3, .5, -.1, 1.}, {-.6, .5, -.1, 1.},
      {-.6, .5, -.1, 1.}, {-.6, .5, -.1, 1.2}, {.2, .3, .4, 1.2}, {.2, .3, .4, 1.2}};
}

TEST(CudaBackendPrograms, NativeInterpreterMatchesCpuForEveryOpcodeAndComposite) {
  for( const auto& expression : expressions ){
    SCOPED_TRACE(expression);
    const auto view = programModel(expression);
    const auto model = makeCudaWeightModel(view);
    std::vector<double> scratch(model.programScratchSize), referenceScratch(128), formulaScratch(128);
    const CudaProgramView program{model.formulaPrograms.data(), model.formulaInstructions.data(),
        model.formulaConstants.data(), model.formulaInputs.data(), model.compositePrograms.data(),
        model.compositeInstructions.data(), model.compositeInputs.data(), model.inputs.data(),
        model.payloads.data(), model.programScratchOffsets.data(), scratch.data()};
    EXPECT_EQ(model.eventDialIndices[0], model.surfaceOffset);
    EXPECT_DOUBLE_EQ(model.formulaInputs[2].constant, 750.125);
    for( const auto& point : points ){
      for( std::size_t i = 0; i < model.dials.size(); ++i ){
        const auto& dial = model.dials[i];
        if( dial.type != BackendDialType::Formula and dial.type != BackendDialType::Composite ){ continue; }
        const auto expected = dial.type == BackendDialType::Formula
            ? evaluateFormulaDial(view.propagation, dial, point.data(), formulaScratch.data())
            : evaluateComposite(view.propagation, dial, point.data(), referenceScratch.data(), formulaScratch.data());
        EXPECT_NEAR(evaluateCudaProgram(program, dial, i, point.data()), expected,
                    2e-13 * std::max(1., std::abs(expected)));
      }
    }
  }
}

TEST(CudaBackendPrograms, ZeroParameterProgramsPreserveEventConstantsAndClampInfinities) {
  for( const auto& expression : {"1+.001*[q3]", "exp(1000)", "log(0)"} ){
    const auto view = constantModel(expression);
    const auto model = makeCudaWeightModel(view);
    std::vector<double> scratch(model.programScratchSize);
    const CudaProgramView program{model.formulaPrograms.data(), model.formulaInstructions.data(),
        model.formulaConstants.data(), model.formulaInputs.data(), model.compositePrograms.data(),
        model.compositeInstructions.data(), model.compositeInputs.data(), model.inputs.data(),
        model.payloads.data(), model.programScratchOffsets.data(), scratch.data()};
    const double expected = std::string(expression) == "exp(1000)" ? 7.
        : (std::string(expression) == "log(0)" ? -7. : 1.750125);
    for( std::size_t i = 0; i < model.dials.size(); ++i ){
      EXPECT_DOUBLE_EQ(evaluateCudaProgram(program, model.dials[i], i, nullptr), expected);
    }
  }
}

TEST(CudaBackendPrograms, ThreeParameterInvalidationIsCompleteAndDisjoint) {
  const auto model = makeCudaWeightModel(programModel("[delta]+[alpha]+[third]"));
  ASSERT_EQ(model.multiInputGroups.size(), 1);
  CudaResponseUpdatePlan plan;
  const std::vector<double> previous{.3, .2, .1, 1.};
  for( std::size_t parameter = 0; parameter < 3; ++parameter ){
    auto current = previous;
    current[parameter] += .1;
    plan.select(model.parameterDialOffsets, previous, current, model.surfaceGroups, model.multiInputGroups);
    EXPECT_EQ(plan.dialCount, parameter < 2 ? 3 : 2);
  }
  plan.select(model.parameterDialOffsets, previous, {-.3, -.2, -.1, 1.}, model.surfaceGroups, model.multiInputGroups);
  EXPECT_EQ(plan.dialCount, 3); // Every root appears exactly once despite three changed inputs.
  plan.select(model.parameterDialOffsets, previous, previous, model.surfaceGroups, model.multiInputGroups);
  EXPECT_EQ(plan.dialCount, 0);
}

TEST(CudaBackendPrograms, RejectsMalformedBytecodeBindingsAndForwardReferences) {
  auto view = programModel("[delta]");
  view.propagation.formulaPrograms[0].instructions[0].operand = 100;
  EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
  view = programModel("[delta]");
  view.propagation.formulaPrograms[0].stackSize = 0;
  EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
  view = programModel("[delta]");
  view.propagation.formulaInputs[0].parameterIndex = 3;
  EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
  view = programModel("[delta]");
  view.propagation.compositeInputs[0].parameterIndex = 3;
  EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
  view = programModel("[delta]");
  view.propagation.compositeInputs[0].useMirror = true;
  EXPECT_THROW((void)makeCudaWeightModel(view), std::runtime_error);
}

#ifdef GUNDAM_TEST_CUDA_DEVICE
TEST(CudaBackendPrograms, DeviceZeroParameterProgramsAndClampedInfinities) {
  for( const auto* mode : {"cachedResponses", "recomputeResponses"} ){
    for( const auto& expression : {"1+.001*[q3]", "exp(1000)", "log(0)"} ){
      const auto view = constantModel(expression);
      CudaBackend cuda;
      cuda.configure(ConfigReader(JsonType{{"propagationMode", mode}}));
      cuda.build(view);
      const double expected = std::string(expression) == "exp(1000)" ? 7.
          : (std::string(expression) == "log(0)" ? -7. : 1.750125);
      for( int i = 0; i < 2; ++i ){
        const auto token = cuda.requestPropagation(PropagationInputs{});
        const auto& weights = cuda.getEventWeightsHostView(token);
        ASSERT_EQ(weights.size(), 2);
        for( double weight : weights ){ EXPECT_NEAR(weight, expected, 1e-12); }
        if( i and std::string(mode) == "cachedResponses" ){
          EXPECT_EQ(cuda.getLastTimingSummary().cacheCandidateCount, 0);
        }
      }
    }
  }
}

TEST(CudaBackendPrograms, DeviceMatchesCpuInBothModesAndLayouts) {
  for( const auto* mode : {"cachedResponses", "recomputeResponses"} ){
    for( const auto* layout : {"eventMajor", "warpInterleaved"} ){
      for( const auto& expression : expressions ){
        SCOPED_TRACE(::testing::Message() << mode << " " << layout << " " << expression);
        const auto view = programModel(expression);
        CpuBackend cpu;
        CudaBackend cuda;
        cuda.configure(ConfigReader(JsonType{{"propagationMode", mode}, {"weightLayout", layout}}));
        cpu.build(view);
        cuda.build(view);
        for( std::size_t i = 0; i < points.size(); ++i ){
          PropagationInputs inputs;
          inputs.parameters.values = points[i];
          const auto reference = cpu.requestPropagation(inputs), token = cuda.requestPropagation(inputs);
          const auto& expected = cpu.getEventWeightsHostView(reference);
          const auto& actual = cuda.getEventWeightsHostView(token);
          ASSERT_EQ(actual.size(), expected.size());
          for( std::size_t j = 0; j < actual.size(); ++j ){
            EXPECT_NEAR(actual[j], expected[j], 3e-12 * std::max(1., std::abs(expected[j])));
          }
          EXPECT_NEAR(cuda.getHistogramSumsHostView(token)[0], cpu.getHistogramSumsHostView(reference)[0], 1e-9);
          if( std::string(mode) == "cachedResponses" and (i == 4 or i == 7) ){
            EXPECT_EQ(cuda.getLastTimingSummary().cacheCandidateCount, 0);
          }
        }
      }
    }
  }
}
#endif
