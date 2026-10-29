#include "CudaBackend.h"
#include "CpuBackend.h"
#include "CudaDialPrograms.h"
#include "CpuCompositeProgram.h"
#include "FormulaCompiler.h"
#include "UniformSpline.h"
#include "GeneralSpline.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
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

  EngineView chainModel(BackendDialType leaf_, bool mirror_) {
    EngineView view;
    auto& m = view.propagation;
    m.parameterCount = 2; m.totalBins = 1;
    m.dialInputs = {{0, mirror_, -2., 4.}, {1}};
    m.formulaPrograms = {FormulaCompiler::compile("[d]+[a]*(.056*[q]-41.6)", {"d", "a", "q"}),
                         FormulaCompiler::compile("min([shift],[cap])", {"shift", "cap"})};
    std::vector<double> payload{.5, -40., 1., 0., 2., 40.};
    if( leaf_ == BackendDialType::UniformSpline ){
      UniformSpline spline; spline.buildDial({{-40., .5, .01}, {0., 1., .02}, {40., 2., .03}});
      payload = spline.getDialData();
    }
    else if( leaf_ == BackendDialType::GeneralSpline ){
      GeneralSpline spline; spline.buildDial({{-40., .5, .01}, {-5., 1., .02}, {40., 2., .03}});
      payload = spline.getDialData();
    }
    for( std::size_t i = 0; i < 6; ++i ){
      const auto first = m.compositeInstructions.size();
      BackendDialDescriptor formula;
      formula.type = BackendDialType::Formula; formula.inputCount = 2;
      formula.firstInput = m.compositeInputs.size(); formula.firstFormulaInput = m.formulaInputs.size();
      formula.hasMinResponse = formula.hasMaxResponse = true; formula.minResponse = -40.; formula.maxResponse = 40.;
      m.formulaInputs.insert(m.formulaInputs.end(), {{0, 0.}, {1, 0.}, {-1, 500. + .25*i}});
      m.compositeInputs.insert(m.compositeInputs.end(), {{0, mirror_, -1., 3.}, {1}});
      m.compositeInstructions.push_back({formula, 0., 0.});
      BackendDialDescriptor leaf;
      leaf.type = leaf_; leaf.inputCount = 1; leaf.firstInput = m.compositeInputs.size();
      leaf.payloadOffset = m.dialPayloads.size(); leaf.payloadSize = payload.size();
      m.dialPayloads.insert(m.dialPayloads.end(), payload.begin(), payload.end());
      m.compositeInputs.push_back({2, mirror_, -30., 60.});
      m.compositeInstructions.push_back({leaf, -40., 40.});
      BackendDialDescriptor cap;
      cap.type = BackendDialType::Formula; cap.inputCount = 1; cap.formulaProgramIndex = 1;
      cap.firstInput = m.compositeInputs.size(); cap.firstFormulaInput = m.formulaInputs.size();
      m.compositeInputs.push_back({3, mirror_, -3., 7.});
      m.formulaInputs.insert(m.formulaInputs.end(), {{0, 0.}, {-1, 1.1 + .01*i}});
      m.compositeInstructions.push_back({cap, 0., 0.});
      m.compositePrograms.push_back({first, 3, 2, 4});
      BackendDialDescriptor root; root.type = BackendDialType::Composite; root.inputCount = 2; root.compositeProgram = i;
      m.dials.push_back(root);
      EventView event; event.resultIndex = i; event.globalBinIndex = 0;
      m.events.push_back(event); m.eventWeights.push_back({1., i, 1}); m.eventDialIndices.push_back(i);
    }
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
        model.payloads.data(), model.programScratchOffsets.data(), scratch.data(),
        model.compositeInstances.data(), model.compositeNodeData.data()};
    EXPECT_EQ(model.eventDialIndices[0], model.surfaceOffset);
    EXPECT_LE(model.formulaPrograms.size(), view.propagation.formulaPrograms.size());
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
        model.payloads.data(), model.programScratchOffsets.data(), scratch.data(),
        model.compositeInstances.data(), model.compositeNodeData.data()};
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

TEST(CudaBackendPrograms, CompactChainsShareTemplatesAndPreserveAllEdges) {
  for( const auto leaf : {BackendDialType::Graph, BackendDialType::UniformSpline, BackendDialType::GeneralSpline} ){
    for( const bool mirror : {false, true} ){
      const auto view = chainModel(leaf, mirror);
      NativeDialProgramModel native; native.build(view.propagation, view.propagation.dials);
      const auto gpu = makeCudaWeightModel(view);
      ASSERT_EQ(native.compositePrograms.size(), 1);
      ASSERT_EQ(native.compositeInstructions.size(), 3);
      EXPECT_EQ(native.compositeInstances.size(), 6);
      ASSERT_EQ(native.formulaPrograms.size(), 2);
      EXPECT_EQ(native.formulaPrograms[0].instructions.size(), 5);
      EXPECT_EQ(native.formulaPrograms[0].nativePlan.kernel, NativeFormulaKernel::MultiplyThenAdd);
      EXPECT_EQ(native.formulaPrograms[1].nativePlan.operation, FormulaOpcode::Min);
      ASSERT_TRUE(native.compositePrograms[0].isChain);
      EXPECT_EQ(gpu.programScratchSize, 0);
      EXPECT_TRUE(gpu.onlyFastPrograms);
      EXPECT_TRUE(gpu.programScratchOffsets.empty());
      std::vector<double> scratch(gpu.programScratchSize);
      const CudaProgramView packed{gpu.formulaPrograms.data(), gpu.formulaInstructions.data(), gpu.formulaConstants.data(),
          gpu.formulaInputs.data(), gpu.compositePrograms.data(), gpu.compositeInstructions.data(), gpu.compositeInputs.data(),
          gpu.inputs.data(), gpu.payloads.data(), gpu.programScratchOffsets.data(), scratch.data(),
          gpu.compositeInstances.data(), gpu.compositeNodeData.data()};
      double graph[64]{}, formula[64]{};
      for( const double x : {-1000., -40., -1.2, -0., .3, 40., 1000.} ){
        for( const double alpha : {-2., -0., .7, 1e308} ){
          const double parameters[]{x, alpha};
          for( std::size_t i = 0; i < view.propagation.dials.size(); ++i ){
            const auto expected = evaluateComposite(view.propagation, view.propagation.dials[i], parameters, graph, formula);
            const auto bits = [](double value_){ std::uint64_t result; std::memcpy(&result, &value_, sizeof(result)); return result; };
            EXPECT_EQ(bits(evaluatePackedComposite(native, view.propagation.dials[i], parameters, nullptr, nullptr)), bits(expected));
            const auto index = gpu.eventDialIndices[i];
            EXPECT_EQ(bits(evaluateCudaProgram(packed, gpu.dials[index], index, parameters)), bits(expected));
            EXPECT_EQ(bits(evaluateCudaFastProgram(packed, gpu.dials[index], parameters)), bits(expected));
          }
        }
      }
    }
  }
}

TEST(CudaBackendPrograms, DependencyRangesRemainDirectAboveHalfCoverage) {
  auto view = chainModel(BackendDialType::Graph, false);
  auto& m = view.propagation;
  m.parameterCount = 3;
  m.dialInputs.push_back({2}); m.dialInputs.push_back({1});
  // Four of six program roots depend on (0,1), two on (2,1).
  m.dials[4].firstInput = m.dials[5].firstInput = 2;
  const auto packed = makeCudaWeightModel(view);
  CudaResponseUpdatePlan plan;
  plan.select(packed.parameterDialOffsets, {0., 0., 0.}, {1., 0., 0.},
              packed.surfaceGroups, packed.multiInputGroups, packed.parameterDialIndices);
  ASSERT_EQ(plan.ranges.size(), 1);
  EXPECT_EQ(plan.dialCount, 4);
  EXPECT_TRUE(plan.ranges[0].direct);
  EXPECT_EQ(plan.ranges[0].endDial - plan.ranges[0].firstDial, 4);
  plan.select(packed.parameterDialOffsets, {0., 0., 0.}, {1., 1., 1.},
              packed.surfaceGroups, packed.multiInputGroups, packed.parameterDialIndices);
  EXPECT_EQ(plan.dialCount, 6); // Never dispatch a root twice when several inputs change.
}

#ifdef GUNDAM_TEST_CUDA_DEVICE
TEST(CudaBackendPrograms, DeviceCompactChainsMatchOriginalInterpreter) {
  for( const auto* mode : {"cachedResponses", "recomputeResponses"} ){
    for( const auto leaf : {BackendDialType::Graph, BackendDialType::UniformSpline, BackendDialType::GeneralSpline} ){
      const auto view = chainModel(leaf, true);
      CudaBackend gpu;
      gpu.configure(ConfigReader(JsonType{{"propagationMode", mode}})); gpu.build(view);
      double graph[64]{}, formula[64]{};
      for( const double x : {-1000., -40., -1.2, -0., .3, 40., 1000.} ){
        PropagationInputs inputs; inputs.parameters.values = {x, .7};
        const auto token = gpu.requestPropagation(inputs);
        const auto& weights = gpu.getEventWeightsHostView(token);
        for( std::size_t i = 0; i < weights.size(); ++i ){
          const auto expected = evaluateComposite(view.propagation, view.propagation.dials[i], inputs.parameters.values.data(), graph, formula);
          EXPECT_NEAR(weights[i], expected, 3e-12 * std::max(1., std::abs(expected)));
        }
      }
    }
  }
}

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
