#include "CpuBackend.h"
#include "EngineLayout.h"
#include "LikelihoodInterface.h"
#include "CompositeDial.h"
#include "Bilinear.h"
#include "Bicubic.h"
#include "CompactSpline.h"
#include "UniformSpline.h"
#include "MonotonicSpline.h"
#include "GeneralSpline.h"
#include "Graph.h"
#include "Norm.h"
#include "Shift.h"
#include "RootFormula.h"
#include "CompiledLibDial.h"
#include "Tabulated.h"
#include "Kriged.h"
#include "ExternalWeightDispatcher.h"
#include "GundamGlobals.h"

#include <TH2D.h>
#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <cstdint>

namespace {
  std::uint64_t bits(double value_) {
    std::uint64_t result;
    std::memcpy(&result, &value_, sizeof(result));
    return result;
  }

  class CpuCompositeTest : public testing::TestWithParam<int> {};
}

TEST_P(CpuCompositeTest, MatchesOriginalBitsThroughNestedGraphAndCacheUpdates) {
  LikelihoodInterface likelihood;
  auto& propagator = likelihood.getModelPropagator();
  auto& samples = propagator.getSampleSet().getSampleList();
  samples.resize(1);
  samples[0].setIndex(0);
  samples[0].configure(ConfigReader(JsonType{
      {"name", "sample"}, {"binning", {{"binningDefinition", {
          {{"name", "X"}, {"edges", {0., 1.}}}}}}}}));

  std::vector<ParameterSet> parameters(1);
  for( int i = 0; i < 2; ++i ){ parameters[0].getParameterList().emplace_back(&parameters[0]); }
  std::array<DialInputBuffer, 4> buffers;
  const bool surface = GetParam() >= 7;
  const std::array<int, 4> arities{2, surface ? 2 : 1, 1, 2};
  for( std::size_t i = 0; i < buffers.size(); ++i ){
    buffers[i].setParSetRef(&parameters);
    for( int j = 0; j < arities[i]; ++j ){
      DialInputBuffer::ParameterReference ref;
      ref.parSetIndex = 0; ref.parIndex = j;
      // Distinct mirrors at root, nested boundary and leaf edges.
      ref.mirrorEdges = {-.7 - .1 * i, 2.3 + .2 * j};
      buffers[i].addParameterReference(ref);
    }
    buffers[i].initialise();
  }
  std::array<DialResponseSupervisor, 5> supervisors;
  for( std::size_t i = 0; i < supervisors.size(); ++i ){
    supervisors[i].setMinResponse(-.35 + .02 * i);
    supervisors[i].setMaxResponse(1.55 + .03 * i);
    supervisors[i].setSmoothingWidths(.13, .17);
  }
  TH2D histogram("cpuCompositeSurface", "", 5, -2.5, 2.5, 5, -2.5, 2.5);
  histogram.SetDirectory(nullptr);
  for( int x = 1; x <= 5; ++x ) for( int y = 1; y <= 5; ++y ){
    histogram.SetBinContent(x, y, .1 * x * x - .17 * y + .03 * x * y);
  }
  const std::vector<DialUtils::DialPoint> points{
      {-2., -.9, .25}, {-1., -.2, .4}, {0., .15, .2}, {1., 1.7, .6}, {2., .7, -.3}};
  DialBaseObject leaf(Norm{});
  switch( GetParam() ){
    case 0: break;
    case 1: { Shift dial; dial.setShiftValue(-.31); leaf = DialBaseObject(dial); break; }
    case 2: { Graph dial; dial.buildDial(points); leaf = DialBaseObject(dial); break; }
    case 3: { CompactSpline dial; dial.buildDial(points); leaf = DialBaseObject(dial); break; }
    case 4: { UniformSpline dial; dial.buildDial(points); leaf = DialBaseObject(dial); break; }
    case 5: { MonotonicSpline dial; dial.buildDial(points); leaf = DialBaseObject(dial); break; }
    case 6: { GeneralSpline dial; dial.buildDial(points); leaf = DialBaseObject(dial); break; }
    case 7: { Bilinear dial; dial.buildDial(histogram); leaf = DialBaseObject(dial); break; }
    case 8: { Bicubic dial; dial.buildDial(histogram); leaf = DialBaseObject(dial); break; }
  }
  Bilinear output;
  output.buildDial(histogram);
  const JsonType p{{"parameterName", "p"}}, q{{"parameterName", "q"}};
  const JsonType source{{"dialResponse", "source"}}, response{{"dialResponse", "leaf"}};
  const std::string type = surface ? "Surface" : (GetParam() == 0 ? "Norm" : GetParam() == 2 ? "Graph" : "Spline");
  CompositeDial graph(ConfigReader(JsonType{{"nodes", {
      {{"name", "output"}, {"dialType", "Surface"}, {"dialInputList", {response, source}}},
      {{"name", "leaf"}, {"dialType", type}, {"dialInputList", surface ? JsonType{source, q} : JsonType{source}}},
      {{"name", "source"}, {"dialType", "Norm"}, {"dialInputList", {p}}}}}, {"output", "output"}}), {"p", "q"});
  std::array<DialInterface, 4> interfaces;
  interfaces[0].setDial(DialBaseObject(output));
  interfaces[1].setDial(leaf);
  interfaces[2].setDial(DialBaseObject(Norm{}));
  for( int i = 0; i < 3; ++i ){
    interfaces[i].setInputBufferRef(&buffers[i]);
    interfaces[i].setResponseSupervisorRef(&supervisors[i]);
    graph.getNodes()[i].collectionIndex = i;
  }
  graph.bind([&](int index_){ return &interfaces[index_]; });
  interfaces[3].setDial(DialBaseObject(graph));
  interfaces[3].setInputBufferRef(&buffers[3]);
  interfaces[3].setResponseSupervisorRef(&supervisors[3]);
  CompositeDial nested(ConfigReader(JsonType{{"nodes", {
      {{"name", "nested"}, {"dialType", "Composite"}, {"dialInputList", {p, q}}}}}, {"output", "nested"}}), {"p", "q"});
  nested.getNodes()[0].collectionIndex = 3;
  nested.bind([&](int index_){ return &interfaces[index_]; });
  DialInterface root;
  root.setDial(DialBaseObject(nested));
  root.setInputBufferRef(&buffers[3]);
  root.setResponseSupervisorRef(&supervisors[4]);
  std::array<Event, 19> events;
  auto& cache = propagator.getEventDialCache().getCache();
  for( auto& event : events ){
    event.getIndices().sample = event.getIndices().bin = 0;
    event.getWeights().base = 1.23456789;
    cache.push_back({&event});
    cache.back().dialResponseCacheList.emplace_back(root);
    cache.back().dialResponseCacheList.emplace_back(interfaces[2]);
  }
  Backends::EngineLayout layout;
  const int previousThreads = GundamGlobals::getNbCpuThreads();
  for( int threads : {1, 3} ){
    GundamGlobals::setNumberOfThreads(threads);
    for( bool extrapolate : {false, true} ){
      interfaces[1].getDialBaseRef()->setAllowExtrapolation(extrapolate);
      // Bound graphs retain the interface's shared leaf object.
      layout.build(likelihood);
      ASSERT_EQ(layout.view.propagation.compositePrograms.size(), 1);
      ASSERT_GT(layout.view.propagation.compositeInstructions.size(), 3);
      for( auto traversal : {Backends::CpuCacheTraversal::Auto, Backends::CpuCacheTraversal::Indexed} ){
        Backends::CpuBackend backend;
        backend.build(layout.view);
        backend.setDiagnosticCacheTraversal(traversal);
        for( const auto& point : {std::array<double, 2>{.5, .25}, {.5, .25}, {-3., .25}, {-3., 1.8}, {0., 0.}, {1.7, -2.}, {.5, .25}} ){
          for( int i = 0; i < 2; ++i ){ parameters[0].getParameterList()[i].setParameterValue(point[i]); }
          for( auto& buffer : buffers ){ buffer.update(); }
          Backends::PropagationInputs inputs;
          for( const auto& binding : layout.bindings.parameters ){ inputs.parameters.values.push_back(binding.parameter->getParameterValue()); }
          double product = 1.;
          product *= root.evalResponse();
          product *= interfaces[2].evalResponse();
          const double expected = events[0].getWeights().base * product;
          const auto token = backend.requestPropagation(inputs);
          for( auto weight : backend.getEventWeightsHostView(token) ){ EXPECT_EQ(bits(weight), bits(expected)); }
          EXPECT_EQ(backend.getLastTimingSummary().cachedDialCount, 2);
          const auto repeated = backend.requestPropagation(inputs);
          EXPECT_EQ(backend.getLastTimingSummary().cacheCandidateCount, 0);
          EXPECT_EQ(bits(backend.getEventWeightsHostView(repeated)[0]), bits(expected));
        }
      }
    }
  }
  GundamGlobals::setNumberOfThreads(previousThreads);

  // Direct leaf programs also exercise one-input cache handling, exact graph
  // endpoints, stored spline bounds, and extrapolation without response clamps.
  for( auto& ref : buffers[1].getInputParameterIndicesList() ){ ref.mirrorEdges.minValue = std::nan(""); }
  supervisors[1].setMinResponse(std::nan(""));
  supervisors[1].setMaxResponse(std::nan(""));
  DialResponseSupervisor unlimited;
  CompositeDial direct(ConfigReader(JsonType{{"nodes", {
      {{"name", "leaf"}, {"dialType", type}, {"dialInputList", surface ? JsonType{p, q} : JsonType{p}}}}},
      {"output", "leaf"}}), surface ? std::vector<std::string>{"p", "q"} : std::vector<std::string>{"p"});
  direct.getNodes()[0].collectionIndex = 1;
  direct.bind([&](int index_){ return &interfaces[index_]; });
  root.setDial(DialBaseObject(direct));
  root.setInputBufferRef(&buffers[1]);
  root.setResponseSupervisorRef(&unlimited);
  for( auto& entry : cache ){ entry.dialResponseCacheList.pop_back(); }
  for( bool extrapolate : {false, true} ){
    interfaces[1].getDialBaseRef()->setAllowExtrapolation(extrapolate);
    layout.build(likelihood);
    Backends::CpuBackend backend;
    backend.build(layout.view);
    for( double x : {-4., -2., std::nextafter(-2., -3.), -.0, .13, .9, 2., std::nextafter(2., 3.), 4., .13} ){
      parameters[0].getParameterList()[0].setParameterValue(x);
      parameters[0].getParameterList()[1].setParameterValue(.37);
      buffers[1].update();
      Backends::PropagationInputs inputs;
      for( const auto& binding : layout.bindings.parameters ){ inputs.parameters.values.push_back(binding.parameter->getParameterValue()); }
      const double expected = events[0].getWeights().base * root.evalResponse();
      const auto token = backend.requestPropagation(inputs);
      for( auto weight : backend.getEventWeightsHostView(token) ){ EXPECT_EQ(bits(weight), bits(expected)) << "x=" << x; }
    }
  }
  // Unsupported leaves fail during compilation, before any response evaluation.
  auto* bound = dynamic_cast<CompositeDial*>(root.getDialBaseRef());
  ASSERT_NE(bound, nullptr);
  std::vector<std::shared_ptr<const DialBase>> unsupported{
      std::make_shared<RootFormula>(), std::make_shared<CompiledLibDial>(),
      std::make_shared<Tabulated>(nullptr, 0, .5),
      std::make_shared<Kriged>(nullptr, 0, std::vector<int>{}, std::vector<double>{}),
      std::make_shared<ExternalWeightDispatcher>(nullptr, 0)};
  for( const auto& dial : unsupported ){
    bound->getNodes()[0].dial = dial;
    try{
      layout.build(likelihood);
      FAIL() << "Accepted unsupported composite leaf " << dial->getDialTypeName();
    }
    catch( const std::runtime_error& error ){
      EXPECT_NE(std::string(error.what()).find("Composite/leaf"), std::string::npos);
    }
  }

}

INSTANTIATE_TEST_SUITE_P(SimpleLeaves, CpuCompositeTest, testing::Range(0, 9));

TEST(CpuCompositeValidation, RejectsForwardScratchReferences) {
  Backends::EngineView view;
  auto& model = view.propagation;
  model.parameterCount = 1;
  model.dialInputs.emplace_back(0);
  model.compositePrograms.push_back({0, 1, 1, 1});
  Backends::CompositeInstruction instruction;
  instruction.dial.inputCount = 1;
  model.compositeInstructions.push_back(instruction);
  model.compositeInputs.emplace_back(1); // Only the argument at slot zero exists.
  Backends::BackendDialDescriptor root;
  root.type = Backends::BackendDialType::Composite;
  root.inputCount = 1;
  model.dials.push_back(root);
  Backends::CpuBackend backend;
  EXPECT_THROW(backend.build(view), std::runtime_error);
  model.compositeInputs[0].parameterIndex = 0;
  EXPECT_NO_THROW(backend.build(view));
}
