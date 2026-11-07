#include "CpuBackend.h"
#include "EngineLayout.h"
#include "LikelihoodInterface.h"
#include "Bilinear.h"
#include "Graph.h"
#include "Norm.h"
#include "Shift.h"
#include "Tabulated.h"
#include "Kriged.h"

#include <TH2D.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

TEST(EngineLayout, SharesInputSequencesAndPreservesWeightsAfterRebuild) {
  LikelihoodInterface likelihood;
  auto& model = likelihood.getModelPropagator();
  auto& samples = model.getSampleSet().getSampleList();
  samples.resize(1);
  samples[0].setIndex(0);
  samples[0].configure(ConfigReader(JsonType{
      {"name", "sample"}, {"binning", {{"binningDefinition", {
          {{"name", "X"}, {"edges", {0., 1.}}}}}}}}));

  std::vector<ParameterSet> parameters(1);
  for( int i = 0; i < 2; ++i ){ parameters[0].getParameterList().emplace_back(&parameters[0]); }
  std::array<DialInputBuffer, 5> buffers;
  const std::vector<std::vector<int>> orders{{0}, {0}, {0, 1}, {1, 0}, {}};
  for( std::size_t i = 0; i < buffers.size(); ++i ){
    buffers[i].setParSetRef(&parameters);
    for( auto index : orders[i] ){
      DialInputBuffer::ParameterReference reference;
      reference.parSetIndex = 0;
      reference.parIndex = index;
      if( i == 1 or i == 3 ){ reference.mirrorEdges = {0., 2.}; }
      buffers[i].addParameterReference(reference);
    }
    buffers[i].initialise();
  }

  Graph graph;
  graph.buildDial(std::vector<DialUtils::DialPoint>{{0., 1., 0.}, {2., 1.5, 0.}});
  TH2D histogram("layoutSurface", "", 4, -1., 3., 4, -1., 3.);
  histogram.SetDirectory(nullptr);
  for( int x = 1; x <= 4; ++x ){
    for( int y = 1; y <= 4; ++y ){
      histogram.SetBinContent(x, y, 1. + .1 * histogram.GetXaxis()->GetBinCenter(x)
                                     + .2 * histogram.GetYaxis()->GetBinCenter(y));
    }
  }
  Bilinear surface;
  surface.buildDial(histogram);
  Shift shift;
  shift.setShiftValue(1.25);
  std::array<DialInterface, 8> interfaces;
  interfaces[0].setDial(DialBaseObject(Norm{}));
  interfaces[1].setDial(DialBaseObject(graph));
  interfaces[2].setDial(DialBaseObject(Norm{}));
  for( int i = 3; i <= 5; ++i ){ interfaces[i].setDial(DialBaseObject(surface)); }
  for( int i = 6; i <= 7; ++i ){ interfaces[i].setDial(DialBaseObject(shift)); }
  const std::array<int, 8> bufferIndices{0, 0, 1, 2, 2, 3, 4, 4};
  for( std::size_t i = 0; i < interfaces.size(); ++i ){
    interfaces[i].setInputBufferRef(&buffers[bufferIndices[i]]);
  }
  std::array<Event, 3> events;
  auto& cache = model.getEventDialCache().getCache();
  for( std::size_t i = 0; i < events.size(); ++i ){
    events[i].getIndices().sample = events[i].getIndices().bin = 0;
    events[i].getWeights().base = 1. + .1 * i;
    cache.push_back({&events[i]});
    if( i == 2 ){ continue; } // Also cover an event with no dials.
    for( std::size_t j = 0; j < interfaces.size(); ++j ){
      cache.back().dialResponseCacheList.emplace_back(interfaces[i == 0 ? j : interfaces.size() - 1 - j]);
    }
  }
  cache.push_back({nullptr});
  // A constant dial may have no input buffer at all. The native response
  // cache is not evaluated here; the layout must accept this empty sequence.
  interfaces[7].setInputBufferRef(nullptr);

  Backends::EngineLayout layout;
  for( int rebuild = 0; rebuild < 2; ++rebuild ){
    layout.build(likelihood);
    const auto& propagation = layout.view.propagation;
    ASSERT_EQ(propagation.dials.size(), 8);
    ASSERT_EQ(propagation.events.size(), 3);
    ASSERT_EQ(propagation.eventDialIndices.size(), 16);
    ASSERT_EQ(propagation.dialInputs.size(), 6);
    EXPECT_EQ(propagation.dials[0].firstInput, propagation.dials[1].firstInput);
    EXPECT_NE(propagation.dials[0].firstInput, propagation.dials[2].firstInput);
    EXPECT_EQ(propagation.dials[3].firstInput, propagation.dials[4].firstInput);
    EXPECT_NE(propagation.dials[3].firstInput, propagation.dials[5].firstInput);
    Backends::CpuBackend backend;
    backend.build(layout.view);
    for( const auto& point : {std::array<double, 2>{.5, 1.}, {2.5, .25}, {.5, 1.}, {.5, 1.}} ){
      for( int i = 0; i < 2; ++i ){ parameters[0].getParameterList()[i].setParameterValue(point[i]); }
      for( auto& buffer : buffers ){ buffer.update(); }
      Backends::PropagationInputs inputs;
      for( const auto& binding : layout.bindings.parameters ){
        inputs.parameters.values.push_back(binding.parameter->getParameterValue());
      }
      const auto token = backend.requestPropagation(inputs);
      backend.materialize(token, Backends::OutputRequest::EventWeights);
      const auto& weights = backend.getEventWeightsHostView(token);
      ASSERT_EQ(weights.size(), events.size());
      for( std::size_t i = 0; i < events.size(); ++i ){
        double expected = events[i].getWeights().base;
        for( const auto& response : cache[i].dialResponseCacheList ){
          const auto* interface = response.dialInterface;
          const auto* buffer = interface->getInputBufferRef();
          expected *= interface->getDialBaseRef()->evalResponse(buffer == nullptr ? buffers[4] : *buffer);
        }
        EXPECT_NEAR(weights[i], expected, std::abs(expected) * 1e-12);
      }
    }
  }
}

TEST(EngineLayout, TableDialsShareSourcesWithoutRequestingScalarPayloads) {
  LikelihoodInterface likelihood;
  auto& model = likelihood.getModelPropagator();
  auto& samples = model.getSampleSet().getSampleList();
  samples.resize(1);
  samples[0].setIndex(0);
  samples[0].configure(ConfigReader(JsonType{
      {"name", "sample"}, {"binning", {{"binningDefinition", {
          {{"name", "X"}, {"edges", {0., 1.}}}}}}}}));
  std::vector<double> first{1., 2., 3.}, second{4., 5.};
  DialInputBuffer buffer;
  std::array<DialInterface, 3> interfaces;
  interfaces[0].setDial(DialBaseObject(Tabulated(&first, 1, .25)));
  interfaces[1].setDial(DialBaseObject(Kriged(&first, 3, {2, 0, 2}, {-.25, .5, .75})));
  interfaces[2].setDial(DialBaseObject(Kriged(&second, 1, {1}, {1.})));
  Event event;
  event.getIndices().sample = event.getIndices().bin = 0;
  auto& cache = model.getEventDialCache().getCache();
  cache.push_back({&event});
  for( auto& interface : interfaces ){
    interface.setInputBufferRef(&buffer);
    cache.back().dialResponseCacheList.emplace_back(interface);
  }
  cache.back().dialResponseCacheList.emplace_back(interfaces[0]);
  Backends::EngineLayout layout;
  for( int rebuild = 0; rebuild < 2; ++rebuild ){
    layout.build(likelihood);
    const auto& propagation = layout.view.propagation;
    EXPECT_TRUE(propagation.dialPayloads.empty());
    ASSERT_EQ(propagation.dials.size(), 3);
    ASSERT_EQ(propagation.externalWeightBlocks.size(), 2);
    ASSERT_EQ(layout.bindings.externalWeights.size(), 2);
    EXPECT_EQ(layout.bindings.externalWeights[0].table, &first);
    EXPECT_EQ(layout.bindings.externalWeights[1].table, &second);
    EXPECT_EQ(propagation.dials[0].externalBlockIndex, propagation.dials[1].externalBlockIndex);
    EXPECT_NE(propagation.dials[0].externalBlockIndex, propagation.dials[2].externalBlockIndex);
    EXPECT_EQ(propagation.tableTerms.size(), 6);
    EXPECT_EQ(propagation.eventDialIndices, (std::vector<std::uint32_t>{0, 1, 2, 0}));
  }
}

TEST(EngineLayout, ExportsLiteLikelihoodKindsFromConfiguration) {
  for( const char* type : {"BarlowBeestonLiteGaussian", "BarlowBeestonLitePoisson"} ){
    LikelihoodInterface likelihood;
    likelihood.configure(ConfigReader(JsonType{{"jointProbabilityConfig", {{"type", type}}}}));
    auto& samples = likelihood.getModelPropagator().getSampleSet().getSampleList();
    samples.resize(1);
    samples[0].setIndex(0);
    samples[0].configure(ConfigReader(JsonType{{"name", "model"}, {"binning", {{"binningDefinition", {
        {{"name", "X"}, {"edges", {0., 1.}}}}}}}}));
    Sample data;
    data.configure(ConfigReader(JsonType{{"name", "data"}, {"binning", {{"binningDefinition", {
        {{"name", "X"}, {"edges", {0., 1.}}}}}}}}));
    data.getHistogram().getBinContentList()[0].sumWeights = 12;
    likelihood.getSamplePairList().push_back({&samples[0], &data});
    Backends::EngineLayout layout;
    layout.build(likelihood);
    ASSERT_EQ(layout.view.likelihood.samples.size(), 1);
    const auto& sample = layout.view.likelihood.samples[0];
    const auto expected = std::string(type) == "BarlowBeestonLiteGaussian"
        ? Backends::LikelihoodKind::BarlowBeestonLiteGaussian : Backends::LikelihoodKind::BarlowBeestonLitePoisson;
    EXPECT_EQ(sample.portableOptions.kind, expected);
    EXPECT_EQ(sample.portableOptions.useNominalVariance, 0);
    EXPECT_EQ(sample.portableOptions.throwIfInfLlh, 0);
    EXPECT_TRUE(sample.nominalVariances.empty());
    EXPECT_DOUBLE_EQ(sample.evalBin(12, 10, 3, 0), likelihood.getJointProbabilityPtr()->eval(12, 10, 3, 0));
  }
}
