#include "CompositeDial.h"
#include "EventFormula.h"
#include "RootFormula.h"
#include "Graph.h"
#include "Norm.h"
#include "Event.h"

#include <gtest/gtest.h>

#include <array>
#include <future>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
  void checkCompositeGradient(const CompositeDial& dial_, std::vector<double> point_) {
    DialInputBuffer input;
    input.setInputValues(point_);
    const auto gradient = dial_.evalGradients(input);
    ASSERT_EQ(point_.size(), gradient.size());
    for( size_t i = 0; i < point_.size(); ++i ){
      auto up = point_, down = point_;
      up[i] += 1e-5;
      down[i] -= 1e-5;
      DialInputBuffer upper, lower;
      upper.setInputValues(up);
      lower.setInputValues(down);
      EXPECT_NEAR((dial_.evalResponse(upper) - dial_.evalResponse(lower)) / 2e-5,
                  gradient[i], 1e-7) << "input=" << i;
      EXPECT_DOUBLE_EQ(gradient[i], dial_.evalGradient(input, int(i)));
    }
  }
}

TEST(CompositeGradient, BranchesRepeatedParametersMirrorsClampsAndNestedGraphs) {
#ifndef R__HAS_CLAD
  GTEST_SKIP() << "ROOT was built without Clad";
#endif
  const JsonType p{{"parameterName", "p"}}, q{{"parameterName", "q"}};
  const JsonType source{{"dialResponse", "source"}}, left{{"dialResponse", "left"}};
  CompositeDial dial(ConfigReader(JsonType{{"nodes", {
      {{"name", "source"}, {"dialType", "Graph"}, {"dialInputList", {p}}},
      {{"name", "left"}, {"dialType", "Formula"}, {"dialInputList", {
          {{"name", "a"}, {"dialResponse", "source"}}, {{"name", "b"}, {"parameterName", "q"}}}}},
      {{"name", "output"}, {"dialType", "Formula"}, {"dialInputList", {
          {{"name", "a"}, {"dialResponse", "left"}},
          {{"name", "b"}, {"dialResponse", "source"}},
          {{"name", "c"}, {"parameterName", "p"}}}}}}}, {"output", "output"}}), {"p", "q"});
  Graph curve;
  curve.buildDial(std::vector<DialUtils::DialPoint>{{-2., -2., 0.}, {0., 0., 0.}, {2., 2., 0.}});
  curve.setAllowExtrapolation(true);
  RootFormula square, output;
  square.setFormulaStr("[a]*[a]+[b]");
  square.setInputNames({"a", "b"});
  output.setFormulaStr("[a]*[b]+[c]*[c]");
  output.setInputNames({"a", "b", "c"});
  std::array<DialInterface, 3> interfaces;
  std::array<DialInputBuffer, 3> buffers;
  std::array<DialResponseSupervisor, 3> supervisors;
  std::vector<ParameterSet> parameters(1);
  for( int i = 0; i < 3; ++i ){ parameters[0].getParameterList().emplace_back(&parameters[0]); }
  interfaces[0].setDial(DialBaseObject(curve));
  interfaces[1].setDial(DialBaseObject(square));
  interfaces[2].setDial(DialBaseObject(output));
  for( int i = 0; i < 3; ++i ){
    buffers[i].setParSetRef(&parameters);
    for( int j = 0; j <= i; ++j ){
      DialInputBuffer::ParameterReference ref;
      ref.parSetIndex = 0; ref.parIndex = j;
      // Distinct folds at intermediate and direct parameter edges.
      ref.mirrorEdges = {-.8, 2.1};
      buffers[i].addParameterReference(ref);
    }
    buffers[i].initialise();
    interfaces[i].setInputBufferRef(&buffers[i]);
    interfaces[i].setResponseSupervisorRef(&supervisors[i]);
    dial.getNodes()[i].collectionIndex = i;
  }
  for( bool clamp : {false, true} ){
    if( clamp ){ supervisors[1].setMaxResponse(.3); }
    dial.bind([&](int index_){ return &interfaces[index_]; });
    ASSERT_TRUE(dial.hasAnalyticGradient());
    CompositeDial nested(ConfigReader(JsonType{{"nodes", {
        {{"name", "child"}, {"dialType", "Composite"}, {"dialInputList", {p, q}}}}},
        {"output", "child"}}), {"p", "q"});
    DialInterface child;
    DialInputBuffer childInputs;
    childInputs.setParSetRef(&parameters);
    for( int j = 0; j < 2; ++j ){
      DialInputBuffer::ParameterReference ref;
      ref.parSetIndex = 0; ref.parIndex = j;
      childInputs.addParameterReference(ref);
    }
    childInputs.initialise();
    DialResponseSupervisor childSupervisor;
    child.setDial(DialBaseObject(dial));
    child.setInputBufferRef(&childInputs);
    child.setResponseSupervisorRef(&childSupervisor);
    nested.getNodes()[0].collectionIndex = 0;
    nested.bind([&](int){ return &child; });
    ASSERT_TRUE(nested.hasAnalyticGradient());
    std::vector<std::future<void>> jobs;
    for( const auto& point : {std::vector<double>{.2, .4}, {0., .4}, {3.2, -1.1}, {-2.3, .8}} ){
      jobs.emplace_back(std::async(std::launch::async, [&, point]{
        checkCompositeGradient(dial, point);
        checkCompositeGradient(nested, point);
      }));
    }
    for( auto& job : jobs ){ job.get(); }
  }
}

TEST(CompositeGradient, EventInputsStayConstantAndUnsupportedNodesAreNamed) {
#ifndef R__HAS_CLAD
  GTEST_SKIP() << "ROOT was built without Clad";
#endif
  RootFormula formula;
  formula.setFormulaStr("[energy]*[value]*[value]");
  formula.setInputNames({"energy", "value"});
  auto definition = std::make_shared<EventFormula::Definition>();
  definition->eventVariables = {"energy"};
  definition->inputs = {{-1, 0}, {0, -1}};
  const std::vector<std::string> names{"energy"};
  Event event;
  event.getVariables().setVarNameList(names);
  event.getVariables().fetchVariable("energy").set(2.5);
  EventFormula bound(&formula, definition, event, {});
  const JsonType config{{"nodes", {{{"name", "eventNode"}, {"dialType", "Formula"},
      {"dialInputList", {{{"name", "energy"}, {"eventVariable", "energy"}},
                         {{"name", "value"}, {"parameterName", "p"}}}}}}}, {"output", "eventNode"}};
  CompositeDial dial(ConfigReader(config), {"p"});
  DialInputBuffer buffer;
  std::vector<ParameterSet> parameters(1);
  parameters[0].getParameterList().emplace_back(&parameters[0]);
  buffer.setParSetRef(&parameters);
  DialInputBuffer::ParameterReference ref;
  ref.parSetIndex = ref.parIndex = 0;
  buffer.addParameterReference(ref);
  buffer.initialise();
  DialResponseSupervisor supervisor;
  DialInterface interface;
  interface.setDial(DialBaseObject(bound));
  interface.setInputBufferRef(&buffer);
  interface.setResponseSupervisorRef(&supervisor);
  dial.getNodes()[0].collectionIndex = 0;
  dial.bind([&](int){ return &interface; });
  ASSERT_TRUE(dial.hasAnalyticGradient());
  checkCompositeGradient(dial, {.4});
  DialInputBuffer input;
  input.setInputValues({.4});
  EXPECT_NEAR(2., dial.evalGradients(input)[0], 1e-12);

  // Value evaluation remains available when a sub-formula cannot be derived.
  formula.setFormulaStr("2+TMath::BesselJ0([value])+[energy]");
  formula.setInputNames({"energy", "value"});
  try{
    static_cast<void>(dial.hasAnalyticGradient());
    FAIL() << "Unsupported sub-dial was accepted";
  }
  catch( const std::exception& error ){
    EXPECT_NE(std::string::npos, std::string(error.what()).find("eventNode"));
  }
}
