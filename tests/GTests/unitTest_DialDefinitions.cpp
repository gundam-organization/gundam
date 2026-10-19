#include <iostream>
#include <string>
#include <memory>
#include <cmath>

#include <gtest/gtest.h>

#include <TGraph.h>
#include <TSpline.h>
#include <TPad.h>

////////////////////////////////////////////////////////////////////////
// Test the DialDefinitions
#include "DialUtils.h"
#include "DialInputBuffer.h"
#include "CompactSpline.h"
#include "MonotonicSpline.h"
#include "GeneralSpline.h"
#include "UniformSpline.h"
#include "CalculateCompactSpline.h"
#include "DialCollection.h"
#include "Graph.h"
#include "Norm.h"
#include "RootSpline.h"
#include <future>

TEST(DialDefinitions,CompactSplineTruncated) {
    TGraph graph1;
    graph1.SetPoint(0,-3.0, 1.75);
    graph1.SetPoint(1,-2.0, 1.5);
    graph1.SetPoint(2,-1.0, 1.0);
    graph1.SetPoint(3, 0.0, 0.0);
    graph1.SetPoint(4, 1.0, 1.0);
    graph1.SetPoint(5, 2.0, 1.5);
    graph1.SetPoint(6, 3.0, 1.75);

    using DialPoints = std::vector<DialUtils::DialPoint>;
    DialPoints graph1Points = DialUtils::getPointList(&graph1);

    // Create an input buffer for the parameters
    DialInputBuffer buffer;
    buffer.getInputBuffer().resize(1);

    // Create the dial.
    CompactSpline dial;
    dial.buildDial(graph1Points);
    dial.setAllowExtrapolation(false);

    // Setup for the reference value calculation
    std::vector<double> data{dial.getDialData()};
    auto reference =
        [&](double r) -> double {
            return CalculateCompactSpline(r,0.0, 1000.0, data.data(),
                                          data.size()-2);
        };

    double range = graph1Points.back().x - graph1Points.front().x;
    for (double r = graph1Points.front().x;
         r <= graph1Points.back().x;
         r = r + range/100.0) {
        buffer.getInputBuffer()[0] = r;
        double d = dial.evalResponse(buffer);
        double c = reference(r);
        EXPECT_NEAR(d,c,1e-9) << "Must match CalculateCompactSpline";
    }

    // Check that the extrapolation is truncated at the minimum knot.
    {
        double minKnot = graph1Points.front().x;
        double truncVal = graph1Points.front().y;
        for (double r = minKnot - range; r < minKnot; r = r + range/1000.0) {
            buffer.getInputBuffer()[0] = r;
            double d = dial.evalResponse(buffer);
            EXPECT_NEAR(d,truncVal,1E-9)
                << "Extrapolation must truncate at " << r;
        }
    }

    // Check that extrapolation is truncated at the maximum knot
    {
        double maxKnot = graph1Points.back().x;
        double truncVal = graph1Points.back().y;
        for (double r = maxKnot; r < maxKnot + range; r = r + range/1000.0) {
            buffer.getInputBuffer()[0] = r;
            double d = dial.evalResponse(buffer);
            EXPECT_NEAR(d,truncVal,1E-9)
                << "Extrapolation must truncate at " << r;
        }
    }


}

TEST(DialDefinitions,CompactSplineExtrapolated) {
    TGraph graph1;
    graph1.SetPoint(0,-3.0, 1.75);
    graph1.SetPoint(1,-2.0, 1.5);
    graph1.SetPoint(2,-1.0, 1.0);
    graph1.SetPoint(3, 0.0, 0.0);
    graph1.SetPoint(4, 1.0, 1.0);
    graph1.SetPoint(5, 2.0, 1.5);
    graph1.SetPoint(6, 3.0, 1.75);

    using DialPoints = std::vector<DialUtils::DialPoint>;
    DialPoints graph1Points = DialUtils::getPointList(&graph1);

    // Create an input buffer for the parameters
    DialInputBuffer buffer;
    buffer.getInputBuffer().resize(1);

    // Create the dial.
    CompactSpline dial;
    dial.buildDial(graph1Points);
    dial.setAllowExtrapolation(true);

    // Setup for the reference value calculation
    std::vector<double> data{dial.getDialData()};
    auto reference =
        [&](double r) -> double {
            return CalculateCompactSpline(r,0.0, 1000.0, data.data(),
                                          data.size()-2);
        };

    double range = graph1Points.back().x - graph1Points.front().x;
    for (double r = graph1Points.front().x - range;
         r <= graph1Points.back().x + range;
         r = r + range/100.0) {
        buffer.getInputBuffer()[0] = r;
        double d = dial.evalResponse(buffer);
        double c = reference(r);
        EXPECT_NEAR(d,c,1e-9) << "Must match CalculateCompactSpline";
    }
}

TEST(DialDefinitions,MonotonicSpline) {
    TGraph graph1;
    graph1.SetPoint(0,-3.0, 1.75);
    graph1.SetPoint(1,-2.0, 1.5);
    graph1.SetPoint(2,-1.0, 1.0);
    graph1.SetPoint(3, 0.0, 0.0);
    graph1.SetPoint(4, 1.0, 1.0);
    graph1.SetPoint(5, 2.0, 1.5);
    graph1.SetPoint(6, 3.0, 1.75);

    using DialPoints = std::vector<DialUtils::DialPoint>;
    DialPoints graph1Points = DialUtils::getPointList(&graph1);

    // Create an input buffer for the parameters
    DialInputBuffer buffer;
    buffer.getInputBuffer().resize(1);

    // Create the reference dial.
    CompactSpline reference;
    reference.buildDial(graph1Points);
    reference.setAllowExtrapolation(false);

    // Create the dial to be tested.  The monotonic spline should match
    // Catmull Rom for these points.
    MonotonicSpline dial;
    dial.buildDial(graph1Points);
    dial.setAllowExtrapolation(false);

    double range = graph1Points.back().x - graph1Points.front().x;
    for (double r = graph1Points.front().x - range;
         r <= graph1Points.back().x + range;
         r = r + range/100.0) {
        buffer.getInputBuffer()[0] = r;
        double d = dial.evalResponse(buffer);
        double c = reference.evalResponse(buffer);
        EXPECT_NEAR(d,c,1e-9) << "Must match CompactSpline";
    }
}

TEST(DialDefinitions,CatmullRomGeneral) {
    TGraph graph1;
    graph1.SetPoint(0,-3.0, 1.75);
    graph1.SetPoint(1,-2.0, 1.5);
    graph1.SetPoint(2,-1.0, 1.0);
    graph1.SetPoint(3, 0.0, 0.0);
    graph1.SetPoint(4, 1.0, 1.0);
    graph1.SetPoint(5, 2.0, 1.5);
    graph1.SetPoint(6, 3.0, 1.75);

    using DialPoints = std::vector<DialUtils::DialPoint>;
    DialPoints graph1Points = DialUtils::getPointList(&graph1);

    // Create an input buffer for the parameters
    DialInputBuffer buffer;
    buffer.getInputBuffer().resize(1);

    // Create the dial.
    CompactSpline reference;
    reference.buildDial(graph1Points);
    reference.setAllowExtrapolation(false);

    DialPoints catmullRomPoints{graph1Points};
    DialUtils::fillCatmullRomSlopes(catmullRomPoints);
    GeneralSpline dial;
    dial.buildDial(catmullRomPoints);
    dial.setAllowExtrapolation(false);

    double range = graph1Points.back().x - graph1Points.front().x;
    for (double r = graph1Points.front().x - range;
         r <= graph1Points.back().x + range;
         r = r + range/100.0) {
        buffer.getInputBuffer()[0] = r;
        double d = dial.evalResponse(buffer);
        double c = reference.evalResponse(buffer);
        EXPECT_NEAR(d,c,1e-9) << "Must match CompactSpline";
    }
}

TEST(DialDefinitions,CatmullRomUniform) {
    TGraph graph1;
    graph1.SetPoint(0,-3.0, 1.75);
    graph1.SetPoint(1,-2.0, 1.5);
    graph1.SetPoint(2,-1.0, 1.0);
    graph1.SetPoint(3, 0.0, 0.0);
    graph1.SetPoint(4, 1.0, 1.0);
    graph1.SetPoint(5, 2.0, 1.5);
    graph1.SetPoint(6, 3.0, 1.75);

    using DialPoints = std::vector<DialUtils::DialPoint>;
    DialPoints graph1Points = DialUtils::getPointList(&graph1);

    // Create an input buffer for the parameters
    DialInputBuffer buffer;
    buffer.getInputBuffer().resize(1);

    // Create the dial.
    CompactSpline reference;
    reference.buildDial(graph1Points);
    reference.setAllowExtrapolation(false);

    DialPoints catmullRomPoints{graph1Points};
    DialUtils::fillCatmullRomSlopes(catmullRomPoints);
    UniformSpline dial;
    dial.buildDial(catmullRomPoints);
    dial.setAllowExtrapolation(false);

    double range = graph1Points.back().x - graph1Points.front().x;
    for (double r = graph1Points.front().x - range;
         r <= graph1Points.back().x + range;
         r = r + range/100.0) {
        buffer.getInputBuffer()[0] = r;
        double d = dial.evalResponse(buffer);
        double c = reference.evalResponse(buffer);
        EXPECT_NEAR(d,c,1e-9) << "Must match CalculateCompactSpline";
    }
}

// Local Variables:
// mode:c++
// c-basic-offset:4
// End:

// Loading optimizations must preserve the response and its domain.

namespace {
  DialCollection makeLoadingCollection(bool extrapolate_ = true, const std::string& options_ = ""){
    DialCollection collection(nullptr);
    collection.configure(ConfigReader(JsonType{
        {"dialType", "Spline"}, {"dialBranchData", "graph"},
        {"allowDialExtrapolation", extrapolate_}, {"options", options_}}));
    collection.getDialInputBufferList().resize(1);
    return collection;
  }

  template<typename T> void checkSharedSpline(){
    auto collection = makeLoadingCollection();
    std::vector<DialUtils::DialPoint> points{{-1, 2, -2}, {0, 1, 0}, {1, 2, 2}, {2, 5, 4}};
    auto build = [&](){
      auto dial = std::make_unique<T>();
      dial->buildDial(points);
      return dial;
    };
    const auto first = collection.storeEventDial(build());
    EXPECT_EQ(first, collection.storeEventDial(build()));
    // Equal knots with different Hermite slopes are different curves.
    points[1].slope = 0.125;
    auto changedSlope = build();
    if( changedSlope->getDialData() != collection.getDialInterfaceList()[first].getDialBaseRef()->getDialData() ){
      EXPECT_NE(first, collection.storeEventDial(std::move(changedSlope)));
    }
    points[1].slope = 0;
    auto extrapolated = build();
    extrapolated->setAllowExtrapolation(true);
    EXPECT_NE(first, collection.storeEventDial(std::move(extrapolated)));
    points[1].y += 0.125;
    EXPECT_NE(first, collection.storeEventDial(build()));
    collection.clear();
    EXPECT_EQ(0, collection.storeEventDial(build()));
  }
}

TEST(DialLoading, SharesLightweightSplines){
  checkSharedSpline<UniformSpline>();
  checkSharedSpline<GeneralSpline>();
  checkSharedSpline<CompactSpline>();
  checkSharedSpline<MonotonicSpline>();
}

TEST(DialLoading, ConcurrentDuplicates){
  auto collection = makeLoadingCollection();
  std::vector<std::future<size_t>> results;
  for( int i = 0; i < 4; ++i ){
    results.emplace_back(std::async(std::launch::async, [&](){
      size_t index = 0;
      for( int j = 0; j < 100; ++j ){
        auto dial = std::make_unique<Graph>();
        dial->buildDial(std::vector<DialUtils::DialPoint>{{0, 1, 0}, {1, 2, 0}});
        index = collection.storeEventDial(std::move(dial));
      }
      return index;
    }));
  }
  for( auto& result : results ){ EXPECT_EQ(0, result.get()); }
  EXPECT_EQ(1, collection.getDialInterfaceList().size());
}

TEST(DialLoading, SimplifiesAffineAndPreservesDomain){
  TGraph graph;
  for( int i = 0; i < 5; ++i ){ graph.SetPoint(i, i, 2 * i + 1); }
  for( bool extrapolate : {false, true} ){
    auto collection = makeLoadingCollection(extrapolate);
    auto dial = collection.makeDial(&graph);
    ASSERT_NE(nullptr, dynamic_cast<Graph*>(dial.get()));
    EXPECT_EQ(4, dial->getDialData().size());
    DialInputBuffer buffer;
    buffer.getInputBuffer().resize(1);
    for( double x : {-1., 0., 0.25, 2.5, 4., 5.} ){
      buffer.getInputBuffer()[0] = x;
      double effectiveX = extrapolate ? x : std::max(0., std::min(4., x));
      EXPECT_NEAR(2 * effectiveX + 1, dial->evalResponse(buffer), 1e-12);
    }
  }
  for( int i = 0; i < 5; ++i ){ graph.SetPoint(i, i, i); }
  auto extrapolated = makeLoadingCollection(true).makeDial(&graph);
  EXPECT_NE(nullptr, dynamic_cast<Norm*>(extrapolated.get()));
  auto bounded = makeLoadingCollection(false).makeDial(&graph);
  EXPECT_NE(nullptr, dynamic_cast<Graph*>(bounded.get()));
}

TEST(DialLoading, KeepsCurvedSlopesAndRootSplines){
  double x[]{0, 1, 2, 3};
  double y[]{0, 1, 2, 3};
  TSpline3 curved("curved", x, y, 4, "b1e1", 0., 0.);
  auto dial = makeLoadingCollection().makeDial(&curved);
  EXPECT_EQ(nullptr, dynamic_cast<Norm*>(dial.get()));
  EXPECT_EQ(nullptr, dynamic_cast<Graph*>(dial.get()));
  auto collection = makeLoadingCollection(true, "ROOT");
  auto first = collection.makeDial(&curved);
  ASSERT_NE(nullptr, dynamic_cast<RootSpline*>(first.get()));
  const auto index = collection.storeEventDial(std::move(first));
  EXPECT_NE(index, collection.storeEventDial(collection.makeDial(&curved)));
}

TEST(DialLoading, PreservesExactSplineBounds){
  struct BoundedSpline : UniformSpline {
    void setUpperBound(double upper_){ _splineBounds_.max = upper_; }
  };
  auto collection = makeLoadingCollection();
  auto first = std::make_unique<BoundedSpline>();
  first->buildDial(std::vector<DialUtils::DialPoint>{{0, 1, 0}, {1, 2, 2}, {2, 5, 4}});
  auto other = std::make_unique<BoundedSpline>(*first);
  other->setUpperBound(std::nextafter(2., 3.));
  ASSERT_EQ(first->getDialData(), other->getDialData());
  auto index = collection.storeEventDial(std::move(first));
  EXPECT_NE(index, collection.storeEventDial(std::move(other)));
}

TEST(DialLoading, SharesBinnedCurvesWithIndependentInputs){
  DialCollection collection(nullptr);
  collection.configure(ConfigReader(JsonType{{"dialType", "Norm"}}));
  auto& interfaces = collection.getDialInterfaceList();
  interfaces.clear();
  interfaces.resize(3);
  auto& bins = collection.getDialBinSet().getBinList();
  bins.resize(3);
  std::vector<DialInputBuffer> inputs(3);
  std::vector<DialResponseSupervisor> supervisors(3);
  for( size_t i = 0; i < interfaces.size(); ++i ){
    auto dial = std::make_unique<UniformSpline>();
    auto points = std::vector<DialUtils::DialPoint>{{0, 1, 0}, {1, 2, 2}, {2, 5, 4}};
    if( i == 2 ){ points[1].slope = .25; }
    dial->buildDial(points);
    interfaces[i].getDial().dialPtr = std::move(dial);
    inputs[i].getInputBuffer() = {double(i)};
    interfaces[i].setInputBufferRef(&inputs[i]);
    interfaces[i].setResponseSupervisorRef(&supervisors[i]);
    interfaces[i].setDialBinRef(&bins[i]);
  }
  collection.deduplicateBinnedDials();
  ASSERT_EQ(3, interfaces.size());
  EXPECT_EQ(interfaces[0].getDialBaseRef(), interfaces[1].getDialBaseRef());
  EXPECT_NE(interfaces[0].getDialBaseRef(), interfaces[2].getDialBaseRef());
  for( size_t i = 0; i < interfaces.size(); ++i ){
    EXPECT_EQ(&bins[i], interfaces[i].getDialBinRef());
    EXPECT_EQ(&inputs[i], interfaces[i].getInputBufferRef());
    EXPECT_EQ(&supervisors[i], interfaces[i].getResponseSupervisorRef());
    EXPECT_DOUBLE_EQ(1 + i * i, interfaces[i].evalResponse());
  }
  inputs[0].getInputBuffer()[0] = .5;
  EXPECT_DOUBLE_EQ(1.25, interfaces[0].evalResponse());
  EXPECT_DOUBLE_EQ(2., interfaces[1].evalResponse());

  // Copying keeps the historical clone semantics; moving keeps sharing alive.
  auto copy = interfaces[1];
  EXPECT_NE(copy.getDialBaseRef(), interfaces[1].getDialBaseRef());
  auto survivor = std::move(interfaces[1]);
  interfaces.clear();
  EXPECT_DOUBLE_EQ(2., survivor.evalResponse());
  EXPECT_DOUBLE_EQ(2., copy.evalResponse());
}

TEST(DialLoading, LeavesBinnedRootSplinesUnique){
  DialCollection collection(nullptr);
  collection.configure(ConfigReader(JsonType{{"dialType", "Norm"}}));
  collection.getDialBinSet().getBinList().resize(2);
  auto& interfaces = collection.getDialInterfaceList();
  interfaces.resize(2);
  for( auto& interface : interfaces ){
    auto dial = std::make_unique<RootSpline>();
    dial->buildDial(std::vector<DialUtils::DialPoint>{{0, 1, 0}, {1, 2, 2}, {2, 5, 4}});
    interface.getDial().dialPtr = std::move(dial);
  }
  collection.deduplicateBinnedDials();
  EXPECT_NE(interfaces[0].getDialBaseRef(), interfaces[1].getDialBaseRef());
}
