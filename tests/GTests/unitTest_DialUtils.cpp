#include <iostream>
#include <string>
#include <memory>
#include <cmath>

#include <gtest/gtest.h>

#include <TGraph.h>
#include <TSpline.h>
#include <TPad.h>

////////////////////////////////////////////////////////////////////////
// Test the DialUtils
#include "DialUtils.h"

TEST(DialUtils,GetPointList) {
    // Test the getPointList functions
    TGraph graph1;
    graph1.SetPoint(0,-1.0, 1.0);
    graph1.SetPoint(1, 0.0, 0.0);
    graph1.SetPoint(2, 1.0, 1.0);

    using DialPoints = std::vector<DialUtils::DialPoint>;
    DialPoints graph1Points = DialUtils::getPointList(&graph1);

    // Check that the graph got copied.
    EXPECT_GT(graph1Points.size(), 0)
        << "Must have points in graph";
    for (std::size_t i = 0; i<graph1.GetN(); ++i) {
        EXPECT_NEAR(graph1.GetX()[i], graph1Points[i].x, 1e-6)
            << "X Value should match for index " << i;
        EXPECT_NEAR(graph1.GetY()[i], graph1Points[i].y, 1e-6)
            << "Y Value should match for index " << i;
    }

    // Check that a spline get's copied.
    TSpline3 spline1("spline1",&graph1);
    DialPoints spline1Points = DialUtils::getPointList(&spline1);
    EXPECT_GT(spline1Points.size(), 0)
        << "Must have points in spline";
    EXPECT_EQ(graph1Points.size(), spline1Points.size())
        << "Spline and graph must have same number of points";

    for (std::size_t i = 0; i<graph1Points.size(); ++i) {
        EXPECT_NEAR(graph1Points[i].x, spline1Points[i].x,1E-6)
            << "X values must be near to each other";
        EXPECT_NEAR(graph1Points[i].y, spline1Points[i].y,1E-6)
            << "Y values must be near to each other";
        EXPECT_NEAR(graph1Points[i].slope, spline1Points[i].slope,1E-6)
            << "Slope values must be near to each other";
    }

    TObject* object1 = &graph1;
    DialPoints object1Points = DialUtils::getPointList(object1);

    EXPECT_EQ(graph1Points.size(), object1Points.size())
        << "Must have the same number of points";

    for (std::size_t i = 0; i<graph1Points.size(); ++i) {
        EXPECT_NEAR(graph1Points[i].x, object1Points[i].x,1E-6)
            << "X values must be near to each other";
        EXPECT_NEAR(graph1Points[i].y, object1Points[i].y,1E-6)
            << "Y values must be near to each other";
        EXPECT_NEAR(graph1Points[i].slope, object1Points[i].slope,1E-6)
            << "Slope values must be near to each other";
    }

    DialPoints slope1Points = DialUtils::getPointListNoSlope(&graph1);

    EXPECT_EQ(graph1Points.size(), slope1Points.size())
        << "Must have the same number of points";

    for (std::size_t i = 0; i<graph1Points.size(); ++i) {
        EXPECT_NEAR(graph1Points[i].x, slope1Points[i].x,1E-6)
            << "X values must be near to each other";
        EXPECT_NEAR(graph1Points[i].y, slope1Points[i].y,1E-6)
            << "Y values must be near to each other";
        EXPECT_NEAR(0.0, slope1Points[i].slope,1E-6)
            << "Slope values must be near to each other";
    }
}

TEST(DialUtils,SlopeCalculations) {
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

    DialPoints akima1Points{graph1Points};
    DialUtils::fillAkimaSlopes(akima1Points);
    for (std::size_t i = 0; i<graph1Points.size(); ++i) {
        EXPECT_NEAR(graph1Points[i].x, akima1Points[i].x,1E-6)
            << "X values must be near to each other";
        EXPECT_NEAR(graph1Points[i].y, akima1Points[i].y,1E-6)
            << "Y values must be near to each other";
    }

    DialPoints catmullRom1Points{graph1Points};
    DialUtils::fillCatmullRomSlopes(catmullRom1Points);
    for (std::size_t i = 0; i<graph1Points.size(); ++i) {
        EXPECT_NEAR(graph1Points[i].x, catmullRom1Points[i].x,1E-6)
            << "X values must be near to each other";
        EXPECT_NEAR(graph1Points[i].y, catmullRom1Points[i].y,1E-6)
            << "Y values must be near to each other";
    }

}

// Local Variables:
// mode:c++
// c-basic-offset:4
// End:
