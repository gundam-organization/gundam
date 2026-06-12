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
