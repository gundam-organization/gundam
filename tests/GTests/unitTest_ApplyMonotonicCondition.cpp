#include <iostream>
#include <string>
#include <memory>
#include <cmath>

#include <gtest/gtest.h>

#include <TSpline.h>
#include <TGraph.h>
#include <TPad.h>

////////////////////////////////////////////////////////////////////////
// Test the CalculateGeneralSpline routine on the CPU.

#include "CalculateGeneralSpline.h"

namespace unitTest {
    // Bring in a robust test version of the Fritsh-Carlson monotonic
    // conditions.  This is a direct insertion of an include file, and it's
    // don't this way so that the test has it's VERY OWN copy.

// #include "FritshCarlsonMonotonicCondition.h"
#ifndef FritshCarlsonMonotonicCondition_h_seen
#define FritshCarlsonMonotonicCondition_h_seen
#include <limits>
    namespace FritshCarlson {
        // Apply the Fritsh-Carlson monotonic condition to the slopes.  This
        // adjusts the slopes (when necessary).
        //
        // F.N.Fritsch, and R.E.Carlson, "Monotone Piecewise Cubic
        // Interpolation" SIAM Journal on Numerical Analysis, Vol. 17, Iss. 2
        // (1980) doi:10.1137/0717021
        //
        // This is a reimplementation based on the paper, with help from the
        // wikipedia article on Monotonic Cubic Interpolation which clarifies
        // some of the original notation. The aim is to reproduce the
        // mathematics as clearly as possible, and to ignore any optimization.
        void MonotonicCondition(
            const std::vector<double>& xPoint,
            const std::vector<double>& yPoint,
            std::vector<double>& slope) {

            // 1) Find the secant line slopes.
            std::vector<double> d(xPoint.size());
            for (std::size_t k = 0; k<xPoint.size()-1; ++k) {
                d[k] = (yPoint[k+1]-yPoint[k])/(xPoint[k+1]-xPoint[k]);
            }

            // 2) Initialize m with the provisional tangent for the interior
            // knots of the cubic.  The tangents provide the limit on the
            // slope.
            std::vector<double> m(xPoint.size());
            for (std::size_t k=1; k<xPoint.size()-1; ++k) {
                m[k] = (d[k-1]+d[k])/2.0;
            }
            m[0] = d[0];
            m[xPoint.size()-1] = d[xPoint.size()-2];

            // 2.A) Set tangents to flat if there is a cusp which is true when
            // d[k] and d[k-1] have opposite signs.
            for (std::size_t k=1; k<xPoint.size(); ++k) {
                if (d[k-1]*d[k] < 0.0) m[k] = 0.0;
            }

            // 3) When the successive points are equal, set the tangents are
            // zero.  This is when d is zero.
            for (std::size_t k = 0; k<xPoint.size()-1; ++k) {
                if (std::abs(d[k]) < std::numeric_limits<double>::epsilon()) {
                    m[k] = 0.0;
                    m[k+1] = 0.0;
                }
            }

            // 4.A) Calculate alpha
            std::vector<double> alpha(xPoint.size());
            for (std::size_t k = 0; k < xPoint.size(); ++k) {
                alpha[k] = 0.0;
                if (std::abs(d[k])
                    > std::numeric_limits<double>::epsilon()) {
                    alpha[k] = m[k]/d[k];
                }
            }

            // 4.B) Calculate beta
            std::vector<double> beta(xPoint.size());
            for (std::size_t k = 0; k < xPoint.size(); ++k) {
                beta[k] = 0.0;
                if (std::abs(d[k]) > std::numeric_limits<double>::epsilon()) {
                    if (k < xPoint.size()-1) beta[k] = m[k+1]/d[k];
                }
            }

            // 4.C) When alpha is zero the slope is zero
            for (std::size_t k = 0; k < xPoint.size(); ++k) {
                if (alpha[k] < 0.0) m[k] = 0.0;
            }

            // 4.D) When beta is zero the next slope is zero.
            for (std::size_t k = 0; k < xPoint.size()-1; ++k) {
                if (beta[k] < 0.0) m[k+1] = 0.0;
            }

            // 5) Prevent overshoot. This applies the criteria for Set 1 in
            // the Fritzch-Carlson paper (See Figure 2) to adjust the bounds
            // on the tangent at each knot.
            double bound = 3.0;
            for (std::size_t k = 0; k < xPoint.size(); ++k) {
                if (alpha[k] > bound) m[k] = bound*d[k];
            }
            for (std::size_t k = 0; k < xPoint.size()-1; ++k) {
                if (beta[k] > bound) m[k+1] = bound*d[k];
            }

            // 6) Fix the slopes based on the tangents.
            for (std::size_t k = 0; k < xPoint.size(); ++k) {
                double s = slope[k];
                // If the slope is positive, it must be less than or equal to
                // the bounding tangent.
                if (slope[k] > 0.0 and slope[k] > m[k]) s = m[k];
                // If the slop is negative, it must be greater than or equal
                // to the bounding tangent.
                if (slope[k] < 0.0 and slope[k] < m[k]) s = m[k];
                // If the slope changed, copy it back into the output.
                if (std::abs(s-slope[k])
                    > std::numeric_limits<double>::epsilon()) {
                    slope[k] = s;
                }
            }
        }
    }
}

// An MIT Style License

// Copyright (c) 2026 Clark McGrew

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Local Variables:
// mode:c++
// c-basic-offset:4
// End:
#endif

namespace {
    // A stupidly simple "PRNG" that is very predictable.
    double myRand(unsigned int& seed) {
        const int m = (2<<15)-1;
        const int a = 1341;
        const int c = 4321;
        seed = (a*seed + c) % m;
        return 1.0*seed/m;
    }

    // There are more clever ways to do this, but do a "BRUTE FORCE" check for
    // cusps with an emphasis on making it super easy to understand.
    int ExpectedFlips(std::vector<double> xValues,
                      std::vector<double> yValues,
                      int points) {
        int count = 0;
        if (yValues.size() < points) points = yValues.size();
        for (std::size_t i = 1; i < points-1; ++i) {
            if ((yValues[i-1] > yValues[i])
                && (yValues[i+1] > yValues[i])) {
                // If the value before and after this element are greater than
                // this one, then this one is a cusp.
                count = count + 1;
            }
            else if ((yValues[i-1] < yValues[i])
                && (yValues[i+1] < yValues[i])) {
                // If the value before and after this element are less than
                // this one, then this one is a cusp.
                count = count + 1;
            }
        }
        return count;
    }

    // Encapulate tests of the test routine so it can be Fritz'ed.
    void testFritshCarlson(int index, TGraph& graph,
                           std::string name, bool plot,
                           unsigned int seed) {

        // Possibly draw the graph.
        if (plot) {
            graph.SetLineStyle(2);
            graph.SetMarkerSize(3);
            graph.Draw("AL*");
        }

        TSpline3 spline("splineOfGraph",&graph);
        if (plot) {
            spline.SetLineWidth(2);
            spline.SetLineColor(kRed);
            spline.Draw("same");
        }

        const int nKnots = spline.GetNp();
        EXPECT_EQ(nKnots,graph.GetN());

        // Get the knots out of TSpline3
        std::vector<double> xValues;
        std::vector<double> yValues;
        std::vector<double> slopes;
        for (int knot = 0; knot < nKnots; ++knot) {
            double xx;
            double yy;
            double ss;
            spline.GetKnot(knot,xx,yy);
            ss = spline.Derivative(xx);
            xValues.push_back(xx);
            yValues.push_back(yy);
            slopes.push_back(ss);
        }

        if (seed != 0) {
            for (int knot = 0; knot < nKnots; ++knot) {
                slopes[knot] = 20.0*myRand(seed) - 10.0;
            }
        }

        // Build the general spline data from TSpline3
        const int dim = 3*nKnots + 2;
        std::vector<double> data(dim);

        // Set crazy values without in the least language dependent way.
        for (int i = 0; i<dim; ++i) data[i] = std::nan("not-set");

        // Fill the unused fields in data.
        data[0] = -3.0;
        data[1] = 0.0;

        // Copy the knots.
        for (int knot = 0; knot < nKnots; ++knot) {
            data[2+3*knot+0] = yValues[knot];
            data[2+3*knot+1] = slopes[knot];
            data[2+3*knot+2] = xValues[knot];
        }

        // Fill a graph with the general spline results.
        TGraph generalSpline;
        int point = 0;
        for (double xxx = spline.GetXmin();
             xxx<=spline.GetXmax(); xxx += 0.01) {
            double calcValue = CalculateGeneralSpline(xxx,-100.0, 100.0,
                                                      data.data(), dim);
            generalSpline.SetPoint(point++, xxx, calcValue);
        }

        // Possibly plot the result for debugging.
        if (plot) {
            generalSpline.SetLineWidth(1.5);
            generalSpline.SetLineColor(kGreen);
            generalSpline.Draw("same");
        }

        // Build a monotonic version of the original general spline.
        std::vector<double> mData(dim);
        for (int i=0; i<dim; ++i) mData[i] = data[i]; // Explicit loop to copy
        std::vector<double> mSlopes = slopes;
        unitTest::FritshCarlson::MonotonicCondition(xValues,yValues,mSlopes);
        for (int knot = 0; knot < nKnots; ++knot) {
            mData[2+3*knot+0] = yValues[knot];
            mData[2+3*knot+1] = mSlopes[knot];
            mData[2+3*knot+2] = xValues[knot];
        }

        // Fill a graph with the monotonic spline.
        TGraph monotonicSpline;
        point = 0;
        for (double xxx = spline.GetXmin();
             xxx<=spline.GetXmax(); xxx += 0.01) {
            double calcValue
                = CalculateGeneralSpline(xxx,-100.0, 100.0,
                                         mData.data(), dim);
            monotonicSpline.SetPoint(point++, xxx, calcValue);
        }

        // Possibly plot the monotonic spline for debugging.
        if (plot) {
            monotonicSpline.SetLineWidth(4);
            monotonicSpline.SetLineColor(kBlue);
            monotonicSpline.Draw("same");
        }

        // Make sure general spline and monotonicSpline agree at knots.  Don't
        // test the last knot to work around a CalculateGeneralSpline bug
        // (that may be fixed.
        for (int i = 0; i<xValues.size()-1; ++i) {
            double x = xValues[i];
            double y = yValues[i];
            double s = slopes[i];
            double m = mSlopes[i];
            double generalValue
                = CalculateGeneralSpline(x,-100.0, 100.0,
                                         data.data(), dim);
            double monotonicValue
                = CalculateGeneralSpline(x,-100.0, 100.0,
                                         mData.data(), dim);
            EXPECT_NEAR(generalValue,monotonicValue,1e-8)
                << name << ":" << index
                << ": General and Monotonic must agree at knot " << i;
        }

        // Count the number of flips in the monotonic spline and see if it
        // matches the expected number of flips.
        int flips = 0;
        double lastXXX = xValues[0];
        double last = CalculateGeneralSpline(lastXXX,-100.0, 100.0,
                                             mData.data(), dim);
        double sign = 1;
        if (last > CalculateGeneralSpline(lastXXX + 0.001,
                                          -100.0, 100.0,
                                          mData.data(), dim)) sign = -1;
        int step = 0;
        // Don't use the last knot to avoid a CalculateGeneralSpline bug.
        double xxxStep = (xValues[xValues.size()-2] - xValues[0])/1000.0;
        for (double xxx = xValues[0]+xxxStep;
             xxx<=xValues[xValues.size()-2]; xxx += xxxStep) {
            double calcValue
                = CalculateGeneralSpline(xxx,-100.0, 100.0,
                                         mData.data(), dim);
            ++step;
            ASSERT_LE(step,10000);
            if (sign > 0 and last > calcValue) {
                ++flips;
                sign = -1;
            }
            else if (sign < 0 and last < calcValue) {
                ++flips;
                sign = 1;
            }
            lastXXX = xxx;
            last = calcValue;
        }

        // Don't use the last knot to avoid old CalculateGeneralSpline bug.
        int expectedFlips = ExpectedFlips(xValues, yValues, xValues.size()-1);
        EXPECT_EQ(flips, expectedFlips)
            << name << ":" << index << ": Wrong number of flips in spline";

        // Possibly draw the results
        if ((index < 5 || (expectedFlips != flips)) && plot) {
            std::ostringstream baseName;
            baseName << "unitTest_ApplyMonotonicCondition"
                     << "_" << name << "_" << index;
            gPad->Print((baseName.str() + ".pdf").c_str());
            gPad->Print((baseName.str() + ".png").c_str());
        }
    }
};

TEST(MonotonicCondition,FritshCarlson) {
    // Test the test code.  This explicitly tries to catch if we have problems
    // with the test since it will be used as a basis to test GUNDAM.
    TGraph graph;
    int point = 0;
    graph.SetPoint(point++,-3.0, 0.0);
    graph.SetPoint(point++, 0.0, 1.0);
    graph.SetPoint(point++, 1.0, 2.0);
    graph.SetPoint(point++, 2.0, 4.0);
    graph.SetPoint(point++, 3.0, 5.0);
    graph.SetPoint(point++, 4.4, 5.5);
    graph.SetPoint(point++, 5.1, 4.5);
    graph.SetPoint(point++, 6.0, 5.5);
    testFritshCarlson(0, graph,"FritshCarlson",true, 0);

    /// Now Fritz it with random values.
    unsigned int seed = 50441;
    for (int i=1; i<1000; ++i) {
        std::cout << "Fritz " << i << std::endl;
        TGraph fritzGraph;
        double x = 0.0;
        int count = 5 + 10.0*myRand(seed);
        for (int j = 0; j<count; ++j) {
            double y = myRand(seed);
            fritzGraph.SetPoint(j,x, y);
            x += myRand(seed)+0.5;
        }
        testFritshCarlson(i, fritzGraph, "FritshCarlson", true, seed);
    }
}

#ifdef TEST_MAKE_MONOTONIC_SPLINE
#include "MakeMonotonicSpline.h"
TEST(MonotonicCondition,MakeMonotonicSpline)
{
    /// Now Fritz it with random values.
    unsigned int seed = 50441;
    for (int trial = 1; trial < 1000; ++trial) {
        std::cout << "Fritz " << trial << std::endl;
        TGraph fritzGraph;
        double x = 0.0;
        int count = 2 + 10.0*myRand(seed);
        for (int j = 0; j<count; ++j) {
            double y = myRand(seed);
            fritzGraph.SetPoint(j,x, y);
            x += myRand(seed)+0.5;
        }
        TSpline3 spline("splineOfGraph",&fritzGraph);
        const int nKnots = spline.GetNp();
        ASSERT_EQ(nKnots,fritzGraph.GetN());

        // Get the knots out of TSpline3
        std::vector<double> xValues;
        std::vector<double> yValues;
        std::vector<double> splineSlopes;
        std::vector<double> fritshCarlson;
        for (int knot = 0; knot < nKnots; ++knot) {
            double xx;
            double yy;
            double ss;
            spline.GetKnot(knot,xx,yy);
            ss = spline.Derivative(xx);
            xValues.push_back(xx);
            yValues.push_back(yy);
            splineSlopes.push_back(ss);
            fritshCarlson.push_back(ss);
        }

        // Apply the explicit Fritsh-Carlson conditions This is for subset L_1
        // in the paper (that's the largest practical subset to calculate, and
        // it's also the simplest subset to calculate)
        ::FritshCarlson::MonotonicCondition(xValues,yValues,fritshCarlson);

        // Apply our conditions
        ::util::MakeMonotonicSpline(xValues,yValues,splineSlopes);

        EXPECT_EQ(fritshCarlson.size(),splineSlopes.size())
            << "MonotonicCondition: Must have same number of slopes";

        // Check that our slopes are always bounded by Fritsh-Carlson
        for (std::size_t i = 0; i < fritshCarlson.size(); ++i) {
            if (fritshCarlson[i] < -std::numeric_limits<double>::epsilon()) {
                EXPECT_GE(splineSlopes[i],fritshCarlson[i])
                    << "Trial " << trial << " slope must be bounded";
            }
            else if (fritshCarlson[i]>std::numeric_limits<double>::epsilon()) {
                EXPECT_LE(splineSlopes[i],fritshCarlson[i])
                    << "Trial " << trial << " slope must be bounded";
            }
            else {
                EXPECT_NEAR(splineSlopes[i],0.0,1E-8)
                    << "Trial " << trial << " slope must be zero";
            }
        }
    }

}
#endif

#define TEST_APPLY_MONOTONIC_CONDITION
#ifdef TEST_APPLY_MONOTONIC_CONDITION
#include <DialUtils.h>
TEST(MonotonicCondition,ApplyMonotonicCondition)
{
    /// Now Fritz it with random values.
    unsigned int seed = 50441;
    for (int trial = 1; trial < 1000; ++trial) {
        std::cout << "Fritz " << trial << std::endl;
        TGraph fritzGraph;
        double x = 0.0;
        int count = 2 + 10.0*myRand(seed);
        for (int j = 0; j<count; ++j) {
            double y = myRand(seed);
            fritzGraph.SetPoint(j,x, y);
            x += myRand(seed)+0.5;
        }
        TSpline3 spline("splineOfGraph",&fritzGraph);
        const int nKnots = spline.GetNp();
        ASSERT_EQ(nKnots,fritzGraph.GetN());

        // Get the knots out of TSpline3
        std::vector<double> xValues;
        std::vector<double> yValues;
        std::vector<double> fritshCarlson;
        std::vector<DialUtils::DialPoint> dialPoints;

        for (int knot = 0; knot < nKnots; ++knot) {
            double xx;
            double yy;
            double ss;
            spline.GetKnot(knot,xx,yy);
            ss = spline.Derivative(xx);
            xValues.push_back(xx);
            yValues.push_back(yy);
            fritshCarlson.push_back(ss);
            dialPoints.emplace_back(DialUtils::DialPoint{xx,yy,ss});
        }

        // Apply the explicit Fritsh-Carlson conditions This is for subset L_1
        // in the paper (that's the largest practical subset to calculate, and
        // it's also the simplest subset to calculate)
        unitTest::FritshCarlson::MonotonicCondition(xValues,yValues,fritshCarlson);

        // Apply our conditions
        DialUtils::applyMonotonicCondition(dialPoints);

        EXPECT_EQ(fritshCarlson.size(),dialPoints.size())
            << "MonotonicCondition: Must have same number of slopes";

        // Check that our slopes are always bounded by Fritsh-Carlson
        for (std::size_t i = 0; i < fritshCarlson.size(); ++i) {
            if (fritshCarlson[i] < -std::numeric_limits<double>::epsilon()) {
                EXPECT_GE(dialPoints[i].slope,fritshCarlson[i])
                    << "Trial " << trial << " slope must be bounded";
            }
            else if (fritshCarlson[i]>std::numeric_limits<double>::epsilon()) {
                EXPECT_LE(dialPoints[i].slope,fritshCarlson[i])
                    << "Trial " << trial << " slope must be bounded";
            }
            else {
                EXPECT_NEAR(dialPoints[i].slope,0.0,1E-8)
                    << "Trial " << trial << " slope must be zero";
            }
        }
    }

}
#endif

// Local Variables:
// mode:c++
// c-basic-offset:4
// End:
