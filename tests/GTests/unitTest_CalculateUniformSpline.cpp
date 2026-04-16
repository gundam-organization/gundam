#include <iostream>
#include <string>
#include <memory>
#include <cmath>

#include <gtest/gtest.h>

#include <TSpline.h>
#include <TGraph.h>
#include <TPad.h>

////////////////////////////////////////////////////////////////////////
// Test the CalculateUniformSpline routine on the CPU.

#include "CalculateUniformSpline.h"

TEST(UniformSpline,MatchTSline3ThreePoints) {
    // Define a TSpline3 and make sure that UniformSpline reproduces it
    TGraph graph;
    graph.SetPoint(0,-3.0, 0.0);
    graph.SetPoint(1, 0.0, 1.0);
    graph.SetPoint(2, 3.0, 5.5);
    TSpline3 spline("splineOfGraph",&graph);
    graph.Draw("AC*");
    graph.SetMinimum(-0.5);
    graph.SetMaximum(6.0);
    spline.SetLineWidth(5);
    spline.SetLineColor(kRed);
    spline.Draw("same");
    const int nKnots = spline.GetNp();
    const int dim = 2*nKnots + 2;
    double data[dim];
    data[0] = -3.0;
    data[1] = 3.0;
    std::cout << "Number of knots " << nKnots << std::endl;
    for (int knot = 0; knot < nKnots; ++knot) {
        double xx;
        double yy;
        double ss;
        spline.GetKnot(knot,xx,yy);
        ss = spline.Derivative(xx);
        data[2+2*knot+0] = yy;
        data[2+2*knot+1] = ss;
    }
    TGraph uniformSpline;
    int point = 0;
    for (double xxx = spline.GetXmin()-1.0;
         xxx<=spline.GetXmax()+1.0; xxx += 0.01) {
        double splineValue = spline.Eval(xxx);
        double calcValue
            = CalculateUniformSpline(xxx,-100.0, 100.0,
                                     data, dim);
        uniformSpline.SetPoint(point++, xxx, calcValue);
        EXPECT_NEAR(splineValue, calcValue, 1E-6)
            << "Test1: Spline Mismatch";
    }
    uniformSpline.SetLineWidth(3);
    uniformSpline.SetLineColor(kGreen);
    uniformSpline.Draw("same");
    gPad->Print("100CheckUniformSpline1.pdf");
    gPad->Print("100CheckUniformSpline1.png");
}

TEST(UniformSpline, MatchTSpine3FivePoint) {
    // Define a TSpline3 and make sure that UniformSpline reproduces it
    TGraph graph(5);
    graph.SetPoint(0,-3.0, 0.0);
    graph.SetPoint(1,-2.0, 1.0);
    graph.SetPoint(2,-1.0, 3.5);
    graph.SetPoint(3, 0.0, 5.5);
    graph.SetPoint(4, 1.0, 5.8);
    graph.Draw("AC*");
    graph.SetMinimum(-1.0);
    graph.SetMaximum(7.0);
    TSpline3 spline("splineOfGraph",&graph);
    spline.SetLineWidth(5);
    spline.SetLineColor(kRed);
    spline.Draw("same");
    const int nKnots = spline.GetNp();
    const int dim = 2*nKnots + 2;
    double data[dim];
    data[0] = -3.0;
    data[1] = 1.0;
    std::cout << "Number of knots " << nKnots << std::endl;
    for (int knot = 0; knot < nKnots; ++knot) {
        double xx;
        double yy;
        double ss;
        spline.GetKnot(knot,xx,yy);
        ss = spline.Derivative(xx);
        data[2+2*knot+0] = yy;
        data[2+2*knot+1] = ss;
    }
    TGraph uniformSpline;
    int point = 0;
    for (double xxx = spline.GetXmin()-1.0;
         xxx<=spline.GetXmax()+1.0; xxx += 0.01) {
        double splineValue = spline.Eval(xxx);
        double calcValue
            = CalculateUniformSpline(xxx,-100.0, 100.0,
                                     data, dim);
        uniformSpline.SetPoint(point++, xxx, calcValue);
        EXPECT_NEAR(splineValue, calcValue, 1E-6)
            << "Test1: Spline Mismatch";
    }
    uniformSpline.SetLineWidth(3);
    uniformSpline.SetLineColor(kGreen);
    uniformSpline.Draw("same");
    gPad->Print("100CheckUniformSpline2.pdf");
    gPad->Print("100CheckUniformSpline2.png");
}

// Local Variables:
// mode:c++
// c-basic-offset:4
// End:
