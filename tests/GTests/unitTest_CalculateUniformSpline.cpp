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

namespace {
    double myRand(unsigned int& seed) {
        const int m = (2<<15)-1;
        const int a = 1341;
        const int c = 4321;
        seed = (a*seed + c) % m;
        return 1.0*seed/m;
    }

    void testUniformSpline3(TGraph& graph, std::string testName, bool plot) {
        TSpline3 spline("splineOfGraph",&graph);
        graph.Draw("AL*");
        graph.SetMinimum(-0.5);
        graph.SetMaximum(6.0);
        spline.SetLineWidth(5);
        spline.SetLineColor(kRed);
        spline.Draw("same");
        const int nKnots = spline.GetNp();
        EXPECT_EQ(nKnots,graph.GetN()) << testName << ": Knots match graph";
        const int dim = 2*nKnots + 2;
        std::vector<double> data(dim);
        double xx;
        double yy;
        double ss;
        spline.GetKnot(0,xx,yy);
        data[0] = xx; // lower bound
        spline.GetKnot(nKnots-1,xx,yy);
        data[1] = (xx - data[0])/(nKnots-1.0);  // X step
        std::cout << testName << ": Number of knots " << nKnots << std::endl;
        for (int knot = 0; knot < nKnots; ++knot) {
            spline.GetKnot(knot,xx,yy);
            ss = spline.Derivative(xx);
            data[2+2*knot+0] = yy;
            data[2+2*knot+1] = ss;
        }
        for (int knot=0; knot < nKnots; ++knot) {
            double xxx;
            double yyy;
            spline.GetKnot(knot,xxx,yyy);
            double splineValue = spline.Eval(xxx);
            double calcValue
                = CalculateUniformSpline(xxx,-100.0, 100.0,
                                         data.data(), dim);
            EXPECT_NEAR(splineValue, calcValue, 1e-5)
                << testName << ": must match at knots";
        }
        TGraph uniformSpline;
        int point = 0;
        for (double xxx = spline.GetXmin()-1.0;
             xxx<=spline.GetXmax()+1.0; xxx += 0.01) {
            double splineValue = spline.Eval(xxx);
            double calcValue
                = CalculateUniformSpline(xxx,-100.0, 100.0,
                                         data.data(), dim);
            uniformSpline.SetPoint(point++, xxx, calcValue);
            EXPECT_NEAR(splineValue, calcValue, 1E-6)
                << testName << ": Spline Mismatch at " << xxx << " " << point-1;
        }
        uniformSpline.SetLineWidth(3);
        uniformSpline.SetLineColor(kGreen);
        uniformSpline.Draw("same");
        if (plot) {
            gPad->Print(
                ("unitTest_CalculateUniformSpline_"+testName+".pdf").c_str());
            gPad->Print(
                ("unitTest_CalculateUniformSpline_"+testName+".png").c_str());
        }
    }

    void testUniformCatmullRom(TGraph& graph, std::string testName, bool plot) {
        graph.Draw("AL*");
        TSpline3 spline("splineOfGraph",&graph);
        spline.SetLineWidth(5);
        spline.SetLineColor(kRed);
        spline.Draw("same");
        const int nKnots = graph.GetN();
        EXPECT_EQ(nKnots,graph.GetN()) << testName << ": Knots match graph";
        const int dim = 2*nKnots + 2;
        std::vector<double> data(dim);
        double xx;
        double yy;
        spline.GetKnot(0,xx,yy);
        data[0] = xx; // lower bound
        spline.GetKnot(nKnots-1,xx,yy);
        data[1] = (xx - data[0])/(nKnots-1.0);  // X step
        std::cout << testName << ": Number of knots " << nKnots << std::endl;
        for (int knot = 0; knot < nKnots; ++knot) {
            double yy = graph.GetPointY(knot);
            double ss = 0;
            if (knot == 0) {
                ss = graph.GetPointY(knot+1) - graph.GetPointY(knot);
                ss /= graph.GetPointX(knot+1) - graph.GetPointX(knot);
            }
            else if (knot == nKnots-1) {
                ss = graph.GetPointY(knot) - graph.GetPointY(knot-1);
                ss /= graph.GetPointX(knot) - graph.GetPointX(knot-1);
            }
            else {
                ss = graph.GetPointY(knot+1) - graph.GetPointY(knot-1);
                ss /= graph.GetPointX(knot+1) - graph.GetPointX(knot-1);
            }
            data[2+2*knot+0] = yy;
            data[2+2*knot+1] = ss;
        }
        for (int knot=0; knot < nKnots; ++knot) {
            double xxx = graph.GetPointX(knot);
            double yyy = graph.GetPointY(knot);
            double calcValue
                = CalculateUniformSpline(xxx,-100.0, 100.0,
                                         data.data(), dim);
            EXPECT_NEAR(yyy, calcValue, 1e-5)
                << testName << ": must match at knots";
        }
        TGraph uniformSpline;
        int point = 0;
        for (double xxx = spline.GetXmin();
             xxx<=spline.GetXmax(); xxx += 0.01) {
            double calcValue
                = CalculateUniformSpline(xxx,-100.0, 100.0,
                                         data.data(), dim);
            uniformSpline.SetPoint(point++, xxx, calcValue);
        }
        uniformSpline.SetLineWidth(3);
        uniformSpline.SetLineColor(kGreen);
        uniformSpline.Draw("same");
        if (plot) {
            gPad->Print(
                ("unitTest_CalculateUniformCatmullRom_"+testName+".pdf").c_str());
            gPad->Print(
                ("unitTest_CalculateUniformCatmullRom_"+testName+".png").c_str());
        }
    }
}

TEST(UniformSpline,MatchTSpline3ThreeKnot) {
    // Define a TSpline3 and make sure that UniformSpline reproduces it
    TGraph graph;
    graph.SetPoint(0,-3.0, 0.0);
    graph.SetPoint(1, 0.0, 1.0);
    graph.SetPoint(2, 3.0, 5.5);
    testUniformSpline3(graph,"MatchTSpline3ThreeKnot",true);
}

TEST(UniformSpine,MatchTSpline3FiveKnot) {
    // Define a TGraph and make sure that UniformSpline reproduces it
    TGraph graph(3);
    graph.SetPoint(0,-1.0, 0.0);
    graph.SetPoint(1, 0.0, 1.0);
    graph.SetPoint(2, 1.0, 3.5);
    graph.SetPoint(3, 2.0, 5.5);
    graph.SetPoint(4, 3.0, 5.8);
    graph.Draw("AL*");

    graph.SetMinimum(-1.0);
    graph.SetMaximum(7.0);
    testUniformSpline3(graph,"MatchTSPline3FiveKnot",true);
}

TEST(UniformSpline,LargeSpline) {
    // Define a large spline and make sure that UniformSpline reproduces
    // it
    TGraph graph;
    int count = 0;
    for (int i = -10; i<11; ++i) {
        double v = i;
        graph.SetPoint(count++,v, std::abs(v));
    }
    testUniformSpline3(graph,"LargeSpline",true);
}

TEST(UniformSpline,MatchEnds) {
    // Define a large spline and make sure that UniformSpline reproduces
    // it
    TGraph graph;
    int count = 0;
    graph.SetPoint(count++,-1.0, 0.0);
    graph.SetPoint(count++, 0.0, 1.0);
    graph.SetPoint(count++, 1.0, 2.0);
    graph.SetPoint(count++, 2.0, 4.0);
    graph.SetPoint(count++, 3.0, 5.0);
    graph.SetPoint(count++, 4.0, 5.5);
    graph.SetPoint(count++, 5.0, 4.5);
    graph.SetPoint(count++, 6.0, 5.5);
    testUniformSpline3(graph,"MatchEnds",true);
}

TEST(UniformSpline,CatmullRomFritz) {
    unsigned int seed = 50441;
    for (int i=0; i<100; ++i) {
        TGraph graph;
        double x = 0.0;
        int count = 10;
        for (int j = 0; j<count; ++j) {
            double y = myRand(seed);
            graph.SetPoint(j,x, y);
            x += 1.0;
        }
        testUniformCatmullRom(graph,"CatmullRomFritz",true);
    }
}

TEST(UniformSpline,TSpline3Fritz) {
    unsigned int seed = 50441;
    for (int i=0; i<100; ++i) {
        TGraph graph;
        double x = 0.0;
        int count = 10;
        for (int j = 0; j<count; ++j) {
            double y = myRand(seed);
            graph.SetPoint(j,x, y);
            x += 1.0;
        }
        testUniformSpline3(graph,"TSpline3Fritz",true);
    }
}

// Local Variables:
// mode:c++
// c-basic-offset:4
// End:
