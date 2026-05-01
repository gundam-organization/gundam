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

namespace {
    double myRand(unsigned int& seed) {
        const int m = (2<<15)-1;
        const int a = 1341;
        const int c = 4321;
        seed = (a*seed + c) % m;
        return 1.0*seed/m;
    }

    void testGeneralSpline(TGraph& graph, std::string testName, bool plot) {
        TSpline3 spline("splineOfGraph",&graph);
        graph.Draw("AL*");
        graph.SetMinimum(-0.5);
        graph.SetMaximum(6.0);
        spline.SetLineWidth(5);
        spline.SetLineColor(kRed);
        spline.Draw("same");
        const int nKnots = spline.GetNp();
        EXPECT_EQ(nKnots,graph.GetN()) << testName << ": Knots match graph";
        const int dim = 3*nKnots + 2;
        double data[dim];
        data[0] = -3.0;
        data[1] = 0.0;
        std::cout << testName << ": Number of knots " << nKnots << std::endl;
        for (int knot = 0; knot < nKnots; ++knot) {
            double xx;
            double yy;
            double ss;
            spline.GetKnot(knot,xx,yy);
            ss = spline.Derivative(xx);
            data[2+3*knot+0] = yy;
            data[2+3*knot+1] = ss;
            data[2+3*knot+2] = xx;
        }
        for (int knot=0; knot < nKnots; ++knot) {
            double xxx;
            double yyy;
            spline.GetKnot(knot,xxx,yyy);
            double splineValue = spline.Eval(xxx);
            double calcValue
                = CalculateGeneralSpline(xxx,-100.0, 100.0,
                                         data, dim);
            EXPECT_NEAR(splineValue, calcValue, 1e-5)
                << testName << ": must match at knots";
        }
        TGraph generalSpline;
        int point = 0;
        for (double xxx = spline.GetXmin()-1.0;
             xxx<=spline.GetXmax()+1.0; xxx += 0.01) {
            double splineValue = spline.Eval(xxx);
            double calcValue
                = CalculateGeneralSpline(xxx,-100.0, 100.0,
                                         data, dim);
            generalSpline.SetPoint(point++, xxx, calcValue);
            EXPECT_NEAR(splineValue, calcValue, 1E-6)
                << testName << ": Spline Mismatch at " << xxx << " " << point-1;
        }
        generalSpline.SetLineWidth(3);
        generalSpline.SetLineColor(kGreen);
        generalSpline.Draw("same");
        if (plot) {
            gPad->Print(
                ("unitTest_CalculateGeneralSpline_"+testName+".pdf").c_str());
            gPad->Print(
                ("unitTest_CalculateGeneralSpline_"+testName+".png").c_str());
        }
    }

    void testGeneralCatmullRom(TGraph& graph, std::string testName, bool plot) {
        graph.Draw("AL*");
        TSpline3 spline("splineOfGraph",&graph);
        spline.SetLineWidth(5);
        spline.SetLineColor(kRed);
        spline.Draw("same");
        const int nKnots = graph.GetN();
        EXPECT_EQ(nKnots,graph.GetN()) << testName << ": Knots match graph";
        const int dim = 3*nKnots + 2;
        double data[dim];
        data[0] = -3.0;
        data[1] = 0.0;
        std::cout << testName << ": Number of knots " << nKnots << std::endl;
        for (int knot = 0; knot < nKnots; ++knot) {
            double xx = graph.GetPointX(knot);
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
            data[2+3*knot+0] = yy;
            data[2+3*knot+1] = ss;
            data[2+3*knot+2] = xx;
        }
        for (int knot=0; knot < nKnots; ++knot) {
            double xxx = graph.GetPointX(knot);
            double yyy = graph.GetPointY(knot);
            double calcValue
                = CalculateGeneralSpline(xxx,-100.0, 100.0,
                                         data, dim);
            EXPECT_NEAR(yyy, calcValue, 1e-5)
                << testName << ": must match at knots";
        }
        TGraph generalSpline;
        int point = 0;
        for (double xxx = spline.GetXmin();
             xxx<=spline.GetXmax(); xxx += 0.01) {
            double calcValue
                = CalculateGeneralSpline(xxx,-100.0, 100.0,
                                         data, dim);
            generalSpline.SetPoint(point++, xxx, calcValue);
        }
        generalSpline.SetLineWidth(3);
        generalSpline.SetLineColor(kGreen);
        generalSpline.Draw("same");
        if (plot) {
            gPad->Print(
                ("unitTest_CalculateGeneralCatmullRom_"+testName+".pdf").c_str());
            gPad->Print(
                ("unitTest_CalculateGeneralCatmullRom_"+testName+".png").c_str());
        }
    }
}

TEST(GeneralSpline,MatchTSpline3ThreeKnot) {
    // Define a TSpline3 and make sure that GeneralSpline reproduces it
    TGraph graph;
    graph.SetPoint(0,-3.0, 0.0);
    graph.SetPoint(1, 0.0, 1.0);
    graph.SetPoint(2, 3.0, 5.5);
    testGeneralSpline(graph,"MatchTSpline3ThreeKnot",true);
}

TEST(GeneralSpine,MatchTSpline3FiveKnot) {
    // Define a TGraph and make sure that GeneralSpline reproduces it
    TGraph graph(3);
    graph.SetPoint(0,-3.0, 0.0);
    graph.SetPoint(1, 0.1, 1.0);
    graph.SetPoint(2, 2.3, 3.5);
    graph.SetPoint(3, 3.0, 5.5);
    graph.SetPoint(4, 4.3, 5.8);
    graph.Draw("AL*");

    graph.SetMinimum(-1.0);
    graph.SetMaximum(7.0);
    testGeneralSpline(graph,"MatchTSPline3FiveKnot",true);
}

TEST(GeneralSpline,LargeSpline) {
    // Define a large spline and make sure that GeneralSpline reproduces
    // it
    TGraph graph;
    int count = 0;
    for (int i = -10; i<11; ++i) {
        double v = i;
        graph.SetPoint(count++,v, std::abs(v));
    }
    testGeneralSpline(graph,"LargeSpline",true);
}

TEST(GeneralSpline,MatchEnds) {
    // Define a large spline and make sure that GeneralSpline reproduces
    // it
    TGraph graph;
    int count = 0;
    graph.SetPoint(count++,-3.0, 0.0);
    graph.SetPoint(count++, 0.0, 1.0);
    graph.SetPoint(count++, 1.0, 2.0);
    graph.SetPoint(count++, 2.0, 4.0);
    graph.SetPoint(count++, 3.0, 5.0);
    graph.SetPoint(count++, 4.4, 5.5);
    graph.SetPoint(count++, 5.1, 4.5);
    graph.SetPoint(count++, 6.0, 5.5);
    testGeneralSpline(graph,"MatchEnds",true);
}

TEST(GeneralSpline,CatmullRomFritz) {
    unsigned int seed = 50441;
    for (int i=0; i<100; ++i) {
        TGraph graph;
        double x = 0.0;
        int count = 10;
        for (int j = 0; j<count; ++j) {
            double y = myRand(seed);
            graph.SetPoint(j,x, y);
            x += myRand(seed)+0.5;
        }
        testGeneralCatmullRom(graph,"CatmullRomFritz",true);
    }
}

TEST(GeneralSpline,TSpline3Fritz) {
    unsigned int seed = 50441;
    for (int i=0; i<100; ++i) {
        TGraph graph;
        double x = 0.0;
        int count = 10;
        for (int j = 0; j<count; ++j) {
            double y = myRand(seed);
            graph.SetPoint(j,x, y);
            x += myRand(seed)+0.5;
        }
        testGeneralSpline(graph,"TSpline3Fritz",true);
    }
}

// Local Variables:
// mode:c++
// c-basic-offset:4
// End:
