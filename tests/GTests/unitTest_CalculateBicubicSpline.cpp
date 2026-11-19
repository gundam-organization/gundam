#include <iostream>
#include <string>
#include <memory>
#include <cmath>

#include <gtest/gtest.h>

#include <TGraph.h>
#include <TGraph2D.h>
#include <TPad.h>

////////////////////////////////////////////////////////////////////////
// Test the CalculateBicubicSpine routine on the CPU.

#include "CalculateBicubicSpline.h"

TEST(BicubicSpline,SevenBySeven) {
    const int nX = 7;
    const int nY = 7;
    double knots[nX*nY];
    double xx[nX];
    double yy[nY];
    for (int i=0; i<nX; ++i) {
        xx[i] = -1.0 + 2.0*i/(nX-1.0);
    }
    for (int i=0; i<nY; ++i) {
        yy[i] = -1.0 + 2.0*i/(nY-1.0);
    }
    for (int i=0; i<nX; ++i) {
        for (int j=0; j<nY; ++j) {
            double x = xx[i];
            double y = yy[j];
            double v = 1.0 - x*x - y*y;
            knots[i*nY + j] = v;
        }
    }

    std::unique_ptr<TGraph2D> graph1(new TGraph2D());
    std::vector<std::unique_ptr<TGraph>> graph1x;
    int p = 0;
    for (double x = -1.5; x <= 1.5; x += 0.1) {
        graph1x.emplace_back(new TGraph());
        int px = 0;
        for (double y = -1.5; y <= 1.5; y += 0.1) {
            double v0
                = CalculateBicubicSpline(x, y,
                                         -10.0, 10.0,
                                         knots, nX, nY,
                                         xx, nX,
                                         yy, nY);
            double v1
                = CalculateBicubicSpline(x, -y,
                                         -10.0, 10.0,
                                         knots, nX, nY,
                                         xx, nX,
                                         yy, nY);
            double v2
                = CalculateBicubicSpline(-x, y,
                                         -10.0, 10.0,
                                         knots, nX, nY,
                                         xx, nX,
                                         yy, nY);
            double v3
                = CalculateBicubicSpline(-x, -y,
                                         -10.0, 10.0,
                                         knots, nX, nY,
                                         xx, nX,
                                         yy, nY);
            EXPECT_NEAR(v0, v1, 1E-8) << "Symmetry +- (test 1)";
            EXPECT_NEAR(v0, v2, 1E-8) << "Symmetry -+ (test 1)";
            EXPECT_NEAR(v0, v3, 1E-8) << "Symmetry -- (test 1)";
            graph1->SetPoint(p++,x,y,v0);
            graph1x.back()->SetPoint(px++, y, v0);
        }
    }
    graph1->Draw("colz");
    gPad->Print("unitTest_BicubicSpline_SevenBySeven2D.pdf");
    gPad->Print("unitTest_BicubicSpline_SevenBySeven2D.png");

    graph1x.front()->Draw("A");
    graph1x.front()->SetMaximum(1.0);
    for (auto& g: graph1x) {
        g->Draw();
        gPad->Update();
    }
    gPad->Print("unitTest_CalculateBicubicSpline_SevenBySevenX.pdf");
    gPad->Print("unitTest_CalculateBicubicSpline_SevenBySevenX.png");
}

TEST(BicubicSpline,TwelveByTwelve) {
    const int nX = 12;
    const int nY = 12;
    double knots[nX*nY];
    double xx[nX];
    double yy[nY];
    for (int i=0; i<nX; ++i) {
        xx[i] = -1.0 + 2.0*i/(nX-1.0);
    }
    for (int i=0; i<nY; ++i) {
        yy[i] = -1.0 + 2.0*i/(nY-1.0);
    }
    for (int i=0; i<nX; ++i) {
        for (int j=0; j<nY; ++j) {
            double x = xx[i];
            double y = yy[j];
            double v = 0.5 - x*x - y*y;
            if (v < 0) v = 0.0;
            else v = 1.0;
            knots[i*nY + j] = v;
        }
    }

    std::unique_ptr<TGraph2D> graph1(new TGraph2D());
    int p = 0;
    for (double x = -1.5; x <= 1.5; x += 0.01) {
        for (double y = -1.5; y <= 1.5; y += 0.01) {
            double v0
                = CalculateBicubicSpline(x, y,
                                         -10.0, 10.0,
                                         knots, nX, nY,
                                         xx, nX,
                                         yy, nY);
                double v1
                    = CalculateBicubicSpline(x, -y,
                                                     -10.0, 10.0,
                                                     knots, nX, nY,
                                                     xx, nX,
                                                     yy, nY);
                double v2
                    = CalculateBicubicSpline(-x, y,
                                                     -10.0, 10.0,
                                                     knots, nX, nY,
                                                     xx, nX,
                                                     yy, nY);
                double v3
                    = CalculateBicubicSpline(-x, -y,
                                                     -10.0, 10.0,
                                                     knots, nX, nY,
                                                     xx, nX,
                                                     yy, nY);
                EXPECT_NEAR(v0, v1, 1E-8) << "Symmetry +- (test 2)";
                EXPECT_NEAR(v0, v2, 1E-8) << "Symmetry -+ (test 2)";
                EXPECT_NEAR(v0, v3, 1E-8) << "Symmetry -- (test 2)";
                graph1->SetPoint(p++,x,y,v0);
        }
    }
    graph1->Draw("colz");
    gPad->Print("unitTest_CalculateBicubicSpline_TwelveByTwelve.pdf");
    gPad->Print("unitTest_CalculateBicubicSpline_TwleveByTwelve.png");
}

TEST(BiCubicSpline,TwelveByTwelveSlice) {
    const int nX = 12;
    const int nY = 12;
    double knots[nX*nY];
    double xx[nX];
    double yy[nY];
    for (int i=0; i<nX; ++i) {
        xx[i] = -1.0 + 2.0*i/(nX-1.0);
    }
    for (int i=0; i<nY; ++i) {
        yy[i] = -1.0 + 2.0*i/(nY-1.0);
    }
    for (int i=0; i<nX; ++i) {
        for (int j=0; j<nY; ++j) {
            double x = xx[i];
            double y = yy[j];
            double v = 0.25 - x*x - y*y;
            if (v < 0) v = 0.0;
            else v = 1.0;
            knots[i*nY + j] = v;
        }
    }

    std::unique_ptr<TGraph> graph1(new TGraph());
    int p = 0;
    for (double x = -1.5; x <= 1.5; x += 0.01) {
        double y = 0.0;
        double v0
            = CalculateBicubicSpline(x, y,
                                     -10.0, 10.0,
                                     knots, nX, nY,
                                     xx, nX,
                                     yy, nY);
        double v2
            = CalculateBicubicSpline(-x, y,
                                     -10.0, 10.0,
                                     knots, nX, nY,
                                     xx, nX,
                                     yy, nY);
        EXPECT_NEAR(v0, v2, 1E-8) << "Symmetry -- (test 3)";
        graph1->SetPoint(p++,x,v0);
    }
    graph1->Draw("AC");
    gPad->Print("unitTest_CalculateBicubicSpline_TwelveByTwelveSlice.pdf");
    gPad->Print("unitTest_CalculateBicubicSpline_TwelveByTwelveSlice.png");
}

TEST(BicubicSpline,Gradient) {
    const int nX = 8;
    const int nY = 9;
    double knots[nX*nY];
    double xx[nX];
    double yy[nY];
    for (int i=0; i<nX; ++i) {
        xx[i] = -1.0 + 2.0*i/(nX-1.0);
    }
    for (int i=0; i<nY; ++i) {
        yy[i] = -1.2 + 2.4*i/(nY-1.0);
    }
    for (int i=0; i<nX; ++i) {
        for (int j=0; j<nY; ++j) {
            const double x = xx[i];
            const double y = yy[j];
            knots[i*nY + j] = 0.3 + 0.2*x - 0.4*y + 0.5*x*y
                              + 0.1*x*x - 0.05*y*y + 0.2*x*x*y;
        }
    }

    constexpr double lowerBound = -1E20;
    constexpr double upperBound = 1E20;
    constexpr double eps = 1E-6;

    for (double x = -0.65; x <= 0.65; x += 0.17) {
        for (double y = -0.75; y <= 0.75; y += 0.19) {
            double grad[2] = {0.0, 0.0};
            const double v = CalculateBicubicSpline(
                grad, x, y, lowerBound, upperBound, knots, nX, nY, xx, nX, yy, nY);
            const double vCompat = CalculateBicubicSpline(
                x, y, lowerBound, upperBound, knots, nX, nY, xx, nX, yy, nY);
            EXPECT_NEAR(v, vCompat, 1E-12);

            const double vxHigh = CalculateBicubicSpline(
                x+eps, y, lowerBound, upperBound, knots, nX, nY, xx, nX, yy, nY);
            const double vxLow = CalculateBicubicSpline(
                x-eps, y, lowerBound, upperBound, knots, nX, nY, xx, nX, yy, nY);
            const double vyHigh = CalculateBicubicSpline(
                x, y+eps, lowerBound, upperBound, knots, nX, nY, xx, nX, yy, nY);
            const double vyLow = CalculateBicubicSpline(
                x, y-eps, lowerBound, upperBound, knots, nX, nY, xx, nX, yy, nY);

            EXPECT_NEAR(grad[0], (vxHigh-vxLow)/(2.0*eps), 1E-7);
            EXPECT_NEAR(grad[1], (vyHigh-vyLow)/(2.0*eps), 1E-7);
            EXPECT_NEAR(grad[0], CalculateBicubicSplineGradient(
                x, y, 0, lowerBound, upperBound, knots, nX, nY, xx, nX, yy, nY), 1E-12);
            EXPECT_NEAR(grad[1], CalculateBicubicSplineGradient(
                x, y, 1, lowerBound, upperBound, knots, nX, nY, xx, nX, yy, nY), 1E-12);
        }
    }

    double grad[2] = {1.0, 1.0};
    CalculateBicubicSpline(
        grad, 0.2, -0.3, -0.1, 0.1, knots, nX, nY, xx, nX, yy, nY);
    EXPECT_EQ(grad[0], 0.0);
    EXPECT_EQ(grad[1], 0.0);
}

// Local Variables:
// mode:c++
// c-basic-offset:4
// End:
