#include <iostream>
#include <string>
#include <memory>
#include <cmath>

#include <gtest/gtest.h>

#include <TGraph.h>
#include <TGraph2D.h>
#include <TPad.h>

////////////////////////////////////////////////////////////////////////
// Test the CalculateBilinearInterpolation routine on the CPU.

#include "CalculateBilinearInterpolation.h"

TEST(BilinearInterpolation,SevenBySeven) {
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
                = CalculateBilinearInterpolation(x, y,
                                                 -10.0, 10.0,
                                                 knots, nX, nY,
                                                 xx, nX,
                                                 yy, nY);
            double v1
                = CalculateBilinearInterpolation(x, -y,
                                                 -10.0, 10.0,
                                                 knots, nX, nY,
                                                 xx, nX,
                                                 yy, nY);
            double v2
                = CalculateBilinearInterpolation(-x, y,
                                                 -10.0, 10.0,
                                                 knots, nX, nY,
                                                     xx, nX,
                                                     yy, nY);
                double v3
                    = CalculateBilinearInterpolation(-x, -y,
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
    gPad->Print("unitTest_BilinearInterpolation_SevenBySeven2D.pdf");
    gPad->Print("unitTest_BilinearInterpolation_SevenBySeven2D.png");

    graph1x.front()->Draw("A");
    graph1x.front()->SetMaximum(1.0);
    for (auto& g: graph1x) {
        g->Draw();
        gPad->Update();
    }
    gPad->Print("unitTest_CalculateBilinearInterpolation_SevenBySevenX.pdf");
    gPad->Print("unitTest_CalculateBilinearInterpolation_SevenBySevenX.png");
}

TEST(BilinearInterpolation,TwelveByTwelve) {
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
                = CalculateBilinearInterpolation(x, y,
                                                 -10.0, 10.0,
                                                 knots, nX, nY,
                                                 xx, nX,
                                                 yy, nY);
            double v1
                = CalculateBilinearInterpolation(x, -y,
                                                 -10.0, 10.0,
                                                 knots, nX, nY,
                                                 xx, nX,
                                                 yy, nY);
            double v2
                = CalculateBilinearInterpolation(-x, y,
                                                 -10.0, 10.0,
                                                 knots, nX, nY,
                                                 xx, nX,
                                                 yy, nY);
            double v3
                = CalculateBilinearInterpolation(-x, -y,
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
    gPad->Print("unitTest_CalculateBilinearInterpolation_TwelveByTwelve.pdf");
    gPad->Print("unitTest_CalculateBilinearInterpolation_TwleveByTwelve.png");
}

TEST(BilinearInterpolation,TwelveByTwelveSlice) {
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
            = CalculateBilinearInterpolation(x, y,
                                             -10.0, 10.0,
                                             knots, nX, nY,
                                             xx, nX,
                                             yy, nY);
        double v2
            = CalculateBilinearInterpolation(-x, y,
                                             -10.0, 10.0,
                                             knots, nX, nY,
                                             xx, nX,
                                             yy, nY);
        EXPECT_NEAR(v0, v2, 1E-8) << "Symmetry -- (test 3)";
        graph1->SetPoint(p++,x,v0);
    }
    graph1->Draw("AC");
    gPad->Print("unitTest_CalculateBilinearInterpolation_TwelveByTwelveSlice.pdf");
    gPad->Print("unitTest_CalculateBilinearInterpolation_TwelveByTwelveSlice.png");
}

// Local Variables:
// mode:c++
// c-basic-offset:4
// End:
