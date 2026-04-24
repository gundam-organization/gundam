#include <iostream>
#include <string>
#include <memory>
#include <cmath>

#include <gtest/gtest.h>

#include <TGraph.h>
#include <TPad.h>

////////////////////////////////////////////////////////////////////////
// Test the CalculateGraph routine on the CPU.

#include "CalculateGraph.h"

TEST(GraphCheck,TwoPoints) {
    // Test linear interpolation between two points
    double data[] = {0.0, 0.0, 1.0, 1.0};
    int nData = 2;
    std::unique_ptr<TGraph> data1(new TGraph());
    for (int p=0; p<nData; ++p) {
        double x = data[2*p+1];
        double y = data[2*p+0];
        data1->SetPoint(p,x,y);
    }
    std::unique_ptr<TGraph> graph1(new TGraph());
    int p = 0;
    for (double x = -2.0; x <= 2.0; x += 0.1) {
        double v = CalculateGraph(x, -10.0, 10.0, data, 2*nData);
        EXPECT_NEAR(x, v, 1E-6) << "Two Point Tolerance";
        graph1->SetPoint(p++,x,v);
    }
    graph1->Draw("AC");
    data1->Draw("*,same");
    gPad->Print("100CheckGraph1.pdf");
    gPad->Print("100CheckGraph1.png");
}

TEST(LinearInterpolation,ThreePointSymmetric) {
    // Test interpolation between three symmetric points
    int nData = 3;
    double data[] = {1.0, -1.0, 0.0, 0.0, 1.0, 1.0};
    std::unique_ptr<TGraph> data1(new TGraph());
    for (int p=0; p<nData; ++p) {
        double x = data[2*p+1];
        double y = data[2*p+0];
        data1->SetPoint(p,x,y);
    }
    std::unique_ptr<TGraph> graph1(new TGraph());
    int p = 0;
    for (double x = -3.0; x <= 3.0; x += 0.1) {
        double v0 = CalculateGraph(x, -10.0, 10.0, data, 2*nData);
        double v1 = CalculateGraph(-x, -10.0, 10.0, data, 2*nData);
        EXPECT_NEAR(v0, v1, 1E-6)
            << "Three Point Graph (test 2) (X=" << x << ")";
        graph1->SetPoint(p++,x,v0);
    }
    graph1->Draw("AC");
    data1->Draw("*,same");
    gPad->Print("100CheckGraph2.pdf");
    gPad->Print("100CheckGraph2.png");
}

TEST(LinearInterpolation,FourPoint) {
    // Test interpolation between 4 points
    int nData = 4;
    double data[] = {
        1.0, -2.0,
        0.0, -1.0,
        0.0, 1.0,
        1.0, 2.0,
    };
    std::unique_ptr<TGraph> data1(new TGraph());
    for (int p=0; p<nData; ++p) {
        double x = data[2*p+1];
        double y = data[2*p+0];
        data1->SetPoint(p,x,y);
    }
    std::unique_ptr<TGraph> graph1(new TGraph());
    int p = 0;
    for (double x = -12.0; x <= 12.0; x += 0.1) {
        double v0 = CalculateGraph(x, -50.0, 50.0, data, 2*nData);
        double v1 = CalculateGraph(-x, -50.0, 50.0, data, 2*nData);
        EXPECT_NEAR(v0, v1, 1E-6)
            << "Four Point Graph (test 3) (X=" << x << ")";
        graph1->SetPoint(p++,x,v0);
    }
    graph1->Draw("AC");
    data1->Draw("*,same");
    gPad->Print("100CheckGraph3.pdf");
    gPad->Print("100CheckGraph3.png");
}

TEST(LinearInterpolation, NineteenPoints) {
    // Test interpolation between 19 points
    int nData = 19;
    double data[] = {
        30.0, -9.0,
        29.0, -8.0,
        20.0, -7.0,
        16.0, -6.0,
        11.0, -5.0,
        7.0, -4.0,
        4.0, -3.0,
        2.0, -2.0,
        1.0, -1.0,
        0.0, 0.0,
        1.0, 1.0,
        2.0, 2.0,
        4.0, 3.0,
        7.0, 4.0,
        11.0, 5.0,
        16.0, 6.0,
        20.0, 7.0,
        29.0, 8.0,
        30.0, 9.0,
    };
    std::unique_ptr<TGraph> data1(new TGraph());
    for (int p=0; p<nData; ++p) {
        double x = data[2*p+1];
        double y = data[2*p+0];
        data1->SetPoint(p,x,y);
    }
    std::unique_ptr<TGraph> graph1(new TGraph());
    int p = 0;
    for (double x = -12.0; x <= 12.0; x += 0.1) {
        double v0 = CalculateGraph(x, -50.0, 50.0, data, 2*nData);
        double v1 = CalculateGraph(-x, -50.0, 50.0, data, 2*nData);
        EXPECT_NEAR(v0, v1, 1E-6)
            << "Large Graph (test 4) (X=" << x << ")";
        graph1->SetPoint(p++,x,v0);
    }
    graph1->Draw("AC");
    data1->Draw("*,same");
    gPad->Print("100CheckGraph4.pdf");
    gPad->Print("100CheckGraph4.png");
}

TEST(LinearInterpolation,FourAsymmetricPositive) {
    // Test interpolation between 4 symmetric points with a positive slope
    int nData = 4;
    double data[] = {
        -1.0, -2.0,
        0.0, -1.0,
        0.0, 1.0,
        1.0, 2.0,
    };
    std::unique_ptr<TGraph> data1(new TGraph());
    for (int p=0; p<nData; ++p) {
        double x = data[2*p+1];
        double y = data[2*p+0];
        data1->SetPoint(p,x,y);
    }
    std::unique_ptr<TGraph> graph1(new TGraph());
    int p = 0;
    for (double x = -12.0; x <= 12.0; x += 0.1) {
        double v0 = CalculateGraph(x, -50.0, 50.0, data, 2*nData);
        double v1 = CalculateGraph(-x, -50.0, 50.0, data, 2*nData);
        EXPECT_NEAR(v0, -v1, 1E-6)
            << "Four Point Graph (test 5) (X=" << x << ")";
        graph1->SetPoint(p++,x,v0);
    }
    graph1->Draw("AC");
    data1->Draw("*,same");
    gPad->Print("100CheckGraph5.pdf");
    gPad->Print("100CheckGraph5.png");
}

TEST(LinearInterpolation,FourAsymmetricNegative) {
    // Test interpolation between 4 asymmetric points with a negative slope
    int nData = 4;
    double data[] = {
        1.0, -2.0,
        0.0, -1.0,
        0.0, 1.0,
        -1.0, 2.0,
    };
    std::unique_ptr<TGraph> data1(new TGraph());
    for (int p=0; p<nData; ++p) {
        double x = data[2*p+1];
        double y = data[2*p+0];
        data1->SetPoint(p,x,y);
    }
    std::unique_ptr<TGraph> graph1(new TGraph());
    int p = 0;
    for (double x = -12.0; x <= 12.0; x += 0.1) {
        double v0 = CalculateGraph(x, -50.0, 50.0, data, 2*nData);
        double v1 = CalculateGraph(-x, -50.0, 50.0, data, 2*nData);
        EXPECT_NEAR(v0, -v1, 1E-6)
            << "Four Point Graph (test 6) (X=" << x << ")";
        graph1->SetPoint(p++,x,v0);
    }
    graph1->Draw("AC");
    data1->Draw("*,same");
    gPad->Print("100CheckGraph6.pdf");
    gPad->Print("100CheckGraph6.png");
}

TEST(LinearInterpolation,FiveAsymmetric) {
    // Test interpolation between 5 asymmetric points
    int nData = 5;
    double data[] = {
        1.0, -2.0,
        0.0, -1.0,
        0.0, 0.0,
        0.0, 1.0,
        -1.0, 2.0,
    };
    std::unique_ptr<TGraph> data1(new TGraph());
    for (int p=0; p<nData; ++p) {
        double x = data[2*p+1];
        double y = data[2*p+0];
        data1->SetPoint(p,x,y);
    }
    std::unique_ptr<TGraph> graph1(new TGraph());
    int p = 0;
    for (double x = -12.0; x <= 12.0; x += 0.1) {
        double v0 = CalculateGraph(x, -50.0, 50.0, data, 2*nData);
        double v1 = CalculateGraph(-x, -50.0, 50.0, data, 2*nData);
        EXPECT_NEAR(v0, -v1, 1E-7)
            << "Five Point Graph (test 7) (X=" << x << ")";
        graph1->SetPoint(p++,x,v0);
    }
    graph1->Draw("AC");
    data1->Draw("*,same");
    gPad->Print("100CheckGraph7.pdf");
    gPad->Print("100CheckGraph7.png");
}

// Local Variables:
// mode:c++
// c-basic-offset:4
// End:
