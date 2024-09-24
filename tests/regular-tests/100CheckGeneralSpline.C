# !/bin/bash
# Wrap a ROOT macro as a script.
root <<EOF

#include <iostream>
#include <string>
#include <memory>
#include <cmath>

#include <TSpline.h>

////////////////////////////////////////////////////////////////////////
// Test the CalculateGeneralSpline routine on the CPU.

#include "${GUNDAM_ROOT}/src/Utils/include/CalculateGeneralSpline.h"

std::string args{"$*"};

int status{0};

/// Fail if fractional difference between "v1" and "v2" is larger than "tol"
/// THIS IS COPIED HERE TO AVOID DEPENDENCIES
#define TOLERANCE(_msg,_v1,_v2,_tol)                              \
    do {                                                          \
        double _v = (_v1)>0 ? (_v1): -(_v1);                      \
        double _vv = (_v2)>0 ? (_v2): -(_v2);                     \
        double _d = std::abs((_v1)-(_v2));                        \
        double _r = _d/std::max(0.5*(_v+_vv),(_tol));             \
        if (_r < (_tol)) {                                        \
            break;                                                \
        }                                                         \
        ++status;                                                 \
        std::cout << "FAIL:";                                     \
        std::cout << " " << _msg                                  \
                  << std::setprecision(8)                         \
                  << std::scientific                              \
                  << " (" << _r << "<" << (_tol) << ")"           \
                  << " [" << #_v1 << "=" << (_v1)                 \
                  << " " << #_v2 << "=" << (_v2)                  \
                  << " " << _d << "]"                             \
                  << std::endl;                                   \
    } while(false);

int main() {
    std::cout << "Hello world" << std::endl;

#define TEST1
#ifdef TEST1
    {
        // Define a TSpline3 and make sure that GeneralSpline reproduces it
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
        const int dim = 3*nKnots + 2;
        double data[dim];
        data[0] = -3.0;
        data[1] = 0.0;
        std::cout << "Number of knots " << nKnots << std::endl;
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
        TGraph generalSpline;
        int point = 0;
        for (double xxx = spline.GetXmin();
             xxx<=spline.GetXmax(); xxx += 0.01) {
            double splineValue = spline.Eval(xxx);
            double calcValue
                = CalculateGeneralSpline(xxx,-100.0, 100.0,
                                       data, dim);
            generalSpline.SetPoint(point++, xxx, calcValue);
            TOLERANCE("Test1: Spline Mismatch", splineValue, calcValue, 1E-6);
        }
        generalSpline.SetLineWidth(3);
        generalSpline.SetLineColor(kGreen);
        generalSpline.Draw("same");
        gPad->Print("100CheckGeneralSpline1.pdf");
        gPad->Print("100CheckGeneralSpline1.png");
    }
#endif

#define TEST2
#ifdef TEST2
    {
        // Define a TSpline3 and make sure that GeneralSpline reproduces it
        TGraph graph(3);
        graph.SetPoint(0,-3.0, 0.0);
        graph.SetPoint(1, 0.1, 1.0);
        graph.SetPoint(2, 2.3, 3.5);
        graph.SetPoint(3, 3.0, 5.5);
        graph.SetPoint(4, 4.3, 5.8);
        graph.Draw("AC*");
        graph.SetMinimum(-1.0);
        graph.SetMaximum(7.0);
        TSpline3 spline("splineOfGraph",&graph);
        spline.SetLineWidth(5);
        spline.SetLineColor(kRed);
        spline.Draw("same");
        const int nKnots = spline.GetNp();
        const int dim = 3*nKnots + 2;
        double data[dim];
        data[0] = -3.0;
        data[1] = 0.0;
        std::cout << "Number of knots " << nKnots << std::endl;
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
        TGraph generalSpline;
        int point = 0;
        for (double xxx = spline.GetXmin();
             xxx<=spline.GetXmax(); xxx += 0.01) {
            double splineValue = spline.Eval(xxx);
            double calcValue
                = CalculateGeneralSpline(xxx,-100.0, 100.0,
                                       data, dim);
            generalSpline.SetPoint(point++, xxx, calcValue);
            TOLERANCE("Test1: Spline Mismatch", splineValue, calcValue, 1E-6);
        }
        generalSpline.SetLineWidth(3);
        generalSpline.SetLineColor(kGreen);
        generalSpline.Draw("same");
        gPad->Print("100CheckGeneralSpline2.pdf");
        gPad->Print("100CheckGeneralSpline2.png");
    }
#endif

    return status;
}
exit(main());
EOF
# Local Variables:
# mode:c++
# c-basic-offset:4
# End:
