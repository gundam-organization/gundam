# !/bin/bash
# Wrap a ROOT macro as a script.
root -n <<EOF

#include <iostream>
#include <string>
#include <memory>
#include <cmath>

#include <TSpline.h>

////////////////////////////////////////////////////////////////////////
// Test the CalculateGeneralSpline routine on the CPU.

#include "${GUNDAM_ROOT}/src/Utils/include/CalculateGeneralSpline.h"
#include "${GUNDAM_ROOT}/src/Utils/include/MakeMonotonicSpline.h"

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
        int point = 0;
        graph.SetPoint(point++,-3.0, 0.0);
        graph.SetPoint(point++, 0.0, 1.0);
        graph.SetPoint(point++, 1.0, 2.0);
        graph.SetPoint(point++, 2.0, 4.0);
        graph.SetPoint(point++, 3.0, 5.0);
        graph.SetPoint(point++, 4.4, 5.5);
        graph.SetPoint(point++, 5.1, 4.5);
        graph.SetPoint(point++, 6.0, 5.5);
        TSpline3 spline("splineOfGraph",&graph);
        graph.SetLineStyle(2);
        graph.Draw("AL*");
        graph.SetMinimum(-1.0);
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
        for (int knot = 0; knot < nKnots; ++knot) {
            data[2+3*knot+0] = yValues[knot];
            data[2+3*knot+1] = slopes[knot];
            data[2+3*knot+2] = xValues[knot];
        }
        TGraph monotonicSpline;
        ::util::MakeMonotonicSpline(xValues,yValues,slopes);
        for (int knot = 0; knot < nKnots; ++knot) {
            data[2+3*knot+0] = yValues[knot];
            data[2+3*knot+1] = slopes[knot];
            data[2+3*knot+2] = xValues[knot];
        }
        point = 0;
        double last{-std::numeric_limits<double>::infinity()};
        int flips = 0;
        int sign = 1;
        for (double xxx = spline.GetXmin(); xxx<=spline.GetXmax(); xxx += 0.01) {
            double splineValue = spline.Eval(xxx);
            double calcValue
                = CalculateGeneralSpline(xxx,-100.0, 100.0,
                                         data, dim);
            if (sign > 0 and last > calcValue) {
                ++flips;
                sign = -1;
            }
            else if (sign < 0 and last < calcValue) {
                ++flips;
                sign = 1;
            }
            monotonicSpline.SetPoint(point++, xxx, calcValue);
            last = calcValue;
        }
        TOLERANCE("Test1: Wrong number of flips in spline", flips, 2, 0.1);
        monotonicSpline.SetLineWidth(2);
        monotonicSpline.Draw("same");
        gPad->Print("100CheckMakeMonotonicSpline1.pdf");
        gPad->Print("100CheckMakeMonotonicSpline1.png");
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
