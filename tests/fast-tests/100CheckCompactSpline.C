# !/bin/bash
# Wrap a ROOT macro as a script.
root <<EOF

#include <iostream>
#include <string>
#include <memory>
#include <cmath>

////////////////////////////////////////////////////////////////////////
// Test the CalculateCompactSpline routine on the CPU.

#include "${GUNDAM_ROOT}/src/Utils/include/CalculateCompactSpline.h"

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

    {
        double data[] = {0.0, 1.0, 0.0, 1.0};
        // Test linear interpolation between two points
        int p = 0;
        for (double x = -1.0; x <= 2.0; x += 0.1) {
            double v = CalculateCompactSpline(x, -10.0, 10.0, data, 2);
            TOLERANCE("Two Point Tolerance", x, v, 1E-6);
        }
    }

    {
        // Test non-linear interpolation between three points
        double data[] = {-1.0, 1.0, 0.0, 1.0, 0.0};
        int nData = 3;
        std::unique_ptr<TGraph> data1(new TGraph());
        for (int p=0; p<nData; ++p) {
            double x = data[0] + p*data[1];
            double y = data[p+2];
            data1->SetPoint(p,x,y);
        }
        std::unique_ptr<TGraph> graph1(new TGraph());
        int p = 0;
        for (double x = -1.1; x <= 1.1; x += 0.1) {
            double v0 = CalculateCompactSpline(x, -10.0, 10.0, data, nData);
            double v1 = CalculateCompactSpline(-x, -10.0, 10.0, data, nData);
            std::ostringstream tmp;
            tmp << "Symmetric tolerance (test 2) (X=" << x << ")";
            TOLERANCE(tmp.str(), v0, v1, 1E-6);
            graph1->SetPoint(p++,x,v0);
        }
        data1->Draw("A*");
        graph1->Draw("C,same");
        gPad->Print("100CheckCompactSpline2.pdf");
        gPad->Print("100CheckCompactSpline2.png");
    }

    {
        // Test interpolation between six points
        int nData = 6;
        double data[] = {-1.0, 2.0/(nData-1), 0.0, 0.0, 1.0, 1.0, 0.0, 0.0};
        std::unique_ptr<TGraph> data1(new TGraph());
        for (int p=0; p<nData; ++p) {
            double x = data[0] + p*data[1];
            double y = data[p+2];
            data1->SetPoint(p,x,y);
        }
        std::unique_ptr<TGraph> graph1(new TGraph());
        int p = 0;
        for (double x = -1.1; x <= 1.1; x += 0.01) {
            double v0 = CalculateCompactSpline(x, -10.0, 10.0, data, nData);
            double v1 = CalculateCompactSpline(-x, -10.0, 10.0, data, nData);
            std::ostringstream tmp;
            tmp << "Symmetric tolerance (test 3) (X=" << x << ")";
            TOLERANCE(tmp.str(), v0, v1, 1E-6);
            graph1->SetPoint(p++,x,v0);
            graph1->SetPoint(p++,x,v0);
        }
        data1->Draw("A*");
        graph1->Draw("C,same");
        gPad->Print("100CheckCompactSpline3.pdf");
        gPad->Print("100CheckCompactSpline3.png");
    }

    {
        // Test interpolation where there can be a lot of overshoot
        int nData = 13;
        double data[] = {-1.0, 2.0/(nData-1),
                         0.5, 1.5,
                         1.0, 1.0, 1.0, 1.0,
                         0.5,
                         1.0, 1.0, 1.0, 1.0,
                         1.5, 0.5};
        std::unique_ptr<TGraph> data1(new TGraph());
        for (int p=0; p<nData; ++p) {
            double x = data[0] + p*data[1];
            double y = data[p+2];
            data1->SetPoint(p,x,y);
        }
        std::unique_ptr<TGraph> graph1(new TGraph());
        int p = 0;
        for (double x = -1.1; x <= 1.1; x += 0.01) {
            double v0 = CalculateCompactSpline(x, -10.0, 10.0, data, nData);
            double v1 = CalculateCompactSpline(-x, -10.0, 10.0, data, nData);
            std::ostringstream tmp;
            tmp << "Symmetric tolerance (test 4) (X=" << x << ")";
            TOLERANCE(tmp.str(), v0, v1, 1E-6);
            graph1->SetPoint(p++,x,v0);
        }
        data1->Draw("A*");
        graph1->Draw("C,same");
        gPad->Print("100CheckCompactSpline4.pdf");
        gPad->Print("100CheckCompactSpline4.png");
    }

    return status;
}
exit(main());
EOF
# Local Variables:
# mode:c++
# c-basic-offset:4
# End:
