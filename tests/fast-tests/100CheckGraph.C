# !/bin/bash
# Wrap a ROOT macro as a script.
root <<EOF

#include <iostream>
#include <string>
#include <memory>
#include <cmath>

////////////////////////////////////////////////////////////////////////
// Test the CalculateCompactSpline routine on the CPU.

#include "${GUNDAM_ROOT}/src/Utils/include/CalculateGraph.h"

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
        for (double x = -1.0; x <= 2.0; x += 0.1) {
            double v = CalculateGraph(x, -10.0, 10.0, data, 2*nData);
            TOLERANCE("Two Point Tolerance", x, v, 1E-6);
            graph1->SetPoint(p++,x,v);
        }
        graph1->Draw("AC");
        data1->Draw("*,same");
        gPad->Print("100CheckGraph1.pdf");
        gPad->Print("100CheckGraph1.png");
    }
#endif

#define TEST2
#ifdef TEST2
    {
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
            std::ostringstream tmp;
            tmp << "Symmetric tolerance (test 2) (X=" << x << ")";
            TOLERANCE(tmp.str(), v0, v1, 1E-6);
            graph1->SetPoint(p++,x,v0);
        }
        graph1->Draw("AC");
        data1->Draw("*,same");
        gPad->Print("100CheckGraph2.pdf");
        gPad->Print("100CheckGraph2.png");
    }
#endif

#define TEST3
#ifdef TEST3
    {
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
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
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
            std::ostringstream tmp;
            tmp << "Symmetric tolerance (test 4) (X=" << x << ")";
            TOLERANCE(tmp.str(), v0, v1, 1E-6);
            graph1->SetPoint(p++,x,v0);
        }
        graph1->Draw("AC");
        data1->Draw("*,same");
        gPad->Print("100CheckGraph3.pdf");
        gPad->Print("100CheckGraph3.png");
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
