# !/bin/bash
# Wrap a ROOT macro as a script.
root <<EOF

#include <iostream>
#include <string>
#include <memory>
#include <cmath>

////////////////////////////////////////////////////////////////////////
// Test the CalculateCompactSpline routine on the CPU.

#include "${GUNDAM_ROOT}/src/Utils/include/CalculateBicubicSpline.h"

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
        if (_d < (_tol)) _r = _d;                                 \
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
                TOLERANCE("Symmetry +- (test 1)", v0, v1, 1E-8);
                TOLERANCE("Symmetry -+ (test 1)", v0, v2, 1E-8);
                TOLERANCE("Symmetry -- (test 1)", v0, v3, 1E-8);
                graph1->SetPoint(p++,x,y,v0);
            }
        }
        graph1->Draw("colz");
        gPad->Print("100CheckBicubic1.pdf");
        gPad->Print("100CheckBicubic1.png");
    }
#endif

#define TEST2
#ifdef TEST2
    {
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
                TOLERANCE("Symmetry +- (test 2)", v0, v1, 1E-8);
                TOLERANCE("Symmetry -+ (test 2)", v0, v2, 1E-8);
                TOLERANCE("Symmetry -- (test 2)", v0, v3, 1E-8);
                graph1->SetPoint(p++,x,y,v0);
            }
        }
        graph1->Draw("colz");
        gPad->Print("100CheckBicubic2.pdf");
        gPad->Print("100CheckBicubic2.png");
    }
#endif

#define TEST3
#ifdef TEST3
    {
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
            TOLERANCE("Symmetry -- (test 3)", v0, v2, 1E-8);
            graph1->SetPoint(p++,x,v0);
        }
        graph1->Draw("AC");
        gPad->Print("100CheckBicubic3.pdf");
        gPad->Print("100CheckBicubic3.png");
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
