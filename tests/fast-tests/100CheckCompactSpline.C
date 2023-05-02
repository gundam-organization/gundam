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
        // Test linear interpolation between two points
        double data[] = {0.0, 1.0, 0.0, 1.0};
        for (double x = 0.0; x <= 2.0; x += 0.1) {
            double v = CalculateCompactSpline(x, 0.0, 10.0, data, 2);
            TOLERANCE("Two Point Tolarance", x, v, 1E-6);
            std::cout << x << " " << v << std::endl;
        }
    }

    {
        // Test non-linear interpolation between three points
        double data[] = {-1.0, 1.0, 0.0, 1.0, 0.0};
        for (double x = 0.0; x <= 1.0; x += 0.1) {
            double v0 = CalculateCompactSpline(x, 0.0, 10.0, data, 3);
            double v1 = CalculateCompactSpline(-x, 0.0, 10.0, data, 3);
            TOLERANCE("Symmetric Tolarance", v0, v1, 1E-6);
            std::cout << x << " " << v0 << " " << v1 << std::endl;
        }
    }

    return status;
}
exit(main());
EOF
# Local Variables:
# mode:c++
# c-basic-offset:4
# End:
