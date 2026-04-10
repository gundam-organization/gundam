#ifndef FritshCarlsonMonotonicCondition_h_seen
#define FritshCarlsonMonotonicCondition_h_seen
#include <limits>

namespace {
    namespace FritshCarlson {
        // Apply the Fritsh-Carlson monotonic condition to the slopes.  This
        // adjusts the slopes (when necessary).
        //
        // F.N.Fritsch, and R.E.Carlson, "Monotone Piecewise Cubic
        // Interpolation" SIAM Journal on Numerical Analysis, Vol. 17, Iss. 2
        // (1980) doi:10.1137/0717021
        //
        // This is a reimplementation based on the paper, with help from the
        // wikipedia article on Monotonic Cubic Interpolation which clarifies
        // some of the original notation. The aim is to reproduce the
        // mathematics as clearly as possible, and ignore any optimization.
        void MonotonicCondition(
            const std::vector<double>& xPoint,
            const std::vector<double>& yPoint,
            std::vector<double>& slope) {

            // 1) Find the secant line slopes.
            std::vector<double> d(xPoint.size());
            for (std::size_t k = 0; k<xPoint.size()-1; ++k) {
                d[k] = (yPoint[k+1]-yPoint[k])/(xPoint[k+1]-xPoint[k]);
            }

            // 2) Initialize tangent for interior data points.
            std::vector<double> m(xPoint.size());
            for (std::size_t k=1; k<xPoint.size()-1; ++k) {
                m[k] = (d[k-1]+d[k])/2.0;
            }
            m[0] = d[0];
            m[xPoint.size()-1] = d[xPoint.size()-2];

            // 2.A) Set tangents to flat if there is a cusp which is true when
            // ds have opposite signs.
            for (std::size_t k=1; k<xPoint.size(); ++k) {
                if (d[k-1]*d[k] < 0.0) m[k] = 0.0;
            }

            // 3) When the successive points are equal, set the slopes to
            // zero.  This is when d is zero.
            for (std::size_t k = 0; k<xPoint.size()-1; ++k) {
                if (std::abs(d[k]) < std::numeric_limits<double>::epsilon()) {
                    m[k] = 0.0;
                    m[k+1] = 0.0;
                }
            }

            // 4.A) Calculate alpha
            std::vector<double> alpha(xPoint.size());
            for (std::size_t k = 0; k < xPoint.size(); ++k) {
                alpha[k] = 0.0;
                if (std::abs(d[k])
                    > std::numeric_limits<double>::epsilon()) {
                    alpha[k] = m[k]/d[k];
                }
            }

            // 4.B) Calculate beta
            std::vector<double> beta(xPoint.size());
            for (std::size_t k = 0; k < xPoint.size(); ++k) {
                beta[k] = 0.0;
                if (std::abs(d[k]) > std::numeric_limits<double>::epsilon()) {
                    if (k < xPoint.size()-1) beta[k] = m[k+1]/d[k];
                }
            }

            // 4.C) When alpha is zero slope is zero
            for (std::size_t k = 0; k < xPoint.size(); ++k) {
                if (alpha[k] < 0.0) m[k] = 0.0;
            }

            // 4.D) When beta is zero next slope is zero.
            for (std::size_t k = 0; k < xPoint.size()-1; ++k) {
                if (beta[k] < 0.0) m[k+1] = 0.0;
            }

            // 5) Prevent overshoot This applies the criteria for set 1 in the
            // Fritzch-Carlson paper (figure 2).
            double bound = 3.0;
            for (std::size_t k = 0; k < xPoint.size(); ++k) {
                if (alpha[k] > bound) m[k] = bound*d[k];
            }
            for (std::size_t k = 0; k < xPoint.size()-1; ++k) {
                if (beta[k] > bound) m[k+1] = bound*d[k];
            }

            // 6) Fix the slopes
            for (std::size_t k = 0; k < xPoint.size(); ++k) {
                double s = slope[k];
                // Limit the range of the slope
                if (slope[k] > 0.0 and slope[k] > m[k]) s = m[k];
                if (slope[k] < 0.0 and slope[k] < m[k]) s = m[k];
                // If the slope changed, copy it back into the output.
                if (std::abs(s-slope[k])
                    > std::numeric_limits<double>::epsilon()) {
                    slope[k] = s;
                }
            }
        }
    }
}

// An MIT Style License

// Copyright (c) 2026 Clark McGrew

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Local Variables:
// mode:c++
// c-basic-offset:4
// End:
#endif
