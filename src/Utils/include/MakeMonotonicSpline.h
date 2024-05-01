#ifndef MakeMonotonicSpline_h_seen
#define MakeMonotonicSpline_h_seen
namespace {
    // Place this in a namespace inside the anonymous namespace!
    namespace util {

        // Apply the Fritsh-Carlson monotonic condition to the slopes.  This
        // adjusts the slopes (when necessary), however, with Catmull-Rom the
        // modified slopes will be ignored and the monotonic criteria is
        // applied as the spline is evaluated (saves memory).
        //
        // F.N.Fritsch, and R.E.Carlson, "Monotone Piecewise Cubic
        // Interpolation" SIAM Journal on Numerical Analysis, Vol. 17, Iss. 2
        // (1980) doi:10.1137/0717021
        void MakeMonotonicSpline(const std::vector<double>& xPoint,
                                 const std::vector<double>& yPoint,
                                 std::vector<double>& slope) {
            for (std::size_t i = 0; i<xPoint.size(); ++i) {
                double m{std::numeric_limits<double>::infinity()};
                if (i>0) {
                    m = (yPoint[i] - yPoint[i-1])/(xPoint[i] - xPoint[i-1]);
                }
                else {
                    m = slope[i];
                }
                double p{std::numeric_limits<double>::infinity()};
                if (i<xPoint.size()-1) {
                    p =(yPoint[i+1]-yPoint[i])/(xPoint[i+1]-xPoint[i]);
                }
                else {
                    p = slope[i];
                }
                double delta = std::min(std::abs(m),std::abs(p));
                // Make sure the slope at a cusp (where the average slope
                // flips sign) is zero.
                if (m*p < 0.0) delta = 0.0;
                // This a conservative bound on when the slope is "safe".
                // It's not the actual value where the spline becomes
                // non-monotonic.
                if (std::abs(slope[i]) > 3.0*delta) {
                    if (slope[i] < 0.0) slope[i] = -3.0*delta;
                    else slope[i] = 3.0*delta;
                }
            }
        }
    }
}

// An MIT Style License

// Copyright (c) 2024 Clark McGrew

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
// compile-command:"$(git rev-parse --show-toplevel)/cmake/scripts/gundam-build.sh"
// End:
#endif
