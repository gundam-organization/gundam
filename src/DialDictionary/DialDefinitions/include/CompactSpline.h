//
// Created by Adrien Blanchet on 22/01/2023.
//

/// @file CompactSpline.h
/// @brief Declares CompactSpline, a Catmull-Rom spline dial on uniformly
///        spaced knots that stores only the knot values.
/// @ingroup DialDictionary

#ifndef GUNDAM_COMPACTSPLINE_H
#define GUNDAM_COMPACTSPLINE_H

#include "DialBase.h"
#include "DialUtils.h"
#include "DialInputBuffer.h"

#include "DialUtils.h"

#include "TGraph.h"

#include <vector>
#include <utility>

/// Evaluates a one-parameter response as a Catmull-Rom cubic spline through
/// uniformly spaced knots.
///
/// A dial is the response function of one or more fit parameters: it turns
/// the current parameter value into a weight factor for an event or a bin.
/// DialCollection::makeSplineDial() creates a CompactSpline for `Spline`
/// dial sets whose options select `catmull-rom` (or `pixar`) without
/// `monotonic`. Unless `ROOT` is requested, it also uses this class for every
/// spline input that has only two points. The dial is then stored in its
/// DialCollection.
///
/// @par Interpolation
/// Only the knot values are stored; the slopes are computed when the spline
/// is evaluated. Between two knots the curve is the cubic Hermite polynomial
/// fixed by the two knot values and the two knot slopes. The slope at an
/// interior knot `i` is the Catmull-Rom estimate `(y[i+1] - y[i-1]) / (2 h)`,
/// with `h` the knot spacing. The slope at the first (last) knot is the slope
/// of the first (last) interval. The curve passes through every knot and has
/// a continuous first derivative, but it is not monotonic in general and may
/// overshoot the knot values (use MonotonicSpline to avoid that). The
/// `slope` of the input DialUtils::DialPoint list is ignored.
///
/// @par Knot spacing
/// The knots must be uniformly spaced. Only the first knot position and the
/// mean spacing are stored, so knot `i` is always placed at `x0 + i h`,
/// whatever its actual position. buildDial() checks the spacing.
///
/// @par Data layout
/// `_splineData_` (returned by getDialData()) holds:
/// - `[0]`: position `x0` of the first knot;
/// - `[1]`: knot spacing `h` = (last x - first x) / (number of knots - 1);
/// - `[2 + i]`: value of knot `i`, for `i` = 0 .. number of knots - 1.
///
/// This is the layout read by CalculateCompactSpline(), which is shared with
/// the GPU weight calculator Cache::Weight::CompactSpline.
///
/// @par Outside the knot range
/// When extrapolation is not allowed (the class default), the input is
/// clamped to the first or last knot position, so the response stays at the
/// end-knot value. When it is allowed, the response continues as a straight
/// line along the first (last) interval.
/// Dials made by DialCollection::makeDial() extrapolate unless the dial
/// set sets `allowDialExtrapolation` to false.
/// @note The Cache::Manager GPU path runs the same kernel on getDialData()
///       but passes the parameter value unclamped: it never checks the
///       extrapolation flag, so it always extrapolates.
///
/// @par Thread safety
/// Once built, the dial is only read by evalResponse(), so it can be
/// evaluated from several threads at once.
///
/// @see MonotonicSpline, UniformSpline, GeneralSpline
/// @ingroup DialDictionary
class CompactSpline : public DialBase {

public:
  /// Creates an empty spline. Call buildDial() before evaluating it.
  CompactSpline() = default;

  // mandatory overrides

  /// Copies this dial, including its knot data and extrapolation flag.
  /// Used by the polymorphic dial container (DialBaseObject) when dials are
  /// copied.
  /// @return A new CompactSpline equal to this one, owned by the caller.
  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<CompactSpline>(*this); }

  /// Gives the type name of this dial. Cache::Manager::Build() uses it to
  /// count the dials of each type and size its GPU buffers.
  /// @return `"CompactSpline"`.
  [[nodiscard]] std::string getDialTypeName() const override { return {"CompactSpline"}; }

  /// Evaluates the Catmull-Rom spline at the current parameter value.
  /// @param input_ Input buffer of the dial. Only element 0, the parameter
  ///        value after any mirroring, is read.
  /// @return The interpolated response. It is not clamped here; the response
  ///         limits of the dial set are applied by DialResponseSupervisor.
  /// @pre buildDial() has been called.
  /// @note In debug builds (NDEBUG not defined) a non-finite input throws.
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;
  /// Gives the knot positions, rebuilt as `x0 + i h` from the first two
  /// entries of `_splineData_`. Used by the Python interface through
  /// DialInterface::getSourceKnotPositions().
  /// @return One `x` per knot; empty before buildDial(). Rebuilt positions
  ///         can differ from the input knots by rounding.
  [[nodiscard]] std::vector<double> getSourceKnotPositions() const override;

  // other overrides

  /// Sets whether evalResponse() may extrapolate beyond the first and last
  /// knot. DialCollection::makeDial() calls it with the dial set's
  /// `allowDialExtrapolation` option, which is true when not configured.
  /// @param allowExtrapolation If false, the input is clamped to the knot
  ///        range. If true, the response continues linearly beyond it.
  void setAllowExtrapolation(bool allowExtrapolation) override { _allowExtrapolation_ = allowExtrapolation; }

  /// Tells whether evalResponse() extrapolates beyond the knot range.
  /// @return The value given to setAllowExtrapolation(); false by default.
  [[nodiscard]] bool getAllowExtrapolation() const override { return _allowExtrapolation_; }

  /// Gives the packed knot data (layout in the class description).
  /// Cache::Manager copies it to the GPU.
  /// @return A reference to `_splineData_`; empty before buildDial().
  [[nodiscard]] const std::vector<double>& getDialData() const override { return _splineData_; }

  /// Describes the content of the dial for debug printouts. Called by
  /// DialInterface::getSummary() when a detailed summary is requested.
  /// @return Three lines giving the packed spline data, the knot range and
  ///         the extrapolation flag.
  [[nodiscard]] std::string getSummary() const override;

  /// Declared to fill the spline data from a TGraph, but never defined.
  /// @param graph_ Graph that would provide the knots.
  /// @warning There is no definition anywhere in the code base and nothing
  ///          calls it; a call would fail at link time. Use buildDial().
  void buildSplineData(TGraph& graph_);

  /// Declared to evaluate the spline, but never defined.
  /// @param input_ Input buffer that would provide the parameter value.
  /// @return Nothing: the function has no definition.
  /// @warning There is no definition anywhere in the code base and nothing
  ///          calls it; a call would fail at link time. Use evalResponse().
  [[nodiscard]] double evaluateSpline(const DialInputBuffer& input_) const;

  /// Gives the range covered by the knots, from the first to the last `x`.
  ///
  /// DialCollection appends it to getDialData() when it compares dials, so
  /// that identical dials can share one object
  /// (DialCollection::deduplicateBinnedDials(), DialCollection::storeEventDial()).
  /// @return The knot range recorded by buildDial().
  [[nodiscard]] const GenericToolbox::Range& getSplineBounds() const { return _splineBounds_; }

  /// Packs the knots into `_splineData_` and records the knot range.
  ///
  /// Only the first and last `x`, the number of points and every `y` are
  /// used; the `slope` of each point is ignored. DialCollection::makeSplineDial()
  /// calls it after sorting the points (DialCollection::checkDialPointList()).
  ///
  /// @param splinePointList_ Knots sorted by increasing `x`, at least two of
  ///        them, uniformly spaced.
  /// @pre The dial is empty; otherwise it throws ("Spline data already set.").
  /// @warning If the spacing is not uniform (a knot more than 1% of a spacing
  ///          away from `x0 + i h`, or a spacing below 1e-6), every knot is
  ///          logged as an error. Debug builds (NDEBUG not defined) then
  ///          throw; release builds keep the dial and treat the knots as
  ///          uniformly spaced.

  void buildDial(const std::vector<DialUtils::DialPoint>& splinePointList_);


protected:
  /// Whether evalResponse() extrapolates outside the knot range (true) or
  /// clamps its input to that range (false).
  bool _allowExtrapolation_{false};

  /// A block of data to calculate the spline values.  This must be filled for
  /// the Cache::Manager to work, and provides the input for spline calculation
  /// functions that can be shared between the CPU and the GPU.
  /// Holds `x0`, `h`, then one value per knot (see the class description).
  std::vector<double> _splineData_{};

  /// Positions of the first and last knot. evalResponse() clamps its input to
  /// this range when extrapolation is not allowed. NaN until buildDial().
  GenericToolbox::Range _splineBounds_{std::nan("unset"), std::nan("unset")};
};


#endif //GUNDAM_COMPACTSPLINE_H
