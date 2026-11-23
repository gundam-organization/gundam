//
// Created by Adrien Blanchet on 22/01/2023.
//

/// @file MonotonicSpline.h
/// @brief Declares MonotonicSpline, a Catmull-Rom spline dial on uniformly
///        spaced knots with Fritsch-Carlson slope limiting.
/// @ingroup DialDictionary

#ifndef GUNDAM_MONOTONICSPLINE_H
#define GUNDAM_MONOTONICSPLINE_H

#include "DialBase.h"
#include "DialUtils.h"
#include "DialInputBuffer.h"

#include "DialUtils.h"

#include "TGraph.h"

#include <vector>
#include <utility>

/// Evaluates a one-parameter response as a monotonic Catmull-Rom spline
/// through uniformly spaced knots.
///
/// A dial is the response function of one or more fit parameters: it turns
/// the current parameter value into a weight factor for an event or a bin.
/// DialCollection::makeSplineDial() creates a MonotonicSpline for `Spline`
/// dial sets whose options contain both `catmull-rom` (or `pixar`) and
/// `monotonic`, when the input has more than two points (two-point inputs
/// become a CompactSpline). The dial is then stored in its DialCollection.
///
/// @par Interpolation
/// The same as CompactSpline, except for the slopes. Only the knot values
/// are stored. Between two knots the curve is the cubic Hermite polynomial
/// fixed by the two knot values and the two knot slopes. The slope at a knot
/// starts as the Catmull-Rom estimate (mean of the two neighbouring interval
/// slopes; at the first and last knot, the end interval's slope) and is then
/// limited, at evaluation time, with the Fritsch-Carlson conditions:
/// - it is set to zero when the two neighbouring interval slopes have
///   opposite signs or one of them is zero (local extremum or flat part);
/// - its magnitude is limited to three times the smaller of the two
///   neighbouring interval slopes, in magnitude.
///
/// @par Monotonicity
/// With these limits the curve stays between the two knot values on every
/// interval (no overshoot), so monotonic knot values give a monotonic
/// response, and a local extremum of the knot values is a flat point of the
/// curve. The curve passes through every knot and has a continuous first
/// derivative. The `slope` of the input DialUtils::DialPoint list is
/// ignored.
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
/// This is the layout read by CalculateMonotonicSpline(), which is shared
/// with the GPU weight calculator Cache::Weight::MonotonicSpline.
///
/// @par Outside the knot range
/// When extrapolation is not allowed (the class default), the input is
/// clamped to the first or last knot position, so the response stays at the
/// end-knot value. When it is allowed, the response continues as a straight
/// line along the first (last) interval, which keeps it monotonic.
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
/// @see CompactSpline, DialUtils::applyMonotonicCondition()
/// @ingroup DialDictionary
class MonotonicSpline : public DialBase {

public:
  /// Creates an empty spline. Call buildDial() before evaluating it.
  MonotonicSpline() = default;

  // mandatory overrides

  /// Copies this dial, including its knot data and extrapolation flag.
  /// Used by the polymorphic dial container (DialBaseObject) when dials are
  /// copied.
  /// @return A new MonotonicSpline equal to this one, owned by the caller.
  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<MonotonicSpline>(*this); }

  /// Gives the type name of this dial. Cache::Manager::Build() uses it to
  /// count the dials of each type and size its GPU buffers.
  /// @return `"MonotonicSpline"`.
  [[nodiscard]] std::string getDialTypeName() const override { return {"MonotonicSpline"}; }

  /// Evaluates the monotonic Catmull-Rom spline at the current parameter
  /// value.
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
  void setAllowExtrapolation(bool allowExtrapolation) override{ _allowExtrapolation_ = allowExtrapolation; }

  /// Tells whether evalResponse() extrapolates beyond the knot range.
  /// @return The value given to setAllowExtrapolation(); false by default.
  [[nodiscard]] bool getAllowExtrapolation() const override { return _allowExtrapolation_; }

  // methods
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

  /// Gives the packed knot data (layout in the class description).
  /// Cache::Manager copies it to the GPU.
  /// @return A reference to `_splineData_`; empty before buildDial().
  [[nodiscard]] const std::vector<double>& getDialData() const override {return _splineData_;}

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
  GenericToolbox::Range _splineBounds_{};
};


#endif //GUNDAM_MONOTONICSPLINE_H
