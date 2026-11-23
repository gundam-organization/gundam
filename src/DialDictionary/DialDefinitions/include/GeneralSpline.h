//
// Created by Adrien Blanchet on 23/01/2023.
//

/// @file GeneralSpline.h
/// @brief Declares GeneralSpline, a cubic Hermite spline dial with given knot
///        slopes on knots at arbitrary positions.
/// @ingroup DialDictionary

#ifndef GUNDAM_GENERALSPLINE_H
#define GUNDAM_GENERALSPLINE_H

#include "DialBase.h"
#include "DialUtils.h"
#include "DialInputBuffer.h"

#include "DialUtils.h"

#include <vector>
#include <utility>


/// Evaluates a one-parameter response as a cubic Hermite spline through
/// knots at arbitrary increasing positions, using a value and a slope stored
/// for each knot.
///
/// A dial is the response function of one or more fit parameters: it turns
/// the current parameter value into a weight factor for an event or a bin.
/// DialCollection::makeSplineDial() creates a GeneralSpline for the same
/// spline types as UniformSpline (anything but `catmull-rom`, `pixar` and
/// `ROOT`, with more than two points) when DialUtils::isUniform() finds the
/// knots not uniformly spaced. The dial is then stored in its
/// DialCollection.
///
/// @par Interpolation
/// Between two knots the curve is the cubic Hermite polynomial fixed by the
/// two knot values and the two knot slopes. The slopes are used exactly as
/// given in the input DialUtils::DialPoint list (see UniformSpline for where
/// DialCollection::makeSplineDial() gets them, including the optional
/// `monotonic` limiting). This class adds no slope condition of its own. The
/// curve passes through every knot and has a continuous first derivative.
///
/// @par Knot spacing
/// Any spacing is allowed, as long as the positions strictly increase. The
/// position of every knot is stored, and the kernel finds the interval with
/// a fixed binary search.
///
/// @par Data layout
/// `_splineData_` (returned by getDialData()) holds:
/// - `[0]`: position of the first knot (not read by the kernel);
/// - `[1]`: mean knot spacing (not read by the kernel; kept so the layout
///   starts like the other spline dials);
/// - `[2 + 3 i]`: value of knot `i`;
/// - `[2 + 3 i + 1]`: slope dy/dx at knot `i`, per unit of the parameter;
/// - `[2 + 3 i + 2]`: position of knot `i`.
///
/// This is the layout read by CalculateGeneralSpline(), which is shared with
/// the GPU weight calculator Cache::Weight::GeneralSpline.
///
/// @par Knot count
/// CalculateGeneralSpline() documents a maximum of 15 knots, and
/// Cache::Weight::GeneralSpline::AddSpline() throws for more than 15 (or
/// fewer than 3) knots. buildDial() does not check the count. On the CPU,
/// the kernel's binary search (steps 16, 8, 4, 2, 1) cannot reach an
/// interval past the 32nd.
///
/// @par Outside the knot range
/// When extrapolation is not allowed (the class default), the input is
/// clamped to the first or last knot position, so the response stays at the
/// end-knot value. When it is allowed, the kernel keeps evaluating the cubic
/// polynomial of the first (last) interval: the extrapolation is cubic, not
/// linear.
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
/// @see UniformSpline, CompactSpline
/// @ingroup DialDictionary
class GeneralSpline : public DialBase {

public:
  /// Creates an empty spline. Call buildDial() before evaluating it.
  GeneralSpline() = default;

  // mandatory overrides

  /// Copies this dial, including its knot data and extrapolation flag.
  /// Used by the polymorphic dial container (DialBaseObject) when dials are
  /// copied.
  /// @return A new GeneralSpline equal to this one, owned by the caller.
  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<GeneralSpline>(*this); }

  /// Gives the type name of this dial. Cache::Manager::Build() uses it to
  /// count the dials of each type and size its GPU buffers.
  /// @return `"GeneralSpline"`.
  [[nodiscard]] std::string getDialTypeName() const override { return {"GeneralSpline"}; }

  /// Evaluates the Hermite spline at the current parameter value.
  /// @param input_ Input buffer of the dial. Only element 0, the parameter
  ///        value after any mirroring, is read.
  /// @return The interpolated response. It is not clamped here; the response
  ///         limits of the dial set are applied by DialResponseSupervisor.
  /// @pre buildDial() has been called.
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;
  /// Gives the position stored with each knot in `_splineData_`. Used by
  /// the Python interface through DialInterface::getSourceKnotPositions().
  /// @return One `x` per knot, in increasing order; empty before buildDial().
  [[nodiscard]] std::vector<double> getSourceKnotPositions() const override;

  // other overrides

  /// Sets whether evalResponse() may extrapolate beyond the first and last
  /// knot. DialCollection::makeDial() calls it with the dial set's
  /// `allowDialExtrapolation` option, which is true when not configured.
  /// @param allowExtrapolation If false, the input is clamped to the knot
  ///        range. If true, the end intervals' cubics are continued beyond it.
  void setAllowExtrapolation(bool allowExtrapolation) override { _allowExtrapolation_ = allowExtrapolation; }

  /// Tells whether evalResponse() extrapolates beyond the knot range.
  /// @return The value given to setAllowExtrapolation(); false by default.
  [[nodiscard]] bool getAllowExtrapolation() const override { return _allowExtrapolation_; }

  /// Gives the packed knot data (layout in the class description).
  /// Cache::Manager copies it to the GPU.
  /// @return A reference to `_splineData_`; empty before buildDial().
  [[nodiscard]] const std::vector<double>& getDialData() const override { return _splineData_; }

  // methods
  /// Gives the range covered by the knots, from the first to the last `x`.
  ///
  /// DialCollection appends it to getDialData() when it compares dials, so
  /// that identical dials can share one object
  /// (DialCollection::deduplicateBinnedDials(), DialCollection::storeEventDial()).
  /// @return The knot range recorded by buildDial().
  [[nodiscard]] const GenericToolbox::Range& getSplineBounds() const { return _splineBounds_; }


  /// Packs the position, value and slope of every knot into `_splineData_`
  /// and records the knot range.
  ///
  /// DialCollection::makeSplineDial() calls it after sorting the points
  /// (DialCollection::checkDialPointList()) and filling their slopes.
  ///
  /// @param splinePointList_ Knots sorted by strictly increasing `x`, at
  ///        least two of them, with `slope` set to dy/dx at each knot.
  /// @pre The dial is empty; otherwise it throws ("Spline data already set.").

  void buildDial(const std::vector<DialUtils::DialPoint>& splinePointList_);

protected:
  /// Whether evalResponse() extrapolates outside the knot range (true) or
  /// clamps its input to that range (false).
  bool _allowExtrapolation_{false};

  /// A block of data to calculate the spline values.  This must be filled for
  /// the Cache::Manager to work, and provides the input for spline calculation
  /// functions that can be shared between the CPU and the GPU.
  /// Holds two header values, then a (value, slope, position) triple per
  /// knot (see the class description).
  std::vector<double> _splineData_{};

  /// Positions of the first and last knot. evalResponse() clamps its input to
  /// this range when extrapolation is not allowed. NaN until buildDial().
  GenericToolbox::Range _splineBounds_{std::nan("unset"), std::nan("unset")};
};


#endif //GUNDAM_GENERALSPLINE_H
