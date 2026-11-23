//
// Created by Adrien Blanchet on 23/01/2023.
//

/// @file UniformSpline.h
/// @brief Declares UniformSpline, a cubic Hermite spline dial with given knot
///        slopes on uniformly spaced knots.
/// @ingroup DialDictionary

#ifndef GUNDAM_UNIFORMSPLINE_H
#define GUNDAM_UNIFORMSPLINE_H


#include "DialBase.h"
#include "DialUtils.h"
#include "DialInputBuffer.h"

#include "DialUtils.h"

#include <vector>
#include <utility>


/// Evaluates a one-parameter response as a cubic Hermite spline through
/// uniformly spaced knots, using a value and a slope stored for each knot.
///
/// A dial is the response function of one or more fit parameters: it turns
/// the current parameter value into a weight factor for an event or a bin.
/// DialCollection::makeSplineDial() creates a UniformSpline for `Spline`
/// dial sets that request neither `catmull-rom` (or `pixar`) nor `ROOT`,
/// when the input has more than two points and DialUtils::isUniform() finds
/// them uniformly spaced (tolerance from the `uniformity(...)` option,
/// 16 float epsilons by default). Otherwise it uses GeneralSpline. The dial
/// is then stored in its DialCollection.
///
/// @par Interpolation
/// Between two knots the curve is the cubic Hermite polynomial fixed by the
/// two knot values and the two knot slopes. The slopes are used exactly as
/// given in the input DialUtils::DialPoint list: DialCollection::makeSplineDial()
/// takes them from the knot derivatives of a ROOT TSpline3 (the input itself,
/// or one built from an input TGraph) or from DialUtils::fillAkimaSlopes(),
/// and limits them with DialUtils::applyMonotonicCondition() when
/// `monotonic` is requested. This class adds no slope condition of its own,
/// so it is monotonic only if the given slopes make it so. The curve passes
/// through every knot and has a continuous first derivative.
///
/// @par Knot spacing
/// The knots must be uniformly spaced. Only the first knot position and the
/// mean spacing are stored, so knot `i` is always placed at `x0 + i h`.
/// buildDial() throws if the spacing is not uniform.
///
/// @par Data layout
/// `_splineData_` (returned by getDialData()) holds:
/// - `[0]`: position `x0` of the first knot;
/// - `[1]`: knot spacing `h` = (last x - first x) / (number of knots - 1);
/// - `[2 + 2 i]`: value of knot `i`;
/// - `[2 + 2 i + 1]`: slope dy/dx at knot `i`, per unit of the parameter.
///
/// This is the layout read by CalculateUniformSpline(), which is shared with
/// the GPU weight calculator Cache::Weight::UniformSpline.
///
/// @par Outside the knot range
/// When extrapolation is not allowed (the class default), the input is
/// clamped to the first or last knot position, so the response stays at the
/// end-knot value. When it is allowed, the kernel keeps evaluating the cubic
/// polynomial of the first (last) interval: the extrapolation is cubic, not
/// linear, unlike CompactSpline and MonotonicSpline.
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
/// @see GeneralSpline, CompactSpline
/// @ingroup DialDictionary
class UniformSpline : public DialBase {

public:
  /// Creates an empty spline. Call buildDial() before evaluating it.
  UniformSpline() = default;

  // mandatory overrides

  /// Copies this dial, including its knot data and extrapolation flag.
  /// Used by the polymorphic dial container (DialBaseObject) when dials are
  /// copied.
  /// @return A new UniformSpline equal to this one, owned by the caller.
  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<UniformSpline>(*this); }

  /// Gives the type name of this dial. Cache::Manager::Build() uses it to
  /// count the dials of each type and size its GPU buffers.
  /// @return `"UniformSpline"`.
  [[nodiscard]] std::string getDialTypeName() const override { return {"UniformSpline"}; }

  /// Evaluates the Hermite spline at the current parameter value.
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
  ///        range. If true, the end intervals' cubics are continued beyond it.
  void setAllowExtrapolation(bool allowExtrapolation) override { _allowExtrapolation_ = allowExtrapolation; }

  /// Tells whether evalResponse() extrapolates beyond the knot range.
  /// @return The value given to setAllowExtrapolation(); false by default.
  [[nodiscard]] bool getAllowExtrapolation() const override { return _allowExtrapolation_; }

  /// Gives the packed knot data (layout in the class description).
  /// Cache::Manager copies it to the GPU.
  /// @return A reference to `_splineData_`; empty before buildDial().
  [[nodiscard]] const std::vector<double>& getDialData() const override { return _splineData_; }

  /// Gives the range covered by the knots, from the first to the last `x`.
  ///
  /// DialCollection appends it to getDialData() when it compares dials, so
  /// that identical dials can share one object
  /// (DialCollection::deduplicateBinnedDials(), DialCollection::storeEventDial()).
  /// @return The knot range recorded by buildDial().
  [[nodiscard]] const GenericToolbox::Range& getSplineBounds() const { return _splineBounds_; }

  /// Packs the knot values and slopes into `_splineData_` and records the
  /// knot range.
  ///
  /// Only the first and last `x`, the number of points, and every `y` and
  /// `slope` are stored. DialCollection::makeSplineDial() calls it after
  /// sorting the points (DialCollection::checkDialPointList()) and filling
  /// their slopes.
  ///
  /// @param splinePointList_ Knots sorted by increasing `x`, at least two of
  ///        them, uniformly spaced, with `slope` set to dy/dx at each knot.
  /// @pre The dial is empty; otherwise it throws ("Spline data already set.").
  /// @warning Throws ("UniformSplines require uniformly spaced knots"), in
  ///          every build type, if a knot is further than sqrt(float epsilon)
  ///          (about 3.5e-4) of a spacing from its expected position
  ///          `x0 + i h`. A `uniformity(...)` tolerance looser than that in
  ///          the dial options lets DialCollection::makeSplineDial() pick
  ///          this class for knots that this check then rejects.

  void buildDial(const std::vector<DialUtils::DialPoint>& splinePointList_);

protected:
  /// Whether evalResponse() extrapolates outside the knot range (true) or
  /// clamps its input to that range (false).
  bool _allowExtrapolation_{false};

  /// A block of data to calculate the spline values.  This must be filled for
  /// the Cache::Manager to work, and provides the input for spline calculation
  /// functions that can be shared between the CPU and the GPU.
  /// Holds `x0`, `h`, then a (value, slope) pair per knot (see the class
  /// description).
  std::vector<double> _splineData_{};

  /// Positions of the first and last knot. evalResponse() clamps its input to
  /// this range when extrapolation is not allowed. NaN until buildDial().
  GenericToolbox::Range _splineBounds_{std::nan("unset"), std::nan("unset")};
};


#endif //GUNDAM_UNIFORMSPLINE_H
