//
// Created by Adrien Blanchet on 30/11/2022.
//

/// @file RootSpline.h
/// @brief Declares RootSpline, a spline dial evaluated with a ROOT TSpline3.
/// @ingroup DialDictionary

#ifndef GUNDAM_ROOT_SPLINE_H
#define GUNDAM_ROOT_SPLINE_H

#include "DialBase.h"
#include "DialInputBuffer.h"

#include "DialUtils.h"

#include "TSpline.h"

/// Evaluates a one-parameter response with a ROOT TSpline3 cubic spline.
///
/// A dial is the response function of one or more fit parameters: it turns
/// the current parameter value into a weight factor for an event or a bin.
/// DialCollection::makeSplineDial() creates a RootSpline only when the dial
/// options contain `ROOT`; the comment there notes that it is slow compared
/// with the other spline dials. The dial is then stored in its
/// DialCollection.
///
/// @par Interpolation
/// buildDial() builds the TSpline3 from the knot positions and values only
/// (DialUtils::buildTSpline3()), with ROOT's default end conditions. The
/// `slope` of the input DialUtils::DialPoint list is ignored, and so is the
/// `monotonic` option. Any knot spacing is accepted.
///
/// @par Outside the knot range
/// When extrapolation is not allowed (the class default), the response is
/// the spline value at the first or last knot. When it is allowed,
/// TSpline3::Eval() is called with the input unchanged, so the extrapolation
/// is whatever ROOT does outside the knot range.
/// Dials made by DialCollection::makeDial() extrapolate unless the dial
/// set sets `allowDialExtrapolation` to false.
///
/// @warning Not supported by Cache::Manager: the class does not override
///          getDialData(), and Cache::Manager::Build() reports it as an
///          unsupported dial type and throws.
/// @note It does not override getAllowExtrapolation() either, so that
///       function returns false (the DialBase default) even after
///       setAllowExtrapolation(true).
///
/// @see CompactSpline, UniformSpline, GeneralSpline
/// @ingroup DialDictionary
class RootSpline : public DialBase {

public:
  /// Creates an empty spline. Call buildDial() before evaluating it.
  RootSpline() = default;

  // mandatory overrides

  /// Copies this dial, including its TSpline3 and extrapolation flag.
  /// Used by the polymorphic dial container (DialBaseObject) when dials are
  /// copied.
  /// @return A new RootSpline equal to this one, owned by the caller.
  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<RootSpline>(*this); }

  /// Gives the type name of this dial, used in printouts.
  /// @return `"RootSpline"`.
  [[nodiscard]] std::string getDialTypeName() const override { return {"RootSpline"}; }

  /// Evaluates the TSpline3 at the current parameter value.
  /// @param input_ Input buffer of the dial. Only element 0, the parameter
  ///        value after any mirroring, is read.
  /// @return The spline value. It is not clamped here; the response limits
  ///         of the dial set are applied by DialResponseSupervisor.
  /// @pre buildDial() or setSpline() has been called.
  /// @note In debug builds (NDEBUG not defined) a non-finite input throws.
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;
  /// Gives the `x` of every knot of the stored TSpline3. Used by the Python
  /// interface through DialInterface::getSourceKnotPositions().
  /// @return One `x` per knot.
  [[nodiscard]] std::vector<double> getSourceKnotPositions() const override;

  // other overrides

  /// Sets whether evalResponse() may extrapolate beyond the first and last
  /// knot. DialCollection::makeDial() calls it with the dial set's
  /// `allowDialExtrapolation` option, which is true when not configured.
  /// @param allowExtrapolation If false, inputs outside the knot range get
  ///        the value at the nearest end knot. If true, TSpline3::Eval() is
  ///        used as is.
  void setAllowExtrapolation(bool allowExtrapolation) override{ _allowExtrapolation_ = allowExtrapolation; }

  // setters

  /// Replaces the spline with a copy of the given one.
  /// @param spline_ Spline to copy.
  /// @note Not called anywhere in the code base; dials are built with
  ///       buildDial().
  void setSpline(const TSpline3 &spline_){ _spline_ = spline_; }

  // getters

  /// Gives read access to the underlying spline.
  /// @return A reference to the stored TSpline3.
  /// @note Not called anywhere in the code base.
  [[nodiscard]] const TSpline3 &getSpline() const { return _spline_; }

  // methods

  /// Builds the TSpline3 from the knots with DialUtils::buildTSpline3().
  /// Called by DialCollection::makeSplineDial() after sorting the points
  /// (DialCollection::checkDialPointList()).
  /// @param splinePointList_ Knots sorted by increasing `x`. Only `x` and `y`
  ///        are used.
  /// @post Any previous spline is replaced (no "already set" check, unlike
  ///       the other spline dials).
  void buildDial(const std::vector<DialUtils::DialPoint>& splinePointList_);

protected:
  /// Whether evalResponse() extrapolates outside the knot range (true) or
  /// returns the end-knot value there (false).
  bool _allowExtrapolation_{false};

  /// The ROOT spline that evalResponse() evaluates. Empty until buildDial()
  /// or setSpline().
  TSpline3 _spline_{};

};

#endif // GUNDAM_ROOT_SPLINE_H
