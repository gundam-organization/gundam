//
// Created by Nadrino on 25/09/2024.
//

/// @file DialUtils.h
/// @brief Declares DialUtils: helpers that read knots from ROOT graphs and
///        splines and prepare them for the graph and spline dials.
/// @ingroup DialDictionary

#ifndef GUNDAM_DIALUTILS_H
#define GUNDAM_DIALUTILS_H

#include "TSpline.h"
#include "TGraph.h"
#include "TObject.h"

#include <vector>
#include <string>
#include <limits>
#include <cmath>


/// Helpers that turn the ROOT object of a graph or spline dial into a list of
/// knots, and prepare that list before a dial is built from it.
///
/// A dial is a response function of one or more fit parameters (see
/// DialBase). DialCollection::makeGraphDial() and
/// DialCollection::makeSplineDial() use these functions to:
/// - read the knots of a TGraph or TSpline3 as DialPoint objects
///   (getPointList(), getPointListNoSlope());
/// - pick a cheaper dial when possible (isFlat() leads to a Shift or no dial,
///   isUniform() to a UniformSpline);
/// - fill or condition the knot slopes (fillAkimaSlopes(),
///   fillCatmullRomSlopes(), applyMonotonicCondition()).
///
/// The dial classes then take the point list in their `buildDial()` methods.
/// RootSpline::buildDial() uses buildTSpline3().
///
/// @ingroup DialDictionary
namespace DialUtils{

  /// One knot of a graph or spline dial: position, value and slope.
  ///
  /// All three fields start as NaN ("unset").
  /// @ingroup DialDictionary
  struct DialPoint{
    double x{std::nan("unset")};     ///< Knot position: the dial input (parameter value) at this knot.
    double y{std::nan("unset")};     ///< Dial response at the knot.
    double slope{std::nan("unset")}; ///< Derivative dy/dx at the knot; 0 when slopes are not computed.
  };

  /// Reads the knots of a ROOT TGraph or TSpline3 passed as a generic
  /// TObject.
  ///
  /// Dispatches on the object's class (TObject::InheritsFrom()) to
  /// getPointList(const TGraph*) or getPointList(const TSpline3*). Used by
  /// DialCollection::makeSplineDial() and, for objects that are not a TGraph,
  /// by DialCollection::makeGraphDial().
  /// @param src_ The dial object read from a file or a tree; may be null.
  /// @return The knots with their slopes. Empty if `src_` is null, is neither
  ///         a TGraph nor a TSpline3, or contains a non-finite value.
  std::vector<DialPoint> getPointList(const TObject* src_);

  /// Reads the points of a TGraph and computes a slope at each one.
  ///
  /// The slopes are the derivatives, at each point, of a ROOT TSpline3 built
  /// through the graph with ROOT's default end conditions.
  /// @param src_ The graph; may be null.
  /// @return One knot per graph point. A one-point graph gives one knot with
  ///         slope 0 (its values are not checked). Empty if `src_` is null or
  ///         has no point, or if a position, value or slope of a multi-point
  ///         graph is not finite.
  std::vector<DialPoint> getPointList(const TGraph* src_);

  /// Reads the knots of a TSpline3 and the spline's derivative at each knot.
  /// @param src_ The spline; may be null.
  /// @return One knot per spline knot, in knot order. A one-knot spline gives
  ///         slope 0. Empty if `src_` is null, has no knot, or if any knot
  ///         position, value or derivative is not finite.
  std::vector<DialPoint> getPointList(const TSpline3* src_);

  /// Copies the points of a TGraph, with every slope set to 0.
  ///
  /// Used by DialCollection::makeGraphDial() for TGraph inputs: a Graph dial
  /// interpolates linearly and needs no slopes, so no spline is built.
  /// @param src_ The graph; may be null.
  /// @return One knot per graph point, in graph order (not sorted, not checked
  ///         for finite values). Empty if `src_` is null or has no point.
  std::vector<DialPoint> getPointListNoSlope(const TGraph* src_);

  /// Overwrites the slopes with those of a Catmull-Rom spline.
  ///
  /// Each interior knot gets the slope of the line through its two
  /// neighbours, `(y[i+1]-y[i-1])/(x[i+1]-x[i-1])`. The first and last knots
  /// get the slope of their only segment. A single knot gets slope 0, and an
  /// empty list is left unchanged.
  /// @param splinePointList_ Knots in increasing x order; their slopes are
  ///        overwritten.
  /// @note DialCollection::makeSplineDial() calls it for the `catmull-rom`
  ///       spline type, but the slopes are not used afterwards: such splines
  ///       become CompactSpline or MonotonicSpline dials, whose `buildDial()`
  ///       keeps only the knot values.
  void fillCatmullRomSlopes(std::vector<DialPoint>& splinePointList_);

  /// Overwrites the slopes with those of an Akima spline.
  ///
  /// With `m[k]` the slope of the segment from knot k to knot k+1, each knot
  /// i with 2 <= i <= n-3 gets
  /// `(w1*m[i-1] + w0*m[i]) / (w1 + w0)`, where `w1 = abs(m[i+1] - m[i])` and
  /// `w0 = abs(m[i-1] - m[i-2])`. Knots 1 and n-2 get the Catmull-Rom slope
  /// (line through their two neighbours). The first and last knots get the
  /// slope of their only segment. A single knot gets slope 0, and an empty
  /// list is left unchanged. Called by DialCollection::makeSplineDial() for
  /// the `akima` spline type.
  /// @param splinePointList_ Knots in increasing x order; their slopes are
  ///        overwritten.
  /// @warning When both weights are 0 (four consecutive segments with the
  ///          same slope, e.g. collinear knots) the weighted formula is still
  ///          used, and the slope becomes NaN (0/0). The fallback
  ///          `(m[i-1] + m[i]) / 2` is only taken when `w0 <= 1e-6` and
  ///          `w1 >= 1e-6`.
  void fillAkimaSlopes(std::vector<DialPoint>& splinePointList_);

  /// Limits the slopes so that the cubic spline through the knots does not
  /// overshoot between them (Fritsch-Carlson monotonic condition).
  ///
  /// Computes Fritsch-Carlson tangents from the segment slopes (zero at local
  /// extrema and next to flat segments, at most 3 times the segment slope),
  /// then compares each existing slope with its tangent: a positive slope
  /// above the tangent, or a negative slope below it, is replaced by the
  /// tangent. Other slopes are kept. Called by
  /// DialCollection::makeSplineDial() for `monotonic` splines that are not
  /// Catmull-Rom (MonotonicSpline handles that case itself while evaluating).
  /// @param splinePointList_ Knots in increasing x order, with their slopes
  ///        already filled; the slopes are modified in place.
  /// @pre The list has at least 2 knots: with fewer, the function reads
  ///      outside its internal arrays. DialCollection only calls it on lists
  ///      of 3 or more knots.
  void applyMonotonicCondition(std::vector<DialPoint>& splinePointList_);

  /// Tells whether all knots have the same value, within an absolute
  /// tolerance.
  ///
  /// DialCollection::makeDialShortCircuit() uses it to replace a flat response
  /// by a Shift dial, or by no dial when the value is 1.
  /// @param splinePointList_ The knots.
  /// @param tolerance_ Largest allowed `abs(y - y[0])`. The default (2 float
  ///        epsilons, about 2.4e-7) allows for values that went through a
  ///        float.
  /// @return True if every value is within `tolerance_` of the first one, or
  ///         if the list has fewer than 2 knots.
  bool isFlat(const std::vector<DialPoint>& splinePointList_, double tolerance_ = 2*std::numeric_limits<float>::epsilon());

  /// Tells whether the knots are evenly spaced in x.
  ///
  /// DialCollection::makeSplineDial() uses it to choose UniformSpline rather
  /// than GeneralSpline, and to check the knots of Catmull-Rom splines.
  /// @param splinePointList_ Knots in increasing x order.
  /// @param tolerance_ Largest allowed distance between a knot and its
  ///        evenly spaced position, as a fraction of the average spacing. The
  ///        default is 16 float epsilons (about 1.9e-6).
  /// @return True if every knot is within `tolerance_` of its evenly spaced
  ///         position, or if the list has fewer than 2 knots.
  bool isUniform(const std::vector<DialPoint>& splinePointList_, double tolerance_ = 16*std::numeric_limits<float>::epsilon());

  /// Computes the slope of the straight line through two knots.
  /// @param start_ First knot.
  /// @param end_ Second knot.
  /// @return `(end_.y - start_.y) / (end_.x - start_.x)`. Not finite if the
  ///         two knots have the same x.
  double getSlope(const DialPoint& start_, const DialPoint& end_);

  /// Builds a ROOT TSpline3 through the knot positions and values.
  ///
  /// Used by RootSpline::buildDial(). The slopes in the list are ignored:
  /// ROOT computes its own spline coefficients with its default end
  /// conditions.
  /// @param splinePointList_ Knots in increasing x order.
  /// @return The spline, with an empty title.
  TSpline3 buildTSpline3(const std::vector<DialPoint>& splinePointList_);

}

#endif //GUNDAM_DIALUTILS_H
