//
// Created by Adrien Blanchet on 19/01/2023.
//

/// @file Graph.h
/// @brief Declares Graph, the piecewise-linear dial.
/// @ingroup DialDictionary

#ifndef GUNDAM_GRAPH_H
#define GUNDAM_GRAPH_H

#include "DialBase.h"
#include "DialUtils.h"

#include "TGraph.h"

#include <vector>


/// A DialBase class to do piecewise linear interpolation.  This is
/// initialized using a ROOT TGraph, and will return the same values, but uses
/// much less space and is generally faster.
///
/// A dial is a response function of one or more fit parameters (see
/// DialBase). DialCollection::makeGraphDial() creates a Graph for
/// `dialType: Graph` collections. DialCollection::makeDialShortCircuit() also
/// uses one for any graph or spline input with exactly two knots, whatever
/// the dial type (two knots with the same value give a Shift instead), and
/// for graph or spline inputs whose points all lie on one straight line,
/// which are reduced to their first and last points. The knots are stored
/// packed in `_data_`.
///
/// Between knots the response is the straight line through the two
/// surrounding knots. Outside the knot range it stays at the end-knot value,
/// unless extrapolation is allowed (setAllowExtrapolation()), in which case
/// the first or last segment is extended.
///
/// The GPU counterpart is Cache::Weight::Graph, which runs the same
/// CalculateGraph() kernel.
/// @note In the current code the dial is built from a DialPoint list
///       (buildDial(const std::vector<DialUtils::DialPoint>&)); the TGraph
///       overload is not called anywhere in src/.
/// @note On the GPU the extrapolation flag is not used: the kernel always
///       extends the end segments. Cache::Weight::Graph also rejects graphs
///       with more than 15 knots.
/// @ingroup DialDictionary
class Graph : public DialBase {

public:
  /// Creates an empty graph.
  /// Call buildDial() before evaluating.
  Graph() = default;

  /// Copies this graph dial.
  /// @return A new Graph with a copy of the knot data and of the
  ///         extrapolation flag.
  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<Graph>(*this); }

  /// Gives the type name of this dial.
  /// @return `"Graph"`. Cache::Manager::Build() matches this name to count
  ///         graph dials.
  [[nodiscard]] std::string getDialTypeName() const override { return {"Graph"}; }

  /// Interpolates linearly between the knots around the parameter value.
  /// @param input_ Buffer whose first entry is the parameter value x.
  /// @return Without extrapolation: the first knot's value if x is at or below
  ///         the first knot, the last knot's value if x is at or above the last
  ///         knot, and otherwise the linear interpolation between the two
  ///         surrounding knots. With extrapolation allowed, the first or last
  ///         segment is extended linearly outside the knot range.
  /// @pre buildDial() was called with at least one knot.
  /// @note Builds without NDEBUG throw if x is not finite.
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;
  /// Gives the `x` of every point stored in `_data_`. Used by the Python
  /// interface through DialInterface::getSourceKnotPositions().
  /// @return One `x` per point, in increasing order; empty before buildDial().
  [[nodiscard]] std::vector<double> getSourceKnotPositions() const override;

  /// Chooses what happens outside the knot range.
  /// @param allowExtrapolation True to extend the first or last segment
  ///        linearly; false (default) to return the end-knot value.
  void setAllowExtrapolation(bool allowExtrapolation) override;

  /// Tells whether the dial extends its end segments.
  /// @return The flag set by setAllowExtrapolation().
  [[nodiscard]] bool getAllowExtrapolation() const override;

  /// Builds the dial from the points of a ROOT TGraph.
  /// @param grf Source graph. A sorted copy is used; the graph itself is not
  ///        modified.
  /// @param option_ Not used.
  /// @post The previous knots are replaced. Throws (LogThrowIf()) if the graph
  ///       has no point.
  /// @note Not called anywhere in src/.
  void buildDial(const TGraph& grf, const std::string& option_="");

  /// Builds the dial from a list of knots.
  /// @param pointList_ Knots in increasing x order (DialCollection sorts them
  ///        in DialCollection::checkDialPointList()); the slopes are ignored.
  /// @post The previous knots are replaced. The list is neither sorted nor
  ///       checked, so an empty list gives a dial that must not be evaluated.
  void buildDial(const std::vector<DialUtils::DialPoint>& pointList_);

  /// Gives the packed knot data, for Cache::Manager.
  /// @return `_data_`, laid out as `{y0, x0, y1, x1, ...}`, the layout that
  ///         CalculateGraph() expects.
  [[nodiscard]] const std::vector<double>& getDialData() const override {return _data_;}

protected:
  /// If true, evalResponse() extends the end segments instead of returning
  /// the end-knot values.
  bool _allowExtrapolation_{false};

  /// The data for the graph packed as {y0,x0,y1,x1,y2,x2,...}
  std::vector<Double_t> _data_;
};

#endif //GUNDAM_GRAPH_H
