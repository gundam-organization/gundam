//
// Created by Adrien Blanchet on 19/01/2023.
//

/// @file RootGraph.h
/// @brief Declares RootGraph, a dial that evaluates a full ROOT TGraph.
/// @ingroup DialDictionary

#ifndef GUNDAM_ROOT_GRAPH_H
#define GUNDAM_ROOT_GRAPH_H

#include "DialBase.h"
#include "DialInputBuffer.h"

#include "TGraph.h"


/// Dial that stores a ROOT TGraph and evaluates it with TGraph::Eval().
///
/// A dial is a response function of one or more fit parameters (see
/// DialBase). Graph does the same job with a compact array and is the class
/// DialCollection uses for graph dials.
///
/// Not supported by Cache::Manager (no GPU implementation).
/// @note No code in src/ creates a RootGraph (DialCollection.cpp only
///       includes the header), so the class is currently unused.
/// @ingroup DialDictionary
class RootGraph : public DialBase {

public:
  /// Creates a dial with an empty graph.
  /// Call setGraph() before evaluating.
  RootGraph() = default;

  // mandatory overrides

  /// Copies this dial.
  /// @return A new RootGraph with a copy of the TGraph and of the
  ///         extrapolation flag.
  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<RootGraph>(*this); }

  /// Gives the type name of this dial.
  /// @return `"RootGraph"`.
  [[nodiscard]] std::string getDialTypeName() const override { return {"RootGraph"}; }

  /// Evaluates the graph at the parameter value.
  /// @param input_ Buffer whose first entry is the parameter value x.
  /// @return Without extrapolation: the y of the first stored point if x is
  ///         at or below its x, the y of the last stored point if x is at or
  ///         above its x, and otherwise TGraph::Eval(x) (linear
  ///         interpolation). With extrapolation allowed, TGraph::Eval(x)
  ///         everywhere.
  /// @pre The graph has at least one point, sorted by increasing x: the range
  ///      check uses the first and last stored points, and setGraph() does
  ///      not sort.
  /// @note Builds without NDEBUG throw if x is not finite.
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;
  /// Gives the `x` of every point of the stored TGraph. Used by the Python
  /// interface through DialInterface::getSourceKnotPositions().
  /// @return One `x` per point, in the order of the TGraph.
  [[nodiscard]] std::vector<double> getSourceKnotPositions() const override;

  // other overrides

  /// Chooses what happens outside the range of the graph points.
  /// @param allowExtrapolation True to let TGraph::Eval() handle any x; false
  ///        (default) to return the end-point value.
  void setAllowExtrapolation(bool allowExtrapolation) override { _allowExtrapolation_ = allowExtrapolation; }

  /// Tells whether the dial evaluates the graph outside its point range.
  /// @return The flag set by setAllowExtrapolation().
  [[nodiscard]] bool getAllowExtrapolation() const override { return _allowExtrapolation_; }

  /// Sets the graph to evaluate.
  /// @param graph Source graph; it is copied as is (not sorted).
  void setGraph(const TGraph &graph){ _graph_ = graph; }

private:
  /// If true, evalResponse() calls TGraph::Eval() outside the point range
  /// instead of returning the end-point value.
  bool _allowExtrapolation_{false};

  /// The graph evaluated by evalResponse().
  /// Set by setGraph().
  TGraph _graph_{};
};

#endif //GUNDAM_ROOT_GRAPH_H
