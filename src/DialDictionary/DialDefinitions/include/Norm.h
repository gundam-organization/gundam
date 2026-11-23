//
// Created by Adrien Blanchet on 29/11/2022.
//

/// @file Norm.h
/// @brief Declares Norm, the normalisation dial whose response is the value
///        of its parameter.
/// @ingroup DialDictionary

#ifndef GUNDAM_NORM_H
#define GUNDAM_NORM_H

#include "DialBase.h"

// for norm dial, no cache is needed


/// Normalisation dial: its response is the value of its single input
/// parameter.
///
/// A dial is a response function of one or more fit parameters (see
/// DialBase). A Norm dial scales the weight of every event it applies to by
/// the parameter value: a value of 1.2 gives events 20% more weight.
/// DialCollection::makeDial() (the overload with no argument) creates it for
/// `dialType: Norm` (or `Normalization`) collections, once for the whole
/// collection, and for collections with a `parametersBinningPath`, once per
/// bin. It also replaces a graph or spline whose points lie on the line
/// y = x when extrapolation is allowed. One Norm is shared by all the events
/// it applies to, and it has no data member.
///
/// The GPU counterpart is Cache::Weight::Normalization, which multiplies the
/// event weight by the parameter value.
/// @ingroup DialDictionary
class Norm : public DialBase {

public:
  /// Creates the dial.
  /// It has nothing to configure.
  Norm() = default;

  // mandatory overrides

  /// Copies this normalisation dial.
  /// @return A new Norm.
  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<Norm>(*this); }

  /// Gives the type name of this dial.
  /// @return `"Norm"`. Cache::Manager::Build() matches this name to count
  ///         normalisation dials.
  [[nodiscard]] std::string getDialTypeName() const override { return {"Norm"}; }

  /// Returns the parameter value as the weight factor.
  /// @param input_ Buffer whose first entry is the parameter value (after
  ///        mirroring, if enabled); other entries are ignored.
  /// @return `input_.getInputBuffer()[0]`, unchanged. Negative values are
  ///         left to the DialResponseSupervisor, which by default clamps the
  ///         response at 0.
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override { return input_.getInputBuffer()[0]; }

};


#endif //GUNDAM_NORM_H
