//
// Created by Adrien Blanchet on 29/11/2022.
//

/// @file Shift.h
/// @brief Declares Shift, a dial with a constant response.
/// @ingroup DialDictionary

#ifndef GUNDAM_SHIFT_H
#define GUNDAM_SHIFT_H

#include "DialBase.h"

/// Implement a dial to apply a scale factor (not a variable shift).  This can
/// be used to replace a Spline or a Graph when it corresponds to a constant
/// value.
///
/// A dial is a response function of one or more fit parameters (see
/// DialBase); this one ignores the parameters. DialCollection::makeDialShortCircuit()
/// creates a Shift when the knots of a graph or spline read from the input
/// are flat (DialUtils::isFlat(), or a single knot) and their common value is
/// not 1. When the value is 1, no dial is created at all.
///
/// On the GPU, Cache::Manager::Update() does not evaluate Shift dials: it
/// multiplies the event's initial weight by the value once.
/// @ingroup DialDictionary
class Shift : public DialBase {

public:
  /// Creates a dial with response 1 until setShiftValue() is called.
  Shift() = default;

  // mandatory overrides

  /// Copies this constant dial.
  /// @return A new Shift with the same value.
  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<Shift>(*this); }

  /// Gives the type name of this dial.
  /// @return `"Shift"`. Cache::Manager::Build() matches this name to count
  ///         constant dials.
  [[nodiscard]] std::string getDialTypeName() const override { return {"Shift"}; }

  /// Returns the stored constant, whatever the parameter values.
  /// @param input_ Not used. Cache::Manager::Update() even passes an empty
  ///        buffer.
  /// @return The value set by setShiftValue() (1 by default).
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override { return _shiftValue_; }

  /// Gives the constant response. DialCollection uses it to recognise
  /// identical Shift dials, so that they can share one object.
  /// @return `_shiftValue_`.
  [[nodiscard]] double getShiftValue() const { return _shiftValue_; }

  // setters

  /// Sets the constant response.
  /// @param shift_ Weight factor returned by evalResponse(); DialCollection
  ///        passes the value of the first knot of the flat graph or spline.
  void setShiftValue(double shift_){ _shiftValue_ = shift_; }

private:
  /// Constant weight factor returned by evalResponse(); 1 (no effect) by
  /// default.
  double _shiftValue_{1};

};
#endif //GUNDAM_SHIFT_H
