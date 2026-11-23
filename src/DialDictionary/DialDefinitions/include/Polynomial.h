//
// Created by Adrien Blanchet on 10/04/2023.
//

/// @file Polynomial.h
/// @brief Declares Polynomial, a dial whose response is a polynomial in one
///        parameter.
/// @ingroup DialDictionary

#ifndef GUNDAM_POLYNOMIAL_H
#define GUNDAM_POLYNOMIAL_H

#include "DialBase.h"
#include "DialUtils.h"

#include <vector>


/// Dial whose response is a polynomial in its single input parameter.
///
/// A dial is a response function of one or more fit parameters (see
/// DialBase). evalResponse() returns `c[0] + c[1]*x + c[2]*x^2 + ...` with
/// the coefficients given to setCoefficientList(). Unless extrapolation is
/// allowed, x is first clamped to the range given to setSplineBounds().
///
/// Not supported by Cache::Manager (no GPU implementation).
/// @note No code in src/ creates a Polynomial: DialCollection has no dial
///       type that builds one, so the class is currently unused.
/// @ingroup DialDictionary
class Polynomial : public DialBase {

public:
  /// Creates a polynomial with no coefficient (response 0) and unset
  /// bounds.
  Polynomial() = default;

  /// Copies this polynomial dial.
  /// @return A new Polynomial with the same coefficients, bounds and
  ///         extrapolation flag.
  [[nodiscard]] std::unique_ptr<DialBase> clone() const override { return std::make_unique<Polynomial>(*this); }

  /// Gives the type name of this dial.
  /// @return `"Polynomial"`.
  [[nodiscard]] std::string getDialTypeName() const override { return {"Polynomial"}; }

  /// Evaluates the polynomial at the parameter value.
  /// @param input_ Buffer whose first entry is the parameter value x.
  /// @return The sum of `c[n] * x^n` over the coefficients, after clamping x
  ///         to the bounds when extrapolation is not allowed. 0 if no
  ///         coefficient is set.
  [[nodiscard]] double evalResponse(const DialInputBuffer& input_) const override;

  /// Chooses whether x may leave the bounds.
  /// @param allowExtrapolation_ True to evaluate the polynomial at any x;
  ///        false (default) to clamp x to the bounds set by setSplineBounds().
  /// @note getAllowExtrapolation() is not overridden, so it returns false
  ///       even after this is set to true.
  void setAllowExtrapolation(bool allowExtrapolation_) override { _allowExtrapolation_ = allowExtrapolation_; }

  /// Sets the polynomial coefficients.
  /// @param coefficientList_ Coefficients `c[0], c[1], ...`, constant term
  ///        first.
  void setCoefficientList(const std::vector<double> &coefficientList_){ _coefficientList_ = coefficientList_; }

  /// Sets the input range used when extrapolation is not allowed.
  /// @param splineBounds_ Range `[min, max]` to which x is clamped.
  void setSplineBounds(const GenericToolbox::Range& splineBounds_){ _splineBounds_ = splineBounds_; }

private:
  /// Coefficients `c[n]` of `x^n`, constant term first.
  std::vector<double> _coefficientList_{};

  /// Range to which the input is clamped when extrapolation is off. Both
  /// bounds are NaN until setSplineBounds() is called, and comparisons with
  /// NaN are false, so no clamping happens while they are unset.
  GenericToolbox::Range _splineBounds_{std::nan("unset"), std::nan("unset")};

  /// If true, the input is not clamped to `_splineBounds_`.
  bool _allowExtrapolation_{false};

};


#endif //GUNDAM_POLYNOMIAL_H
