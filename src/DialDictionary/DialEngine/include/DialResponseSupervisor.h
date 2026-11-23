//
// Created by Adrien Blanchet on 29/11/2022.
//

/// @file DialResponseSupervisor.h
/// @brief Declares DialResponseSupervisor, the clamp applied to dial
///        responses.
/// @ingroup DialDictionary

#ifndef GUNDAM_DIALRESPONSESUPERVISOR_H
#define GUNDAM_DIALRESPONSESUPERVISOR_H

#include <cmath>
#include <string>
#include <vector>
#include <functional>


/// Clamps a dial response to an optional [min, max] range.
///
/// A dial is a response function of one or more fit parameters. Every
/// DialInterface passes its dial's raw response through process() (see
/// DialInterface::evalResponse()). A bound set to NaN is not applied, so a
/// default-constructed supervisor returns every response unchanged.
///
/// @par Ownership
/// DialCollection owns its supervisors by value and gives each
/// DialInterface a non-owning pointer. In the current code each collection
/// creates exactly one supervisor, shared by all its dials, with the minimum
/// taken from the `minDialResponse` option (if absent: 0, or no minimum for
/// composite nodes and dials that change event variables) and the maximum
/// from `maxDialResponse` (no maximum if absent).
///
/// The GPU Cache::Manager reads both bounds (getMinResponse(),
/// getMaxResponse()) to set up its own clamps.
/// @ingroup DialDictionary
class DialResponseSupervisor {

public:
  /// Creates a supervisor with no bounds; process() then changes nothing.
  DialResponseSupervisor() = default;

  // setters

  /// Sets the lower bound.
  /// @param minResponse Smallest response process() returns; NaN for no
  ///        lower bound.
  void setMinResponse(double minResponse){ _minResponse_ = minResponse; }
  /// Sets the upper bound.
  /// @param maxResponse Largest response process() returns; NaN for no upper
  ///        bound.
  void setMaxResponse(double maxResponse){ _maxResponse_ = maxResponse; }

  // const getters

  /// Gives the lower bound.
  /// @return The lower bound; NaN if none.
  /// @note Cache::Manager uses 0 on the GPU when this value is not finite.
  [[nodiscard]] double getMinResponse() const{ return _minResponse_; }
  /// Gives the upper bound.
  /// @return The upper bound; NaN if none.
  [[nodiscard]] double getMaxResponse() const{ return _maxResponse_; }

  /// Clamps one dial response.
  /// @param reponse_ Raw response of a dial.
  /// @return The lower bound if the response is below it, else the upper
  ///         bound if the response is above it, else the response unchanged.
  ///         A NaN response is returned unchanged.
  [[nodiscard]] double process(double reponse_) const;
  /// Describes the bounds for debug printouts.
  /// @param shallow_ Unused.
  /// @return `minResponse=MIN, maxResponse=MAX`, leaving out unset bounds;
  ///         empty if neither is set.
  [[nodiscard]] std::string getSummary(bool shallow_ = true) const;


private:
  double _minResponse_{std::nan("unset")}; ///< Lower bound applied by process(); NaN means none.
  double _maxResponse_{std::nan("unset")}; ///< Upper bound applied by process(); NaN means none.


};


#endif //GUNDAM_DIALRESPONSESUPERVISOR_H
