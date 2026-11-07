#ifndef GUNDAM_PARAMETER_GRADIENT_H
#define GUNDAM_PARAMETER_GRADIENT_H

#include <unordered_map>

class Parameter;

/// Derivatives keyed by parameter identity; does not own the parameters.
struct ParameterGradient {
  std::unordered_map<const Parameter*, double> values;

  [[nodiscard]] double getValue(const Parameter* parameter_) const {
    return values.at(parameter_);
  }
  void addValue(const Parameter* parameter_, double value_) {
    values.at(parameter_) += value_;
  }
  void resetValue(const Parameter* parameter_) { values[parameter_] = 0.; }
  void clear() { values.clear(); }
};

#endif // GUNDAM_PARAMETER_GRADIENT_H
