#ifndef GUNDAM_GAUSSIAN_CONSTRAINT_H
#define GUNDAM_GAUSSIAN_CONSTRAINT_H

#include "ConfigUtils.h"

#include <memory>
#include <string>
#include <vector>

class TFormula;
class Parameter;
class ParametersManager;

/// A Gaussian measurement of a formula evaluated on physical parameter values.
/// Parameter flags and priors do not affect this independent constraint.
class GaussianConstraint : public JsonBaseClass {
public:
  [[nodiscard]] auto& getName() const{ return _name_; }
  [[nodiscard]] auto isEnabled() const{ return _isEnabled_; }
  [[nodiscard]] auto& getFormula() const{ return _formulaString_; }
  [[nodiscard]] auto getMean() const{ return _mean_; }
  [[nodiscard]] auto getStdDev() const{ return _stdDev_; }

  void validateParameters(const ParametersManager& parametersManager_) const;
  [[nodiscard]] double evalFormula(const ParametersManager& parametersManager_) const;
  [[nodiscard]] double evalResidual(const ParametersManager& parametersManager_) const;
  [[nodiscard]] double eval(const ParametersManager& parametersManager_) const;

protected:
  void configureImpl() override;
  void initializeImpl() override;

private:
  struct ParameterReference {
    std::string name;
    std::string parameterSetName;
    std::string parameterName;
    int parameterIndex{-1};
    int formulaIndex{-1};
  };

  const Parameter& resolveParameter(const ParametersManager& parametersManager_, const ParameterReference& reference_) const;

  std::string _name_;
  bool _isEnabled_{true};
  std::string _formulaString_;
  double _mean_{0};
  double _stdDev_{1};
  std::vector<ParameterReference> _parameterReferenceList_;
  std::shared_ptr<const TFormula> _formula_{};
};

#endif // GUNDAM_GAUSSIAN_CONSTRAINT_H
