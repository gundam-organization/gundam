#ifndef GUNDAM_EXTERNAL_CONSTRAINTS_H
#define GUNDAM_EXTERNAL_CONSTRAINTS_H

#include "GaussianConstraint.h"
#include "ConfigUtils.h"

#include <vector>

class ParametersManager;

/// Independent external measurements contributing to the penalty likelihood.
class ExternalConstraints : public JsonBaseClass {
public:
  [[nodiscard]] auto isEnabled() const{ return _isEnabled_; }
  [[nodiscard]] auto& getGaussianConstraintList() const{ return _gaussianConstraintList_; }
  void validateParameters(const ParametersManager& parametersManager_) const;
  [[nodiscard]] double eval(const ParametersManager& parametersManager_) const;

protected:
  void configureImpl() override;
  void initializeImpl() override;

private:
  bool _isEnabled_{false};
  std::vector<GaussianConstraint> _gaussianConstraintList_;
};

#endif // GUNDAM_EXTERNAL_CONSTRAINTS_H
