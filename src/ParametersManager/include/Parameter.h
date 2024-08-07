//
// Created by Nadrino on 21/05/2021.
//

#ifndef GUNDAM_PARAMETER_H
#define GUNDAM_PARAMETER_H

#include "JsonBaseClass.h"

#include "nlohmann/json.hpp"

#include <vector>
#include <string>


class ParameterSet;

/// Hold a Parameter that is a member of a ParameterSet to be used in the fit.
/// Parameters are always owned by a ParameterSet, and the instantiation
/// resides in the ParameterSet.
class Parameter : public JsonBaseClass {

public:
#define ENUM_NAME PriorType
#define ENUM_FIELDS \
  ENUM_FIELD(Unset, -1) \
  ENUM_FIELD(Gaussian) \
  ENUM_FIELD(Flat)
#include "GenericToolbox.MakeEnum.h"

protected:
  // called through public JsonBaseClass::readConfig() and JsonBaseClass::initialize()
  void readConfigImpl() override;
  void initializeImpl() override;

public:
  Parameter() = delete; // should always provide the owner
  explicit Parameter(const ParameterSet* owner_): _owner_(owner_) {}

  void setIsEnabled(bool isEnabled){ _isEnabled_ = isEnabled; }
  void setIsFixed(bool isFixed){ _isFixed_ = isFixed; }
  void setIsEigen(bool isEigen){ _isEigen_ = isEigen; }
  void setIsFree(bool isFree){ _isFree_ = isFree; }
  void setParameterIndex(int parameterIndex){ _parameterIndex_ = parameterIndex; }
  void setStepSize(double stepSize){ _stepSize_ = stepSize; }

  /// Set the minimum value for this parameter.  Parameter values less than
  /// this value are illegal, and the likelihood is undefined.  The job will
  /// terminate when it encounters an illegal parameter value.  Note: Using a
  /// minimum value when also using eigenvalue decomposition, or PCA results
  /// in undefined behavior because the decomposition will not honor the
  /// boundaries and may set take values that are out of bounds.  In this
  /// case, the job will stop.  Note: If the minimum value is not set, then
  /// the bound is a negative infinity.
  void setMinValue(double minValue){ _minValue_ = minValue; }

  /// Set the maximum value for this parameter.  Parameter values more than
  /// this value are illegal, and the likelihood is undefined. The job will
  /// terminate when it encounters an illegal parameter value.  Note: Using a
  /// minimum value when also using eigenvalue decomposition, or PCA results
  /// in undefined behavior because the decomposition will not honor the
  /// boundaries and may set take values that are out of bounds.  In this
  /// case, the job will stop.  Note: If the maximum value is not set, then
  /// the bound is at positive infinity.
  void setMaxValue(double maxValue){ _maxValue_ = maxValue; }

  /// Record the minimum mirroring boundary being used by any dials for this
  /// parameter.  If this is set, then GUNDAM will constrain the parameter
  /// value passed to the likelihood to be greater than the mirror boundary,
  /// while the input parameter value can continue outside of the bounds.
  void setMinMirror(double minMirror);

  /// Record the maximum mirroring boundary being used by any dials for this
  /// parameter.  If this is set, then GUNDAM will constrain the parameter
  /// value passed to the likelihood to be less than the mirror boundary,
  /// while the input parameter value can continue outside of the bounds.
  void setMaxMirror(double maxMirror);

  /// Record the physical minimum bound for the parameter.  This is the range
  /// where the parameter has a physically meaningful value.  Because of
  /// numeric continuation, the likelihood may have a finite value outside of
  /// the physical range.  From a mathmatical perspective, the value of the
  /// LLH is infinite below the physical minimum.  This can be enforced
  /// using the Likelihood::SetParameterValidity() method.
  void setMinPhysical(double minPhysical){ _minPhysical_ = minPhysical; }

  /// Record the physical maximum bound for the parameter.  This is the range
  /// where the parameter has a physically meaningful value.  Because of
  /// numeric continuation, the likelihood may have a finite value outside of
  /// the physical range.  From a mathmatical perspective, the value of the
  /// LLH is infinite below the physical minimum.  This can be enforced
  /// using the Likelihood::SetParameterValidity() method.
  void setMaxPhysical(double maxPhysical){ _maxPhysical_ = maxPhysical; }
  void setPriorValue(double priorValue){ _priorValue_ = priorValue; }
  void setThrowValue(double throwValue){ _throwValue_ = throwValue; }
  void setStdDevValue(double stdDevValue){ _stdDevValue_ = stdDevValue; }
  void setParameterValue(double parameterValue);
  void setName(const std::string &name){ _name_ = name; }
  void setDialSetConfig(const JsonType &jsonConfig_);
  void setParameterDefinitionConfig(const JsonType &config_);
  void setOwner(const ParameterSet *owner_){ _owner_ = owner_; }
  void setPriorType(PriorType priorType){ _priorType_ = priorType; }

  // Getters
  [[nodiscard]] bool isFree() const{ return _isFree_; }
  [[nodiscard]] bool isFixed() const{ return _isFixed_; }
  [[nodiscard]] bool isEigen() const{ return _isEigen_; }
  [[nodiscard]] bool isEnabled() const{ return _isEnabled_; }
  [[nodiscard]] bool isMaskedForPropagation() const;
  [[nodiscard]] bool gotUpdated() const { return _gotUpdated_; }
  [[nodiscard]] int getParameterIndex() const{ return _parameterIndex_; }
  [[nodiscard]] double getStepSize() const{ return _stepSize_; }
  /// See setMinValue() for documentation.
  [[nodiscard]] double getMinValue() const{ return _minValue_; }
  /// See setMaxValue() for documentation.
  [[nodiscard]] double getMaxValue() const{ return _maxValue_; }
  /// See setMinMirror for documentation.
  [[nodiscard]] double getMinMirror() const{ return _minMirror_; }
  /// See setMaxMirror for documentation.
  [[nodiscard]] double getMaxMirror() const{ return _maxMirror_; }
  [[nodiscard]] double getPriorValue() const{ return _priorValue_; }
  [[nodiscard]] double getThrowValue() const{ return _throwValue_; }
  /// See setMinPhysical for documentation.
  [[nodiscard]] double getMinPhysical() const{ return _minPhysical_; }
  /// See setMinPhysical for documentation.
  [[nodiscard]] double getMaxPhysical() const{ return _maxPhysical_; }
  [[nodiscard]] double getStdDevValue() const{ return _stdDevValue_; }
  [[nodiscard]] double getParameterValue() const;
  [[nodiscard]] const std::string &getName() const{ return _name_; }
  [[nodiscard]] const JsonType &getDialDefinitionsList() const{ return _dialDefinitionsList_; }
  [[nodiscard]] const ParameterSet *getOwner() const{ return _owner_; }
  [[nodiscard]] PriorType getPriorType() const{ return _priorType_; }

  /// Copy the prior value of the parameter into the current value.  This will
  /// fail if the prior value has not been set.
  void setValueAtPrior();

  /// Copy the current value of the parameter into the prior value.
  void setCurrentValueAsPrior();

  /// Check that the parameter value is between the minimum and maximum bound
  /// for the parameter (could be +/- infinity).  Note: Since a NaN is not a
  /// number, it is not within the bounds.
  [[nodiscard]] bool isValueWithinBounds() const;

  /// Return the difference between the current parameter value and the prior
  /// parameter value in units of standard deviations (defined by
  /// setStdDevvalue()).  This is a signed difference.
  [[nodiscard]] double getDistanceFromNominal() const;

  [[nodiscard]] std::string getSummary(bool shallow_=false) const;
  [[nodiscard]] std::string getTitle() const;
  [[nodiscard]] std::string getFullTitle() const;

private:
  // Parameters
  bool _isEnabled_{true};
  bool _isFixed_{false};
  bool _isEigen_{false};
  bool _isFree_{false};
  bool _gotUpdated_{false};
  int _parameterIndex_{-1}; // to get the right definition in the json config (in case "name" is not specified)
  double _parameterValue_{std::nan("unset")};
  double _priorValue_{std::nan("unset")};
  double _throwValue_{std::nan("unset")};
  double _stdDevValue_{std::nan("unset")};
  double _minValue_{std::nan("unset")};
  double _maxValue_{std::nan("unset")};
  double _minMirror_{std::nan("unset")};
  double _maxMirror_{std::nan("unset")};
  double _minPhysical_{std::nan("unset")};
  double _maxPhysical_{std::nan("unset")};
  double _stepSize_{std::nan("unset")};
  std::string _name_{};
  std::string _dialsWorkingDirectory_{"."};
  JsonType _parameterConfig_{};
  JsonType _dialDefinitionsList_{};

  // Internals
  const ParameterSet* _owner_{nullptr};
  PriorType _priorType_{PriorType::Gaussian};

};


#endif //GUNDAM_PARAMETER_H
