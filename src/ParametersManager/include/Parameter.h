//
// Created by Nadrino on 21/05/2021.
//

#ifndef GUNDAM_PARAMETER_H
#define GUNDAM_PARAMETER_H

#include "ConfigUtils.h"

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
  // called through JsonBaseClass::configure() and JsonBaseClass::initialize()
  void configureImpl() override;
  void initializeImpl() override;

public:
  explicit Parameter(const ParameterSet* owner_): _owner_(owner_) {}
  Parameter() = delete; // Cannot be independently constructed.

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
  void setMinValue(double minValue){ _parameterLimits_.min = minValue; }

  /// Set the maximum value for this parameter.  Parameter values more than
  /// this value are illegal, and the likelihood is undefined. The job will
  /// terminate when it encounters an illegal parameter value.  Note: Using a
  /// minimum value when also using eigenvalue decomposition, or PCA results
  /// in undefined behavior because the decomposition will not honor the
  /// boundaries and may set take values that are out of bounds.  In this
  /// case, the job will stop.  Note: If the maximum value is not set, then
  /// the bound is at positive infinity.
  void setMaxValue(double maxValue){ _parameterLimits_.max = maxValue; }

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
  void setMinPhysical(double minPhysical){ _physicalLimits_.min = minPhysical; }

  /// Record the physical maximum bound for the parameter.  This is the range
  /// where the parameter has a physically meaningful value.  Because of
  /// numeric continuation, the likelihood may have a finite value outside of
  /// the physical range.  From a mathmatical perspective, the value of the
  /// LLH is infinite below the physical minimum.  This can be enforced
  /// using the Likelihood::SetParameterValidity() method.
  void setMaxPhysical(double maxPhysical){ _physicalLimits_.max = maxPhysical; }
  void setPriorValue(double priorValue){ _priorValue_ = priorValue; }
  void setThrowValue(double throwValue){ _throwValue_ = throwValue; }
  void setStdDevValue(double stdDevValue){ _stdDevValue_ = stdDevValue; }

  /// Set the parameter value.  This always checks the parameter validity, but
  /// if force is true, then it will only print warnings, otherwise it stops
  /// with EXIT_FAILURE.
  void setParameterValue(double parameterValue, bool force=false);
  void setName(const std::string &name){ _name_ = name; }
  void setDialSetConfig(const JsonType &jsonConfig_);
  void setOwner(const ParameterSet *owner_){ _owner_ = owner_; }
  void setPriorType(PriorType priorType){ _priorType_ = priorType; }
  void setMarginalised(bool isMarginalised){ _isMarginalised_ = isMarginalised; }

  /// Query if a value is in the domain of likelihood for this parameter.  Math
  /// remediation for those of us (including myself) who don't recall grammar
  /// school math: The DOMAIN of a function is the range over which it is
  /// defined.  For instance, the domain of the information transfer speed
  /// (dX/dT) is greater than or equal to zero.  To be in the domain, a value
  /// must not be NaN, and be between minValue and maxValue (if they are
  /// defined).
  [[nodiscard]] bool isInDomain(double value, bool verbose=false) const;

  /// Query if a value is in the range where the parameter will have a
  /// physically meaningful value.  For example, within special relativity, the
  /// information transfer speed is between zero and the speed of light.
  [[nodiscard]] bool isPhysical(double value) const;

  /// Query if a value will be mirrored.  This is true if the parameter value
  /// is not between the minimum and maximum mirror values.
  [[nodiscard]] bool isMirrored(double value) const;

  /// Query if a value matchs the validity requirements.
  [[nodiscard]] bool isValidValue(double value) const;

  [[nodiscard]] bool isFree() const{ return _isFree_; }
  [[nodiscard]] bool isFixed() const{ return _isFixed_; }
  [[nodiscard]] bool isEigen() const{ return _isEigen_; }
  [[nodiscard]] bool isEnabled() const{ return _isEnabled_; }
  [[nodiscard]] bool gotUpdated() const { return _gotUpdated_; }
  [[nodiscard]] int getParameterIndex() const{ return _parameterIndex_; }
  [[nodiscard]] double getStepSize() const{ return _stepSize_; }
  /// See setMinValue() for documentation.
  [[nodiscard]] double getMinValue() const{ return _parameterLimits_.min; }
  /// See setMaxValue() for documentation.
  [[nodiscard]] double getMaxValue() const{ return _parameterLimits_.max; }
  /// See setMinMirror for documentation.
  [[nodiscard]] double getMinMirror() const{ return _mirrorRange_.min; }
  /// See setMaxMirror for documentation.
  [[nodiscard]] double getMaxMirror() const{ return _mirrorRange_.max; }
  [[nodiscard]] double getPriorValue() const{ return _priorValue_; }
  [[nodiscard]] double getThrowValue() const{ return _throwValue_; }
  /// See setMinPhysical for documentation.
  [[nodiscard]] double getMinPhysical() const{ return _physicalLimits_.min; }
  /// See setMinPhysical for documentation.
  [[nodiscard]] double getMaxPhysical() const{ return _physicalLimits_.max; }
  [[nodiscard]] double getStdDevValue() const{ return _stdDevValue_; }
  [[nodiscard]] double getParameterValue() const;
  [[nodiscard]] const std::string &getName() const{ return _name_; }
  [[nodiscard]] const JsonType &getDialDefinitionsList() const{ return _dialDefinitionsList_; }
  [[nodiscard]] const ParameterSet *getOwner() const{ return _owner_; }
  [[nodiscard]] PriorType getPriorType() const{ return _priorType_; }
  [[nodiscard]] bool isMarginalised() const{ return _isMarginalised_; }

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

  [[nodiscard]] std::string getSummary() const;
  [[nodiscard]] std::string getTitle() const;
  [[nodiscard]] std::string getFullTitle() const;

  /// Define the type of validity that needs to be required by
  /// hasValidParameterValues.  This accepts a string with the possible values
  /// being:
  ///
  ///  "range" (default) -- Between the parameter minimum and maximum values.
  ///  "norange"         -- Do not require parameters in the valid range
  ///  "mirror"          -- Between the mirrored values (if parameter has
  ///                       mirroring).
  ///  "nomirror"        -- Do not require parameters in the mirrored range
  ///  "physical"        -- Only physically meaningful values.
  ///  "nophysical"      -- Do not require parameters in the physical range.
  ///
  /// Example: setParameterValidity("range,mirror,physical")
  void setValidity(const std::string& validity);
  void setValidity(int validity) {_validFlags_ = validity;}

  // print
  void printConfiguration() const;

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

  GenericToolbox::Range _parameterLimits_;
  GenericToolbox::Range _physicalLimits_;
  GenericToolbox::Range _mirrorRange_;

  double _stepSize_{std::nan("unset")};
  std::string _name_{};
  std::string _dialsWorkingDirectory_{"."};
  bool _isMarginalised_{false};
  JsonType _parameterConfig_{};
  JsonType _dialDefinitionsList_{};

  // Internals
  const ParameterSet* _owner_{nullptr};
  PriorType _priorType_{PriorType::Gaussian};

  /// A set of flags used to define if the parameter set has valid parameter
  /// values.
  /// "1" -- require valid parameters (Parameter::isInDomain will be true)
  /// "2" -- require in the mirrored range (is inside mirrored range).
  /// "4" -- require in the physical range (Parameter::isPhysical will be true)
  int _validFlags_{1};

};
#endif //GUNDAM_PARAMETER_H
