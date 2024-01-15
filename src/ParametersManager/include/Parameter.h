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
  void setMinValue(double minValue){ _minValue_ = minValue; }
  void setMaxValue(double maxValue){ _maxValue_ = maxValue; }
  // Record the mirroring being used by any dials.
  void setMinMirror(double minMirror);
  void setMaxMirror(double maxMirror);
  // Record the physical bounds for the parameter.  This is the range where
  // the parameter has a physically meaningful value.
  void setMinPhysical(double minPhysical){ _minPhysical_ = minPhysical; }
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
  [[nodiscard]] bool gotUpdated() const { return _gotUpdated_; }
  [[nodiscard]] int getParameterIndex() const{ return _parameterIndex_; }
  [[nodiscard]] double getStepSize() const{ return _stepSize_; }
  [[nodiscard]] double getMinValue() const{ return _minValue_; }
  [[nodiscard]] double getMaxValue() const{ return _maxValue_; }
  [[nodiscard]] double getMinMirror() const{ return _minMirror_; }
  [[nodiscard]] double getMaxMirror() const{ return _maxMirror_; }
  [[nodiscard]] double getPriorValue() const{ return _priorValue_; }
  [[nodiscard]] double getThrowValue() const{ return _throwValue_; }
  [[nodiscard]] double getMinPhysical() const{ return _minPhysical_; }
  [[nodiscard]] double getMaxPhysical() const{ return _maxPhysical_; }
  [[nodiscard]] double getStdDevValue() const{ return _stdDevValue_; }
  [[nodiscard]] double getParameterValue() const{ return _parameterValue_; }
  [[nodiscard]] const std::string &getName() const{ return _name_; }
  [[nodiscard]] const JsonType &getDialDefinitionsList() const{ return _dialDefinitionsList_; }
  [[nodiscard]] const ParameterSet *getOwner() const{ return _owner_; }
  [[nodiscard]] PriorType getPriorType() const{ return _priorType_; }

  // Core
  void setValueAtPrior();
  void setCurrentValueAsPrior();
  [[nodiscard]] bool isValueWithinBounds() const;
  [[nodiscard]] double getDistanceFromNominal() const; // in unit of sigmas
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
