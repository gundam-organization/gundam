//
// Created by Nadrino on 21/05/2021.
//

#ifndef GUNDAM_FITPARAMETER_H
#define GUNDAM_FITPARAMETER_H

#include "JsonBaseClass.h"

#include "GenericToolbox.h"

#include "nlohmann/json.hpp"

#include <vector>
#include <string>

namespace PriorType{
  ENUM_EXPANDER(
    PriorType, -1,
    Unset,
    Gaussian,
    Flat
  );
}

class FitParameterSet;

class FitParameter : public JsonBaseClass {

public:
  explicit FitParameter(const FitParameterSet* owner_);

  void setIsEnabled(bool isEnabled);
  void setIsFixed(bool isFixed);
  void setIsEigen(bool isEigen);
  void setIsFree(bool isFree);
  void setName(const std::string &name);
  void setParameterIndex(int parameterIndex);
  void setParameterValue(double parameterValue);
  void setPriorValue(double priorValue);
  void setThrowValue(double throwValue);
  void setStdDevValue(double stdDevValue);
  void setDialSetConfig(const nlohmann::json &jsonConfig_);
  void setParameterDefinitionConfig(const nlohmann::json &config_);
  void setMinValue(double minValue);
  void setMaxValue(double maxValue);
  void setStepSize(double stepSize);
  void setOwner(const FitParameterSet *owner_);
  void setPriorType(PriorType::PriorType priorType);

  // Record the mirroring being used by any dials.
  void setMinMirror(double minMirror);
  void setMaxMirror(double maxMirror);

  // Record the physical bounds for the parameter.  This is the range where
  // the parameter has a physically meaningful value.
  void setMinPhysical(double minPhysical);
  void setMaxPhysical(double maxPhysical);

  void setValueAtPrior();
  void setCurrentValueAsPrior();


  // Getters
  [[nodiscard]] bool isEnabled() const;
  [[nodiscard]] bool isFixed() const;
  [[nodiscard]] bool isEigen() const;
  [[nodiscard]] bool isFree() const;
  [[nodiscard]] bool gotUpdated() const { return _gotUpdated_; }
  [[nodiscard]] int getParameterIndex() const;
  [[nodiscard]] double getMinValue() const;
  [[nodiscard]] double getMaxValue() const;
  [[nodiscard]] double getMinMirror() const;
  [[nodiscard]] double getMaxMirror() const;
  [[nodiscard]] double getMinPhysical() const;
  [[nodiscard]] double getMaxPhysical() const;
  [[nodiscard]] double getStepSize() const;
  [[nodiscard]] double getPriorValue() const;
  [[nodiscard]] double getThrowValue() const;
  [[nodiscard]] double getStdDevValue() const;
  [[nodiscard]] double getParameterValue() const;
  [[nodiscard]] const std::string &getName() const;
  [[nodiscard]] const nlohmann::json &getDialDefinitionsList() const;
  [[nodiscard]] const FitParameterSet *getOwner() const;
  [[nodiscard]] PriorType::PriorType getPriorType() const;

  // Core
  [[nodiscard]] double getDistanceFromNominal() const; // in unit of sigmas
  [[nodiscard]] std::string getSummary(bool shallow_=false) const;
  [[nodiscard]] std::string getTitle() const;
  [[nodiscard]] std::string getFullTitle() const;

protected:
  void readConfigImpl() override;
  void initializeImpl() override;

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
  nlohmann::json _parameterConfig_{};
  nlohmann::json _dialDefinitionsList_{};

  // Internals
  const FitParameterSet* _owner_{nullptr};
  PriorType::PriorType _priorType_{PriorType::Gaussian};

};


#endif //GUNDAM_FITPARAMETER_H
