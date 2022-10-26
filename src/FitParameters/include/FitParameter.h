//
// Created by Nadrino on 21/05/2021.
//

#ifndef GUNDAM_FITPARAMETER_H
#define GUNDAM_FITPARAMETER_H

#include "DialSet.h"
#include "JsonBaseClass.h"

#include "GenericToolbox.h"

#include "vector"
#include "string"
#include "nlohmann/json.hpp"

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
  void setEnableDialSetsSummary(bool enableDialSetsSummary);
  void setMinValue(double minValue);
  void setMaxValue(double maxValue);
  void setStepSize(double stepSize);
  void setOwner(const FitParameterSet *owner_);
  void setPriorType(PriorType::PriorType priorType);

  void setValueAtPrior();
  void setCurrentValueAsPrior();


  // Getters
  bool isEnabled() const;
  bool isFixed() const;
  bool isEigen() const;
  bool isFree() const;
  PriorType::PriorType getPriorType() const;
  int getParameterIndex() const;
  const std::string &getName() const;
  double getParameterValue() const;
  double getStdDevValue() const;
  double getPriorValue() const;
  double getThrowValue() const;
  std::vector<DialSet> &getDialSetList();
  double getMinValue() const;
  double getMaxValue() const;
  double getStepSize() const;

  const FitParameterSet *getOwner() const;

  // Core
  double getDistanceFromNominal() const; // in unit of sigmas
  DialSet* findDialSet(const std::string& dataSetName_);
  std::string getSummary() const;
  std::string getTitle() const;
  std::string getFullTitle() const;

protected:
  void readConfigImpl() override;
  void initializeImpl() override;

private:
  // Parameters
  bool _isEnabled_{true};
  bool _isFixed_{false};
  bool _isEigen_{false};
  bool _isFree_{false};
  bool _enableDialSetsSummary_{false};
  int _parameterIndex_{-1}; // to get the right definition in the json config (in case "name" is not specified)
  double _parameterValue_{std::nan("unset")};
  double _priorValue_{std::nan("unset")};
  double _throwValue_{std::nan("unset")};
  double _stdDevValue_{std::nan("unset")};
  double _minValue_{std::nan("unset")};
  double _maxValue_{std::nan("unset")};
  double _stepSize_{std::nan("unset")};
  std::string _name_{};
  std::string _dialsWorkingDirectory_{"."};
  nlohmann::json _parameterConfig_{};
  nlohmann::json _dialDefinitionsList_{};

  // Internals
  const FitParameterSet* _owner_{nullptr};
  std::vector<DialSet> _dialSetList_; // one dial set per detector
  PriorType::PriorType _priorType_{PriorType::Gaussian};


};


#endif //GUNDAM_FITPARAMETER_H
