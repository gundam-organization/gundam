//
// Created by Nadrino on 21/05/2021.
//

#ifndef GUNDAM_FITPARAMETER_H
#define GUNDAM_FITPARAMETER_H

#include "vector"
#include "string"
#include "json.hpp"
#include "GenericToolbox.h"

#include "DialSet.h"

class FitParameter {

public:
  FitParameter();
  virtual ~FitParameter();

  void reset();

  void setIsFixed(bool isFixed);
  void setName(const std::string &name);
  void setParameterIndex(int parameterIndex);
  void setParameterValue(double parameterValue);
  void setPriorValue(double priorValue);
  void setStdDevValue(double stdDevValue);
  void setDialSetConfig(const nlohmann::json &jsonConfig_);
  void setParameterDefinitionConfig(const nlohmann::json &config_);
  void setEnableDialSetsSummary(bool enableDialSetsSummary);
  void setDialsWorkingDirectory(const std::string &dialsWorkingDirectory);
  void setMinValue(double minValue);
  void setMaxValue(double maxValue);
  void setStepSize(double stepSize);

  void initialize();

  // Getters
  bool isEnabled() const;
  bool isFixed() const;
  int getParameterIndex() const;
  const std::string &getName() const;
  double getParameterValue() const;
  double getStdDevValue() const;
  double getPriorValue() const;
  std::vector<DialSet> &getDialSetList();
  double getMinValue() const;
  double getMaxValue() const;
  double getStepSize() const;

  // Core
  double getDistanceFromNominal() const; // in unit of sigmas
  DialSet* findDialSet(const std::string& dataSetName_);
  std::string getSummary() const;
  std::string getTitle() const;

private:
  // Parameters
  std::string _name_;
  int _parameterIndex_{-1}; // to get the right definition in the json config (in case "name" is not specified)
  double _parameterValue_{};
  double _priorValue_{};
  double _stdDevValue_{};
  double _minValue_{std::nan("UNSET")};
  double _maxValue_{std::nan("UNSET")};
  double _stepSize_{std::nan("UNSET")};
  nlohmann::json _parameterConfig_;
  nlohmann::json _dialDefinitionsList_;
  bool _enableDialSetsSummary_;
  std::string _dialsWorkingDirectory_;
  bool _isEnabled_{true};
  bool _isFixed_{false};

  // Internals
  std::vector<DialSet> _dialSetList_; // one dial set per detector

};


#endif //GUNDAM_FITPARAMETER_H
