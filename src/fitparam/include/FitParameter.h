//
// Created by Nadrino on 21/05/2021.
//

#ifndef XSLLHFITTER_FITPARAMETER_H
#define XSLLHFITTER_FITPARAMETER_H

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
  void setEnableDialSetsSummary(bool enableDialSetsSummary);
  void setDialsWorkingDirectory(const std::string &dialsWorkingDirectory);

  void initialize();

  // Getters
  bool isEnabled() const;
  bool isFixed() const;
  int getParameterIndex() const;
  const std::string &getName() const;
  double getParameterValue() const;
  double getStdDevValue() const;
  double getPriorValue() const;
  const std::vector<DialSet> &getDialSetList() const;

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
  nlohmann::json _dialDefinitionsList_;
  bool _enableDialSetsSummary_;
  std::string _dialsWorkingDirectory_;
  bool _isEnabled_{true};
  bool _isFixed_{false};

  // Internals
  std::vector<DialSet> _dialSetList_; // one dial set per detector

};


#endif //XSLLHFITTER_FITPARAMETER_H
