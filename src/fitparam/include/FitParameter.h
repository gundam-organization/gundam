//
// Created by Adrien BLANCHET on 21/05/2021.
//

#ifndef XSLLHFITTER_FITPARAMETER_H
#define XSLLHFITTER_FITPARAMETER_H

#include "vector"
#include "string"
#include "json.hpp"
#include "GenericToolbox.h"

#include "DialSet.h"
#include "AnaEvent.hh"

class FitParameter {

public:
  FitParameter();
  virtual ~FitParameter();

  void reset();

  void setName(const std::string &name);
  void setParameterIndex(int parameterIndex);
  void setParameterValue(double parameterValue);
  void setParameterDefinitionsConfig(const nlohmann::json &jsonConfig);

  void initialize();

  const std::string &getName() const;
  double getParameterValue() const;

  void selectDialSet(const std::string& dataSetName_);
  void reweightEvent(AnaEvent* eventPtr_);


private:
  // Parameters
  std::string _name_;
  int _parameterIndex_{-1}; // to get the right definition in the json config (in case "name" is not specified)
  double _parameterValue_;
  nlohmann::json _parameterDefinitionsConfig_;

  // Internals
  DialSet* _currentDialSetPtr_{nullptr};
  std::vector<DialSet> _dialSetList_; // one dial set per detector

};


#endif //XSLLHFITTER_FITPARAMETER_H
