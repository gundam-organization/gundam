//
// Created by Adrien BLANCHET on 07/04/2022.
//

#ifndef GUNDAM_SCANCONFIG_H
#define GUNDAM_SCANCONFIG_H

#include "json.hpp"

#include "utility"

class ScanConfig {

public:
  ScanConfig();
  explicit ScanConfig(nlohmann::json config_);
  virtual ~ScanConfig();

  const nlohmann::json &getVarsConfig() const;

  int getNbPoints() const;
  const std::pair<double, double> &getParameterSigmaRange() const;
  bool isUseParameterLimits() const;

protected:
  void readConfig();

private:
  nlohmann::json _config_{};
  nlohmann::json _varsConfig_{};

  int _nbPoints_{100};
  std::pair<double, double> _parameterSigmaRange_{-3,3};
  bool _useParameterLimits_{true};


};


#endif //GUNDAM_SCANCONFIG_H
