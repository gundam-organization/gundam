//
// Created by Adrien BLANCHET on 07/04/2022.
//

#ifndef GUNDAM_PARSCANNER_H
#define GUNDAM_PARSCANNER_H

#include "ConfigBasedClass.h"

#include "nlohmann/json.hpp"

#include "utility"

class ParScanner : public ConfigBasedClass {

public:
  ParScanner() = default;
  explicit ParScanner(const nlohmann::json& config_){ this->readConfig(config_); }

  void setNbPoints(int nbPoints);

  const nlohmann::json &getVarsConfig() const { return _varsConfig_; };
  int getNbPoints() const;
  const std::pair<double, double> &getParameterSigmaRange() const;
  bool isUseParameterLimits() const;

protected:
  void readConfigImpl() override;

private:
  nlohmann::json _varsConfig_{};
  int _nbPoints_{100};
  std::pair<double, double> _parameterSigmaRange_{-3,3};
  bool _useParameterLimits_{true};

  struct ScanData{
    std::string folder{};
    std::string title{};
    std::string yTitle{};
    std::vector<double> yPoints{};
    std::function<double()> evalY{};
  };
  std::vector<ScanData> scanDataDict;


};


#endif //GUNDAM_PARSCANNER_H
