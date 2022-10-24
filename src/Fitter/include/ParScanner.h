//
// Created by Adrien BLANCHET on 07/04/2022.
//

#ifndef GUNDAM_PARSCANNER_H
#define GUNDAM_PARSCANNER_H

#include "ConfigBasedClass.h"
#include "FitParameter.h"

#include "TDirectory.h"
#include "nlohmann/json.hpp"

#include "utility"

class FitterEngine;

class ParScanner : public ConfigBasedClass {

public:
  ParScanner() = default;
  explicit ParScanner(const nlohmann::json& config_){ this->readConfig(config_); }

  void setOwner(FitterEngine *owner);
  void setSaveDir(TDirectory *saveDir);
  void setNbPoints(int nbPoints);

  const nlohmann::json &getVarsConfig() const { return _varsConfig_; };
  int getNbPoints() const;
  const std::pair<double, double> &getParameterSigmaRange() const;
  bool isUseParameterLimits() const;

  void scanFitParameters(const std::string& saveSubdir_ = "");
  void scanParameter(FitParameter& par_, const std::string& saveSubdir_ = "");

protected:
  void readConfigImpl() override;
  void initializeImpl() override;

private:
  // Parameters
  bool _useParameterLimits_{true};
  int _nbPoints_{100};
  std::pair<double, double> _parameterSigmaRange_{-3,3};
  nlohmann::json _varsConfig_{};

  // Internals
  FitterEngine* _owner_{nullptr};
  TDirectory* _saveDir_{nullptr};
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
