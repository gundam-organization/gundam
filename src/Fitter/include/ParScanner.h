//
// Created by Adrien BLANCHET on 07/04/2022.
//

#ifndef GUNDAM_PARSCANNER_H
#define GUNDAM_PARSCANNER_H

#include "JsonBaseClass.h"
#include "FitParameter.h"

#include "TDirectory.h"
#include "nlohmann/json.hpp"

#include "utility"

class FitterEngine;

class ParScanner : public JsonBaseClass {

public:
  ParScanner() = default;
  explicit ParScanner(const nlohmann::json& config_){ this->readConfig(config_); }

  // Setters
  void setOwner(FitterEngine *owner);
  void setSaveDir(TDirectory *saveDir);
  void setNbPoints(int nbPoints);

  // Getters
  const nlohmann::json &getVarsConfig() const { return _varsConfig_; };
  int getNbPoints() const;
  const std::pair<double, double> &getParameterSigmaRange() const;
  bool isUseParameterLimits() const;

  // Core
  void scanMinimizerParameters(const std::string& saveSubdir_ = "");
  void scanFitParameters(std::vector<FitParameter>& par_, const std::string& saveSubdir_ = "");
  void scanFitParameter(FitParameter& par_, const std::string& saveSubdir_ = "");
  void generateOneSigmaPlots(const std::string& savePath_ = "");
  void varyEvenRates(const std::vector<double>& paramVariationList_, const std::string& savePath_ = "");


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
