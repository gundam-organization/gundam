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

class Propagator;

class ParScanner : public JsonBaseClass {

public:
  explicit ParScanner(Propagator* owner_);

  // Setters
  void setOwner(Propagator *owner);
  void setNbPoints(int nbPoints);
  void setNbPointsLineScan(int nbPointsLineScan);

  // Getters
  [[nodiscard]] bool isUseParameterLimits() const;
  [[nodiscard]] int getNbPoints() const;
  [[nodiscard]] const std::pair<double, double> &getParameterSigmaRange() const;
  [[nodiscard]] const nlohmann::json &getVarsConfig() const { return _varsConfig_; };

  // Core
  void scanFitParameters(std::vector<FitParameter>& par_, TDirectory* saveDir_);
  void scanFitParameter(FitParameter& par_, TDirectory* saveDir_);
  void scanSegment(TDirectory *saveDir_, const nlohmann::json &end_, const nlohmann::json &start_ = nlohmann::json());
  void generateOneSigmaPlots(TDirectory* saveDir_);
  void varyEvenRates(const std::vector<double>& paramVariationList_, TDirectory* saveDir_);


protected:
  void readConfigImpl() override;
  void initializeImpl() override;

private:
  // Parameters
  bool _useParameterLimits_{true};
  int _nbPoints_{100};
  int _nbPointsLineScan_{_nbPoints_};
  std::pair<double, double> _parameterSigmaRange_{-3,3};
  nlohmann::json _varsConfig_{};

  // Internals
  Propagator* _owner_{nullptr};
  struct ScanData{
    std::string folder{};
    std::string title{};
    std::string yTitle{};
    std::vector<double> yPoints{};
    std::function<double()> evalY{};

    void reset(){

    }
  };
  std::vector<ScanData> _scanDataDict_;


};


#endif //GUNDAM_PARSCANNER_H
