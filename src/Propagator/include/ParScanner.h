//
// Created by Adrien BLANCHET on 07/04/2022.
//

#ifndef GUNDAM_PARSCANNER_H
#define GUNDAM_PARSCANNER_H

#include "JsonBaseClass.h"
#include "Parameter.h"

#include "TDirectory.h"
#include "TGraph.h"
#include "nlohmann/json.hpp"

#include <utility>

class Propagator;

struct ScanData{
  std::string folder{};
  std::string title{};
  std::string yTitle{};
  std::vector<double> yPoints{};
  std::function<double()> evalY{};
};

struct GraphEntry{
  ScanData* scanDataPtr{nullptr};
  Parameter* fitParPtr{nullptr};
  TGraph graph{};
};

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
  [[nodiscard]] const JsonType &getVarsConfig() const { return _varsConfig_; };
  [[nodiscard]] const std::vector<GraphEntry> &getGraphEntriesBuf() const { return _graphEntriesBuf_; };

  // Core
  void scanFitParameters(std::vector<Parameter>& par_, TDirectory* saveDir_);
  void scanFitParameter(Parameter& par_, TDirectory* saveDir_);
  void scanSegment(TDirectory *saveDir_, const JsonType &end_, const JsonType &start_ = JsonType(), int nSteps_=-1);
  void generateOneSigmaPlots(TDirectory* saveDir_);
  void varyEvenRates(const std::vector<double>& paramVariationList_, TDirectory* saveDir_);

  static void muteLogger();
  static void unmuteLogger();

  static void writeGraphEntry(GraphEntry& entry_, TDirectory* saveDir_);


protected:
  void readConfigImpl() override;
  void initializeImpl() override;

private:
  // Parameters
  bool _useParameterLimits_{true};
  int _nbPoints_{100};
  int _nbPointsLineScan_{_nbPoints_};
  std::pair<double, double> _parameterSigmaRange_{-3,3};
  JsonType _varsConfig_{};

  // Internals
  Propagator* _owner_{nullptr};

  std::vector<ScanData> _scanDataDict_;
  std::vector<GraphEntry> _graphEntriesBuf_;


};


#endif //GUNDAM_PARSCANNER_H
