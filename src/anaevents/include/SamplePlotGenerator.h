//
// Created by Adrien BLANCHET on 16/06/2021.
//

#ifndef XSLLHFITTER_SAMPLEPLOTGENERATOR_H
#define XSLLHFITTER_SAMPLEPLOTGENERATOR_H

#include "json.hpp"
#include "TDirectory.h"

#include "AnaSample.hh"

class SamplePlotGenerator {

public:
  SamplePlotGenerator();
  virtual ~SamplePlotGenerator();

  // Reset
  void reset();

  // Setters
  void setSaveTDirectory(TDirectory *saveTDirectory_);
  void setConfig(const nlohmann::json &config);

  // Init
  void initialize();

  void saveSamplePlots(TDirectory *saveTDirectory_, const std::vector<AnaSample> &sampleList_);

private:
  TDirectory* _saveTDirectory_{nullptr};
  nlohmann::json _config_;

  // Internals
  nlohmann::json _varDictionary_;
  nlohmann::json _canvasParameters_;
  nlohmann::json _histogramsDefinition_;
  std::vector<Color_t> defaultColorWheel;


};


#endif //XSLLHFITTER_SAMPLEPLOTGENERATOR_H
