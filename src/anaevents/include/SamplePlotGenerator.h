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

  void saveSamplePlots(const std::vector<AnaSample*>& _sampleList_);

private:
  TDirectory* _saveTDirectory_{nullptr};
  nlohmann::json _config_;


};


#endif //XSLLHFITTER_SAMPLEPLOTGENERATOR_H
