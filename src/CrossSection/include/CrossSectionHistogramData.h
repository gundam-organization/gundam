//
// Created by Nadrino on 26/06/2025.
//

#ifndef CROSSSECTIONHISTOGRAMDATA_H
#define CROSSSECTIONHISTOGRAMDATA_H

#include "BinNormaliser.h"

#include "Sample.h"



struct CrossSectionHistogramData {

  Sample* samplePtr{nullptr};
  Sample* sampleDataPtr{nullptr};
  ConfigReader config{};
  GenericToolbox::RawDataArray branchBinsData{};

  TH1D histogram{};
  std::vector<BinNormaliser> normList{};

};



#endif //CROSSSECTIONHISTOGRAMDATA_H
