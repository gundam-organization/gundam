//
// Created by Nadrino on 22/07/2021.
//

#ifndef GUNDAM_FITSAMPLE_H
#define GUNDAM_FITSAMPLE_H


#include "SampleElement.h"
#include "DataBinSet.h"

#include "nlohmann/json.hpp"
#include <TH1D.h>
#include <TTreeFormula.h>

#include "vector"
#include "string"
#include "memory"


class FitSample {

public:
  FitSample();
  virtual ~FitSample();

  void reset();

  // SETTERS
  void setConfig(const nlohmann::json &config_);

  // INIT
  void initialize();

  // GETTERS
  bool isEnabled() const;
  const std::string &getName() const;
  const std::string &getSelectionCutsStr() const;
  const DataBinSet &getBinning() const;
  const SampleElement &getMcContainer() const;
  const SampleElement &getDataContainer() const;
  SampleElement &getMcContainer();
  SampleElement &getDataContainer();

  // Misc
  bool isDatasetValid(const std::string& datasetName_);

private:
  // Yaml
  nlohmann::json _config_;
  bool _isEnabled_{false};
  std::string _name_;
  std::string _selectionCuts_;
  std::vector<std::string> _enabledDatasetList_;
  double _mcNorm_{1};
  double _dataNorm_{1};

  // Internals
  DataBinSet _binning_;
  SampleElement _mcContainer_;
  SampleElement _dataContainer_;
  std::vector<size_t> _dataSetIndexList_;

};


#endif //GUNDAM_FITSAMPLE_H
