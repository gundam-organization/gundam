//
// Created by Nadrino on 22/07/2021.
//

#ifndef GUNDAM_DATASETLOADER_H
#define GUNDAM_DATASETLOADER_H

#include "DataDispenser.h"
#include "FitParameterSet.h"
#include <FitSampleSet.h>
#include "PlotGenerator.h"

#include <TChain.h>
#include "json.hpp"

#include "vector"
#include "string"


class DataSetLoader {

public:
  DataSetLoader();
  virtual ~DataSetLoader();

  void reset();

  void setConfig(const nlohmann::json &config_);
  void setDataSetIndex(int dataSetIndex);

  void initialize();

  bool isEnabled() const;
  const std::string &getName() const;
  int getDataSetIndex() const;

  const std::string &getSelectedDataEntry() const;
  const std::string &getToyDataEntry() const;

  DataDispenser &getMcDispenser();
  DataDispenser &getSelectedDataDispenser();
  DataDispenser &getToyDataDispenser();
  std::map<std::string, DataDispenser> &getDataDispenserDict();

private:
  nlohmann::json _config_;

  // internals
  bool _isInitialized_{false};
  bool _isEnabled_{false};
  int _dataSetIndex_{-1};
  std::string _name_;
  std::string _selectedDataEntry_{"Asimov"};
  std::string _selectedToyEntry_{"Asimov"};

  DataDispenser _mcDispenser_;
  std::map<std::string, DataDispenser> _dataDispenserDict_;

};


#endif //GUNDAM_DATASETLOADER_H
