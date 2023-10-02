//
// Created by Nadrino on 22/07/2021.
//

#ifndef GUNDAM_DATASETLOADER_H
#define GUNDAM_DATASETLOADER_H

#include "DataDispenser.h"
#include "FitParameterSet.h"
#include <FitSampleSet.h>
#include "PlotGenerator.h"
#include "JsonBaseClass.h"

#include <TChain.h>
#include "nlohmann/json.hpp"

#include <vector>
#include <string>


class DatasetLoader : public JsonBaseClass {

protected:
  void readConfigImpl() override;
  void initializeImpl() override;

public:
  explicit DatasetLoader(const nlohmann::json& config_, int datasetIndex_): _dataSetIndex_(datasetIndex_) { this->readConfig(config_); }

  void setDataSetIndex(int dataSetIndex){ _dataSetIndex_ = dataSetIndex; }
  void setSelectedDataEntry(const std::string &selectedDataEntry){ _selectedDataEntry_ = selectedDataEntry; }

  [[nodiscard]] bool isEnabled() const{ return _isEnabled_; }
  [[nodiscard]] bool isSortLoadedEvents() const{ return _sortLoadedEvents_; }
  [[nodiscard]] bool isShowSelectedEventCount() const{ return _showSelectedEventCount_; }
  [[nodiscard]] bool isDevSingleThreadEventSelection() const{ return _devSingleThreadEventSelection_; }
  [[nodiscard]] bool isDevSingleThreadEventLoaderAndIndexer() const{ return _devSingleThreadEventLoaderAndIndexer_; }
  [[nodiscard]] int getDataSetIndex() const{ return _dataSetIndex_; }
  [[nodiscard]] const std::string &getName() const{ return _name_; }
  [[nodiscard]] const std::string &getToyDataEntry() const{ return _selectedToyEntry_; }
  [[nodiscard]] const std::string &getSelectedDataEntry() const{ return _selectedDataEntry_; }

  DataDispenser &getMcDispenser(){ return _mcDispenser_; }
  DataDispenser &getToyDataDispenser(){ return _dataDispenserDict_.at(_selectedToyEntry_); }
  DataDispenser &getSelectedDataDispenser(){ return _dataDispenserDict_.at(_selectedDataEntry_); }
  std::map<std::string, DataDispenser> &getDataDispenserDict(){ return _dataDispenserDict_; }

  void updateDispenserOwnership();

private:
  // config
  bool _isEnabled_{false};
  bool _showSelectedEventCount_{true};
  int _dataSetIndex_{-1};
  std::string _name_{};
  std::string _selectedDataEntry_{"Asimov"};
  std::string _selectedToyEntry_{"Asimov"};

  bool _sortLoadedEvents_{true}; // needed for reproducibility of toys in stat throw
  bool _devSingleThreadEventLoaderAndIndexer_{false};
  bool _devSingleThreadEventSelection_{false};

  // internals
  DataDispenser _mcDispenser_{this};
  std::map<std::string, DataDispenser> _dataDispenserDict_;

};


#endif //GUNDAM_DATASETLOADER_H
