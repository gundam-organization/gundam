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

public:
  DatasetLoader(const nlohmann::json& config_, int datasetIndex_);

  void setDataSetIndex(int dataSetIndex);

  void setSelectedDataEntry(const std::string &selectedDataEntry);

  [[nodiscard]] bool isEnabled() const;
  [[nodiscard]] bool isShowSelectedEventCount() const;
  [[nodiscard]] bool isDevSingleThreadEventLoaderAndIndexer() const;
  [[nodiscard]] bool isDevSingleThreadEventSelection() const;
  [[nodiscard]] bool isSortLoadedEvents() const;
  [[nodiscard]] int getDataSetIndex() const;
  [[nodiscard]] const std::string &getName() const;
  [[nodiscard]] const std::string &getSelectedDataEntry() const;
  [[nodiscard]] const std::string &getToyDataEntry() const;

  DataDispenser &getMcDispenser();
  DataDispenser &getSelectedDataDispenser();
  DataDispenser &getToyDataDispenser();
  std::map<std::string, DataDispenser> &getDataDispenserDict();

  void updateDispenserOwnership();

protected:
  void readConfigImpl() override;
  void initializeImpl() override;

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
