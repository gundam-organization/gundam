//
// Created by Nadrino on 22/07/2021.
//

#ifndef GUNDAM_DATASETLOADER_H
#define GUNDAM_DATASETLOADER_H

#include "vector"
#include "string"

#include <TChain.h>
#include "json.hpp"

#include "FitParameterSet.h"
#include <FitSampleSet.h>
#include "PlotGenerator.h"


class DataSetLoader {

public:
  DataSetLoader();
  virtual ~DataSetLoader();

  void reset();

  void setConfig(const nlohmann::json &config_);
  void setDataSetIndex(int dataSetIndex);

  void addLeafRequestedForIndexing(const std::string& leafName_);
  void addLeafStorageRequestedForData(const std::string& leafName_);
  void addLeafStorageRequestedForMc(const std::string& leafName_);

  void initialize();

  bool isEnabled() const;
  const std::string &getName() const;
  std::vector<std::string> &getMcActiveLeafNameList();
  std::vector<std::string> &getDataActiveLeafNameList();
  const std::string &getMcNominalWeightFormulaStr() const;
  const std::string &getDataNominalWeightFormulaStr() const;
  const std::vector<std::string> &getMcFilePathList() const;
  const std::vector<std::string> &getDataFilePathList() const;

  // Core
  void load(FitSampleSet* sampleSetPtr_, std::vector<FitParameterSet>* parSetList_);

  // Misc
  TChain* buildChain(bool isData_);
  TChain* buildMcChain();
  TChain* buildDataChain();
  void print();

  void fetchRequestedLeaves(std::vector<FitParameterSet>* parSetList_);
  void fetchRequestedLeaves(FitSampleSet* sampleSetPtr_);
  void fetchRequestedLeaves(PlotGenerator* plotGenPtr_);

protected:
  std::vector<FitSample*> buildListOfSamplesToFill(FitSampleSet* sampleSetPtr_);
  std::vector<std::vector<bool>> makeEventSelection(std::vector<FitSample*>& samplesToFillList, bool loadData_);


private:
  nlohmann::json _config_;

  // internals
  bool _isInitialized_{false};
  bool _isEnabled_{false};
  int _dataSetIndex_{-1};
  std::string _name_;

  std::vector<std::string> _leavesRequestedForIndexing_;
  std::vector<std::string> _leavesStorageRequestedForData_;
  std::vector<std::string> _leavesStorageRequestedForMc_;

  std::string _mcTreeName_;
  std::string _mcNominalWeightFormulaStr_{"1"};
  std::vector<std::string> _mcActiveLeafNameList_;
  std::vector<std::string> _mcFilePathList_;

  std::string _dataTreeName_;
  std::string _dataNominalWeightFormulaStr_{"1"};
  std::vector<std::string> _dataActiveLeafNameList_;
  std::vector<std::string> _dataFilePathList_;

};


#endif //GUNDAM_DATASETLOADER_H
