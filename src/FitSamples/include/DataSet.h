//
// Created by Nadrino on 22/07/2021.
//

#ifndef XSLLHFITTER_DATASET_H
#define XSLLHFITTER_DATASET_H

#include <TChain.h>
#include "json.hpp"

#include "PhysicsEvent.h"


class DataSet {

public:
  DataSet();
  virtual ~DataSet();

  void reset();

  void setConfig(const nlohmann::json &config_);
  void addRequestedLeafName(const std::string& leafName_); // some variables might not be present in data TChain (true vars)
  void addRequestedMandatoryLeafName(const std::string& leafName_); // specify which var should be present in data TChain

  void initialize();

  bool isEnabled() const;
  const std::string &getName() const;
  std::vector<std::string> &getMcActiveLeafNameList();
  std::vector<std::string> &getDataActiveLeafNameList();
  const std::string &getMcNominalWeightFormulaStr() const;
  const std::vector<std::string> &getRequestedLeafNameList() const;
  const std::vector<std::string> &getRequestedMandatoryLeafNameList() const;
  const std::vector<std::string> &getMcFilePathList() const;
  const std::vector<std::string> &getDataFilePathList() const;

  // Misc
  TChain* buildChain(bool isData_);
  TChain* buildMcChain();
  TChain* buildDataChain();
  void print();

private:
  nlohmann::json _config_;

  // internals
  bool _isInitialized_{false};
  bool _isEnabled_{false};
  std::string _name_;

  std::vector<std::string> _requestedLeafNameList_;
  std::vector<std::string> _requestedMandatoryLeafNameList_; // Mandatory variables for data (sample binning, cuts, nominal weight if set)

  std::string _mcTreeName_;
  std::string _mcNominalWeightFormulaStr_{"1"};
  std::vector<std::string> _mcActiveLeafNameList_;
  std::vector<std::string> _mcFilePathList_;

  std::string _dataTreeName_;
  std::vector<std::string> _dataActiveLeafNameList_;
  std::vector<std::string> _dataFilePathList_;

};


#endif //XSLLHFITTER_DATASET_H
