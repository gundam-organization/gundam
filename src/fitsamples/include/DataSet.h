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
  void addRequestedLeafName(const std::string& leafName_);

  void initialize();

  bool isEnabled() const;
  const std::string &getName() const;
  std::shared_ptr<TChain> &getMcChain();
  std::shared_ptr<TChain> &getDataChain();
  const std::vector<std::string> &getEnabledLeafNameList() const;
  const std::vector<std::string> &getMcFilePathList() const;
  const std::vector<std::string> &getDataFilePathList() const;

  // Misc
  void print();

protected:
  void initializeChains();

private:
  nlohmann::json _config_;

  // internals
  bool _isEnabled_{false};
  std::string _name_;

  std::vector<std::string> _enabledLeafNameList_;

  std::string _mcTreeName_;
  std::vector<std::string> _mcFilePathList_;
  std::shared_ptr<TChain> _mcChain_{nullptr};

  std::string _dataTreeName_;
  std::vector<std::string> _dataFilePathList_;
  std::shared_ptr<TChain> _dataChain_{nullptr};


};


#endif //XSLLHFITTER_DATASET_H
