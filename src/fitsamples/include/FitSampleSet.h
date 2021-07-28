//
// Created by Nadrino on 22/07/2021.
//

#ifndef XSLLHFITTER_FITSAMPLESET_H
#define XSLLHFITTER_FITSAMPLESET_H

#include "json.hpp"

#include "GenericToolbox.h"

#include "FitSample.h"
#include "DataSet.h"

ENUM_EXPANDER(
  DataEventType, -1,
  Unset,
  Asimov,
  DataFiles
)


class FitSampleSet {

public:
  FitSampleSet();
  virtual ~FitSampleSet();

  void reset();

  void setConfig(const nlohmann::json &config);

  void initialize();

  std::vector<FitSample> &getFitSampleList();
  std::vector<DataSet> &getDataSetList();

  // Core
  void loadPhysicsEvents();

private:
  bool _isInitialized_{false};
  nlohmann::json _config_;
  DataEventType _dataEventType_{DataEventType::Unset};

  std::vector<FitSample> _fitSampleList_;
  std::vector<DataSet> _dataSetList_;

};


#endif //XSLLHFITTER_FITSAMPLESET_H
