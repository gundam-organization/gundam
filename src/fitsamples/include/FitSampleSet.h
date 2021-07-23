//
// Created by Adrien BLANCHET on 22/07/2021.
//

#ifndef XSLLHFITTER_FITSAMPLESET_H
#define XSLLHFITTER_FITSAMPLESET_H

#include "json.hpp"

#include "FitSample.h"
#include "DataSet.h"


class FitSampleSet {

public:
  FitSampleSet();
  virtual ~FitSampleSet();

  void reset();

  void setConfig(const nlohmann::json &config);

  void initialize();

  const std::vector<FitSample> &getFitSampleList() const;
  std::vector<DataSet> &getDataSetList();

private:
  nlohmann::json _config_;

  std::vector<FitSample> _fitSampleList_;
  std::vector<DataSet> _dataSetList_;

};


#endif //XSLLHFITTER_FITSAMPLESET_H
