//
// Created by Nadrino on 22/07/2021.
//

#ifndef GUNDAM_FITSAMPLESET_H
#define GUNDAM_FITSAMPLESET_H

#include "FitSample.h"
#include "FitParameterSet.h"
#include "Likelihoods.hh"

#include "GenericToolbox.h"
#include "nlohmann/json.hpp"

#include "string"
#include "vector"


class FitSampleSet {

public:
  FitSampleSet();
  virtual ~FitSampleSet();

  void reset();

  void setConfig(const nlohmann::json &config);

  void initialize();

  // Post init
  void copyMcEventListToDataContainer();
  void clearMcContainers();

  // Getters
  const std::vector<FitSample> &getFitSampleList() const;
  std::vector<FitSample> &getFitSampleList();
  const nlohmann::json &getConfig() const;
  const std::shared_ptr<CalcLLHFunc> &getLikelihoodFunctionPtr() const;

  //Core
  bool empty() const;
  double evalLikelihood() const;
  double evalLikelihood(const FitSample& sample_) const;

  // Parallel
  void updateSampleEventBinIndexes() const;
  void updateSampleBinEventList() const;
  void updateSampleHistograms() const;

private:
  bool _isInitialized_{false};
  bool _showTimeStats_{false};
  nlohmann::json _config_;

  std::vector<FitSample> _fitSampleList_;
  std::shared_ptr<CalcLLHFunc> _likelihoodFunctionPtr_{nullptr};
  std::vector<std::string> _eventByEventDialLeafList_;

};


#endif //GUNDAM_FITSAMPLESET_H
