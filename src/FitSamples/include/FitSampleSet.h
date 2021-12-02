//
// Created by Nadrino on 22/07/2021.
//

#ifndef GUNDAM_FITSAMPLESET_H
#define GUNDAM_FITSAMPLESET_H

#include "json.hpp"

#include "GenericToolbox.h"

#include "FitSample.h"
#include "FitParameterSet.h"
#include "Likelihoods.hh"

ENUM_EXPANDER(
  DataEventType, -1,
  Unset,
  Asimov,
  DataFiles,
  FakeData
)


class FitSampleSet {

public:
  FitSampleSet();
  virtual ~FitSampleSet();

  void reset();

  void setConfig(const nlohmann::json &config);

  void addEventByEventDialLeafName(const std::string& leafName_);

  void initialize();

  // Post init
  void loadAsimovData();

  // Getters
  DataEventType getDataEventType() const;
  const std::vector<FitSample> &getFitSampleList() const;
  std::vector<FitSample> &getFitSampleList();

  const nlohmann::json &getConfig() const;

  //Core
  bool empty() const;
  double evalLikelihood() const;
  void writeSampleEvents(TDirectory* saveDir_) const;

  // Parallel
  void updateSampleEventBinIndexes() const;
  void updateSampleBinEventList() const;
  void updateSampleHistograms() const;

private:
  bool _isInitialized_{false};
  bool _showTimeStats_{false};
  nlohmann::json _config_;
  DataEventType _dataEventType_{DataEventType::Unset};

  std::vector<FitSample> _fitSampleList_;

  std::shared_ptr<CalcLLHFunc> _likelihoodFunctionPtr_{nullptr};

  std::vector<std::string> _eventByEventDialLeafList_;

};


#endif //GUNDAM_FITSAMPLESET_H
