//
// Created by Nadrino on 22/07/2021.
//

#ifndef GUNDAM_FITSAMPLESET_H
#define GUNDAM_FITSAMPLESET_H

#include "FitSample.h"
#include "FitParameterSet.h"
#include "Likelihoods.hh"
#include "JointProbability.h"
#include "JsonBaseClass.h"

#include "GenericToolbox.h"
#include "nlohmann/json.hpp"

#include "string"
#include "vector"


class FitSampleSet : public JsonBaseClass {

public:
  // Post init
  void copyMcEventListToDataContainer();
  void clearMcContainers();

  // Getters
  const std::vector<FitSample> &getFitSampleList() const;
  std::vector<FitSample> &getFitSampleList();
  const nlohmann::json &getConfig() const;
  const std::shared_ptr<JointProbability::JointProbability> &getJointProbabilityFct() const;

  //Core
  bool empty() const;
  double evalLikelihood() const;
  double evalLikelihood(const FitSample& sample_) const;

  // Parallel
  void updateSampleEventBinIndexes() const;
  void updateSampleBinEventList() const;
  void updateSampleHistograms() const;

protected:
  void readConfigImpl() override;
  void initializeImpl() override;

private:
  bool _showTimeStats_{false};
  std::vector<FitSample> _fitSampleList_;
  std::shared_ptr<JointProbability::JointProbability> _jointProbabilityPtr_{nullptr};
  std::vector<std::string> _eventByEventDialLeafList_;

};


#endif //GUNDAM_FITSAMPLESET_H
