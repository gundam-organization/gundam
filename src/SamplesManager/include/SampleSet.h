//
// Created by Nadrino on 22/07/2021.
//

#ifndef GUNDAM_SAMPLESET_H
#define GUNDAM_SAMPLESET_H

#include "Sample.h"
#include "ParameterSet.h"
#include "Likelihoods.hh"
#include "JsonBaseClass.h"

#include "GenericToolbox.Time.h"

#include "nlohmann/json.hpp"

#include <string>
#include <vector>


/// Hold a description of all of the event samples (both "data" and the
/// matching "MC") that are going to be managed by the Propagator.  The
/// samples in the set can be referred to by their sample set index.
class SampleSet : public JsonBaseClass {

protected:
  // called through public JsonBaseClass::readConfig() and JsonBaseClass::initialize()
  void readConfigImpl() override;
  void initializeImpl() override;

public:
  // Post init
  void copyMcEventListToDataContainer();
  void clearMcContainers();

  // const getters
  [[nodiscard]] const std::vector<Sample> &getSampleList() const { return _sampleList_; }
  [[nodiscard]] const std::vector<std::string>& getAdditionalVariablesForStorage() const { return _additionalVariablesForStorage_; }

  // non-const getters
  std::vector<Sample> &getSampleList(){ return _sampleList_; }
  std::vector<std::string>& getAdditionalVariablesForStorage() { return _additionalVariablesForStorage_; }

  //Core
  [[nodiscard]] bool empty() const{ return _sampleList_.empty(); }
  double evalLikelihood();
  double evalLikelihood(Sample& sample_);

  // Parallel
  void updateSampleEventBinIndexes() const;
  void updateSampleBinEventList() const;
  void updateSampleHistograms() const;

  // Deprecated
  [[deprecated("use getSampleList()")]] std::vector<Sample> &getFitSampleList(){ return getSampleList(); }
  [[deprecated("use getSampleList()")]] [[nodiscard]] const std::vector<Sample> &getFitSampleList() const { return getSampleList(); }

private:
  // config
  bool _showTimeStats_{false};
  std::vector<Sample> _sampleList_{};

  // internals
  std::vector<std::string> _eventByEventDialLeafList_;
  std::vector<std::string> _additionalVariablesForStorage_;

};


#endif //GUNDAM_SAMPLESET_H
