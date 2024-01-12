//
// Created by Nadrino on 22/07/2021.
//

#ifndef GUNDAM_SAMPLESET_H
#define GUNDAM_SAMPLESET_H

#include "Sample.h"
#include "ParameterSet.h"
#include "Likelihoods.hh"
#include "JointProbability.h"
#include "JsonBaseClass.h"

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
  const std::vector<Sample> &getSampleList() const { return _sampleList_; }
  const std::vector<std::string>& getAdditionalVariablesForStorage() const { return _additionalVariablesForStorage_; }

  // mutable getters
  std::vector<Sample> &getSampleList(){ return _sampleList_; }
  std::vector<std::string>& getAdditionalVariablesForStorage() { return _additionalVariablesForStorage_; }

  //Core
  [[nodiscard]] bool empty() const{ return _sampleList_.empty(); }

  // Parallel
  void updateSampleEventBinIndexes() const;
  void updateSampleBinEventList() const;
  void updateSampleHistograms() const;

private:
  bool _showTimeStats_{false};
  std::vector<Sample> _sampleList_;

  std::vector<std::string> _eventByEventDialLeafList_;
  std::vector<std::string> _additionalVariablesForStorage_;

};


#endif //GUNDAM_SAMPLESET_H
