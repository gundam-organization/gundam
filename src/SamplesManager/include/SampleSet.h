//
// Created by Nadrino on 22/07/2021.
//

#ifndef GUNDAM_SAMPLE_SET_H
#define GUNDAM_SAMPLE_SET_H

#include "Sample.h"
#include "ParameterSet.h"
#include "JsonBaseClass.h"

#include "GenericToolbox.Time.h"

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

  // mutable getters
  std::vector<Sample> &getSampleList(){ return _sampleList_; }

  // core
  [[nodiscard]] bool empty() const{ return _sampleList_.empty(); }
  [[nodiscard]] std::vector<std::string> fetchRequestedVariablesForIndexing() const;

  // deprecated
  [[deprecated("use getSampleList()")]] std::vector<Sample> &getFitSampleList(){ return getSampleList(); }
  [[deprecated("use getSampleList()")]] [[nodiscard]] const std::vector<Sample> &getFitSampleList() const { return getSampleList(); }

private:
  // config
  bool _showTimeStats_{false};
  std::vector<Sample> _sampleList_;

  // internals
  std::vector<std::string> _eventByEventDialLeafList_;

};


#endif //GUNDAM_SAMPLE_SET_H
