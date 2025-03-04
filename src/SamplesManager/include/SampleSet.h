//
// Created by Nadrino on 22/07/2021.
//

#ifndef GUNDAM_SAMPLE_SET_H
#define GUNDAM_SAMPLE_SET_H

#include "Sample.h"

#include <string>
#include <vector>


/// Hold a description of all the event samples
/// that are going to be managed by the Propagator.  The
/// samples in the set can be referred to by their sample set index.
class SampleSet : public JsonBaseClass {

protected:
  // called through JsonBaseClass::configure() and JsonBaseClass::initialize()
  void configureImpl() override;
  void initializeImpl() override;

public:
  // Post init
  void clearEventLists();

  // const getters
  [[nodiscard]] const std::vector<Sample> &getSampleList() const{ return _sampleList_; }
  [[nodiscard]] const std::vector<std::string>& getEventVariableNameList() const{ return _eventVariableNameList_; }

  // mutable getters
  std::vector<Sample> &getSampleList(){ return _sampleList_; }
  std::vector<std::string>& getEventVariableNameList(){ return _eventVariableNameList_; }

  // core
  [[nodiscard]] bool empty() const{ return _sampleList_.empty(); }
  [[nodiscard]] std::vector<std::string> fetchRequestedVariablesForIndexing() const;

  void copyEventsFrom(const SampleSet& src_);
  [[nodiscard]] size_t getNbOfEvents() const;

  // misc
  void printConfiguration() const;
  [[nodiscard]] std::string getSampleBreakdown() const;

private:
  // config
  std::vector<Sample> _sampleList_;

  // cache -- shared
  std::vector<std::string> _eventVariableNameList_;

};


#endif //GUNDAM_SAMPLE_SET_H
