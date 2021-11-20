//
// Created by Adrien BLANCHET on 19/11/2021.
//

#ifndef GUNDAM_EVENTTREEWRITER_H
#define GUNDAM_EVENTTREEWRITER_H

#include <TDirectory.h>

#include "FitSampleSet.h"
#include "FitParameterSet.h"
#include "PhysicsEvent.h"

class EventTreeWriter {

public:
  EventTreeWriter();
  virtual ~EventTreeWriter();

  void setFitSampleSetPtr(const FitSampleSet *fitSampleSetPtr);
  void setParSetListPtr(const std::vector<FitParameterSet> *parSetListPtr);

  void writeSamples(TDirectory* saveDir_) const;
  void writeEvents(TDirectory* saveDir_, const std::string& treeName_, const std::vector<PhysicsEvent> & eventList_) const;

private:
  bool _writeDials_{false};
  const FitSampleSet* _fitSampleSetPtr_{nullptr};
  const std::vector<FitParameterSet>* _parSetListPtr_{nullptr};


};


#endif //GUNDAM_EVENTTREEWRITER_H
