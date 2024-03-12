//
// Created by Adrien BLANCHET on 19/11/2021.
//

#ifndef GUNDAM_EVENT_TREE_WRITER_H
#define GUNDAM_EVENT_TREE_WRITER_H

#include "Propagator.h"

#include "GenericToolbox.Utils.h"

#include <TDirectory.h>

#include <vector>
#include <string>

class EventTreeWriter : public GenericToolbox::ConfigBaseClass<JsonType> {

public:
  EventTreeWriter() = default;

  void writeSamples(TDirectory* saveDir_, const Propagator& propagator_) const;

  void writeEvents(TDirectory* saveDir_, const std::string& treeName_, const std::vector<PhysicsEvent> & eventList_) const;
  void writeEvents(TDirectory* saveDir_, const std::string& treeName_, const std::vector<const EventDialCache::CacheEntry*>& cacheSampleList_) const;

protected:
  void readConfigImpl() override;

  // templates related -> ensure the exact same code is used to write standard vars
  template<typename T> void writeEventsTemplate(TDirectory* saveDir_, const std::string& treeName_, const T& eventList_) const;

  static const PhysicsEvent* getEventPtr(const PhysicsEvent& ev_){ return &ev_; }
  static const PhysicsEvent* getEventPtr(const EventDialCache::CacheEntry* ev_){ return ev_->event; }

  static const std::vector<EventDialCache::DialResponseCache>* getDialElementsPtr( const PhysicsEvent& ev_){ return nullptr; }
  static const std::vector<EventDialCache::DialResponseCache>* getDialElementsPtr( const EventDialCache::CacheEntry* ev_){ return &ev_->dialResponseCacheList; }

private:
  // config
  bool _writeDials_{false};
  int _nPointsPerDial_{3};

  // cache
  mutable const Propagator* propagatorPtr{nullptr};

};


#endif //GUNDAM_EVENT_TREE_WRITER_H
