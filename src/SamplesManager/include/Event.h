//
// Created by Nadrino on 22/07/2021.
//

#ifndef GUNDAM_EVENT_H
#define GUNDAM_EVENT_H

#include "VariableCollection.h"
#include "EventUtils.h"

#include "GenericToolbox.Utils.h"

#include <map>
#include <mutex>
#include <vector>
#include <string>
#include <sstream>


namespace Cache { class Manager; }

class Event{

public:
  Event() = default;

  // const getters
  [[nodiscard]] const EventUtils::Indices& getIndices() const{ return _indices_; }
  [[nodiscard]] const EventUtils::Weights& getWeights() const{ return _weights_; }
  [[nodiscard]] const VariableCollection& getVariables() const{ return _variables_; }

  // mutable getters
  EventUtils::Indices& getIndices(){ return _indices_; }
  EventUtils::Weights& getWeights(){ return _weights_; }
  VariableCollection& getVariables(){ return _variables_; }

  // const core
  [[nodiscard]] double getEventWeight() const;
  [[nodiscard]] std::string getSummary() const;
  friend std::ostream& operator <<( std::ostream& o, const Event& this_ ){ o << this_.getSummary(); return o; }

private:
  // internals
  EventUtils::Indices _indices_{};
  EventUtils::Weights _weights_{};
  VariableCollection _variables_{};

#ifdef GUNDAM_USING_CACHE_MANAGER
  friend class Cache::Manager;
  [[nodiscard]] const EventUtils::Cache& getCache() const{ return _cache_; }
  EventUtils::Cache& getCache(){ return _cache_; }
  EventUtils::Cache _cache_{};
#endif

};


#endif //GUNDAM_EVENT_H
