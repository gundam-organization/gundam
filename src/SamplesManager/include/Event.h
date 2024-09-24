//
// Created by Nadrino on 22/07/2021.
//

#ifndef GUNDAM_EVENT_H
#define GUNDAM_EVENT_H

#include "EventUtils.h"
#include "ParameterSet.h"
#include "DataBinSet.h"
#include "DataBin.h"

#include "GenericToolbox.Root.h"
#include "GenericToolbox.Utils.h"

#include "TTree.h"
#include "TFormula.h"

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
  [[nodiscard]] const EventUtils::Variables& getVariables() const{ return _variables_; }

  // mutable getters
  EventUtils::Indices& getIndices(){ return _indices_; }
  EventUtils::Weights& getWeights(){ return _weights_; }
  EventUtils::Variables& getVariables(){ return _variables_; }

  // const core
  [[nodiscard]] double getEventWeight() const;

  // misc
  void fillBinIndex(const DataBinSet& binSet_){ _indices_.bin = _variables_.findBinIndex(binSet_); }

  [[nodiscard]] std::string getSummary() const;
  friend std::ostream& operator <<( std::ostream& o, const Event& this_ ){ o << this_.getSummary(); return o; }

private:
  // internals
  EventUtils::Indices _indices_{};
  EventUtils::Weights _weights_{};
  EventUtils::Variables _variables_{};

#ifdef GUNDAM_USING_CACHE_MANAGER
  friend class Cache::Manager;
  [[nodiscard]] const EventUtils::Cache& getCache() const{ return _cache_; }
  EventUtils::Cache& getCache(){ return _cache_; }
  EventUtils::Cache _cache_{};
#endif

};


// TEMPLATES IMPLEMENTATION
//template<typename T> auto Event::getVarValue( const std::string &leafName_, size_t arrayIndex_) const -> T {
//  return this->getVariable<T>(leafName_, arrayIndex_);
//}
//template<typename T> auto Event::getVariable( const std::string& leafName_, size_t arrayIndex_) const -> const T&{
//  return this->getVariableAsAnyType(leafName_, arrayIndex_).template getValue<T>();
//}


#endif //GUNDAM_EVENT_H
