//
// Created by Nadrino on 22/07/2021.
//

#ifndef GUNDAM_EVENT_H
#define GUNDAM_EVENT_H

#include "VariableCollection.h"
#include "EventUtils.h"

#include "GenericToolbox.Utils.h"

#include <string>
#include <ostream>


namespace Cache { class Manager; }

class Event{

public:
  Event() = default;

  // const getters
  [[nodiscard]] auto& getIndices() const{ return _indices_; }
  [[nodiscard]] auto& getWeights() const{ return _weights_; }
  [[nodiscard]] auto& getVariables() const{ return _variables_; }

  // mutable getters
  auto& getIndices(){ return _indices_; }
  auto& getWeights(){ return _weights_; }
  auto& getVariables(){ return _variables_; }

  // const core
  [[nodiscard]] size_t getSize() const;
  [[nodiscard]] double getEventWeight() const { return _weights_.current; }
  [[nodiscard]] std::string getSummary() const;
  friend std::ostream& operator <<( std::ostream& o, const Event& this_ ){ o << this_.getSummary(); return o; }

private:
  // internals
  EventUtils::Indices _indices_{};
  EventUtils::Weights _weights_{};
  VariableCollection _variables_{};

};


#endif //GUNDAM_EVENT_H
