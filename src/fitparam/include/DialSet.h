//
// Created by Adrien BLANCHET on 21/05/2021.
//

#ifndef XSLLHFITTER_DIALSET_H
#define XSLLHFITTER_DIALSET_H

#include "string"
#include "vector"
#include "json.hpp"
#include "memory"

#include "GenericToolbox.h"

#include "Dial.h"
#include "AnaEvent.hh"


class DialSet {

public:
  DialSet();
  virtual ~DialSet();

  void reset();

  void setParameterIndex(int parameterIndex);
  void setDialSetConfig(const nlohmann::json &dialSetConfig);

  void initialize();

  // Getters
  std::vector<std::shared_ptr<Dial>> &getDialList();
  const std::vector<std::string> &getDataSetNameList() const;

  // Core
  int getDialIndex(AnaEvent* eventPtr_);
  std::string getSummary() const;

private:
  // Parameters
  nlohmann::json _dialSetConfig_;
  int _parameterIndex_{-1};
  std::string _parameterName_;

  // Internals
  std::vector<std::string> _dataSetNameList_;
  double _parameterNominalValue_; // parameter with which the MC has produced the data set
  std::vector<std::shared_ptr<Dial>> _dialList_;
  DialType::DialType _globalDialType_;

  // Caches
  std::vector<AnaEvent*> _cachedEventPtrList_;
  std::vector<Dial*> _cachedDialPtrList_;

};


#endif //XSLLHFITTER_DIALSET_H
