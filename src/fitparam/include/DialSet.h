//
// Created by Adrien BLANCHET on 21/05/2021.
//

#ifndef XSLLHFITTER_DIALSET_H
#define XSLLHFITTER_DIALSET_H

#include "string"
#include "vector"
#include "json.hpp"

#include "GenericToolbox.h"

#include "Dial.h"


class DialSet {

public:
  DialSet();
  virtual ~DialSet();

  void reset();

  void setParameterIndex(int parameterIndex);
  void setDialSetConfig(const nlohmann::json &dialSetConfig);

  void initialize();


private:
  // Parameters
  nlohmann::json _dialSetConfig_;
  int _parameterIndex_{-1};
  std::string _parameterName_;

  // Internals
  std::string _name_;             // ie detector name (or data set)
  double _parameterNominalValue_; // parameter with which the MC has produced the data set
  std::vector<Dial*> _dialList_;
  DialType::DialType _globalDialType_;

};


#endif //XSLLHFITTER_DIALSET_H
