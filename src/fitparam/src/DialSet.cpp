//
// Created by Adrien BLANCHET on 21/05/2021.
//

#include "DialSet.h"

#include "Logger.h"

DialSet::DialSet() {
  Logger::setUserHeaderStr("[DialSet]");
  this->reset();
}
DialSet::~DialSet() {
  this->reset();
}

void DialSet::reset() {
  _name_ = "";
  _dialList_.clear();
}
