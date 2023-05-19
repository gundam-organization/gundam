//
// Created by Nadrino on 26/05/2021.
//

#include "FitParameter.h"

// Unset for this file since the entire file is deprecated.
#ifdef USE_NEW_DIALS
#undef USE_NEW_DIALS
#endif

#include "NormDial.h"
#include "DialSet.h"

#include "Logger.h"

#include "sstream"


LoggerInit([]{
  Logger::setUserHeaderStr("[NormalizationDial]");
});

NormDial::NormDial(const DialSet* owner_) : Dial(DialType::Norm, owner_) {}

void NormDial::initialize() { Dial::initialize(); }

std::string NormDial::getSummary() {
  _dialParameterCache_ = _owner_->getOwner()->getParameterValue();
  _dialResponseCache_ = this->evalResponse(_dialParameterCache_);
  std::stringstream ss;
  ss << Dial::getSummary();
  return ss.str();
}

double NormDial::evalResponse(double parameterValue_){ return this->capDialResponse(this->calcDial(parameterValue_)); } // no cache
double NormDial::calcDial(double parameterValue_){ return parameterValue_; }
