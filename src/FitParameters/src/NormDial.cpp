//
// Created by Nadrino on 26/05/2021.
//

#include "NormDial.h"
#include "FitParameter.h"

#include "Logger.h"

#include "sstream"


LoggerInit([]{
  Logger::setUserHeaderStr("[NormalizationDial]");
});

NormDial::NormDial() : Dial(DialType::Norm) {
  this->NormDial::reset();
}

void NormDial::reset() { Dial::reset(); }
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

