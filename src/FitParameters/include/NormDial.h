//
// Created by Nadrino on 26/05/2021.
//

#ifndef GUNDAM_NORMDIAL_H
#define GUNDAM_NORMDIAL_H

#include "Dial.h"

#include "memory"


class NormDial : public Dial {

public:
  NormDial();
  std::unique_ptr<Dial> clone() const override { return std::make_unique<NormDial>(*this); }

  void reset() override;
  void initialize() override;

  double evalResponse(double parameterValue_) override;
  double calcDial(double parameterValue_) override;

};


#endif //GUNDAM_NORMDIAL_H
