//
// Created by Nadrino on 26/05/2021.
//

#ifndef GUNDAM_NORMALIZATIONDIAL_H
#define GUNDAM_NORMALIZATIONDIAL_H

#include "Dial.h"

class NormalizationDial : public Dial {

public:
  NormalizationDial();

  void reset() override;
  void initialize() override;

  double evalResponse(double parameterValue_) override;

protected:
  void fillResponseCache() override;

private:
//  double _priorValue_{std::nan("unset")};

};


#endif //GUNDAM_NORMALIZATIONDIAL_H
