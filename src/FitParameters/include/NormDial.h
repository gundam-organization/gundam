//
// Created by Nadrino on 26/05/2021.
//

#ifndef GUNDAM_NORMDIAL_H
#define GUNDAM_NORMDIAL_H

#include "Dial.h"

#include "memory"

#ifdef USE_NEW_DIALS
#warning Not used with new dial implementation
#endif

class NormDial : public Dial {

public:
  explicit NormDial(const DialSet* owner_);
  [[nodiscard]] std::unique_ptr<Dial> clone() const override { return std::make_unique<NormDial>(*this); }

  void initialize() override;

  double evalResponse(double parameterValue_) override;
  double calcDial(double parameterValue_) override;

  std::string getSummary() override;

};


#endif //GUNDAM_NORMDIAL_H
