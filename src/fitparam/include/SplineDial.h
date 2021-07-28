//
// Created by Nadrino on 26/05/2021.
//

#ifndef XSLLHFITTER_SPLINEDIAL_H
#define XSLLHFITTER_SPLINEDIAL_H

#include "TSpline.h"

#include "Dial.h"

class SplineDial : public Dial {

public:

  SplineDial();

  void reset() override;

  void setSplinePtr(TSpline3 *splinePtr);

  void initialize() override;

  std::string getSummary() override;
  void updateResponseCache(const double &parameterValue_) override;

private:
  TSpline3* _splinePtr_{nullptr};

};


#endif //XSLLHFITTER_SPLINEDIAL_H
