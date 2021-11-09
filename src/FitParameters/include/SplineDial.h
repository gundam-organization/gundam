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
  void setMinimumSplineResponse(double minimumSplineResponse);

  void initialize() override;

  std::string getSummary() override;

protected:
  void fillResponseCache() override;


private:
  bool _throwIfResponseIsNegative_{true};
  bool _fastEval_{false};
  double _minimumSplineResponse_{std::nan("unset")};

  TSpline3* _splinePtr_{nullptr};

};


#endif //XSLLHFITTER_SPLINEDIAL_H
