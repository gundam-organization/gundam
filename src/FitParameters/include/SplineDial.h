//
// Created by Nadrino on 26/05/2021.
//

#ifndef GUNDAM_SPLINEDIAL_H
#define GUNDAM_SPLINEDIAL_H

#include "memory"

#include "TSpline.h"

#include "Dial.h"

class SplineDial : public Dial {

public:

  SplineDial();

  void reset() override;

  void copySpline(const TSpline3* splinePtr_);
  void createSpline(TGraph* grPtr_);
  void setMinimumSplineResponse(double minimumSplineResponse);

  void initialize() override;

  std::string getSummary() override;

protected:
  void fillResponseCache() override;


private:
  bool _throwIfResponseIsNegative_{true};
  bool _fastEval_{false};
  double _minimumSplineResponse_{std::nan("unset")};

//  TSpline3* _splinePtr_{nullptr};
  std::shared_ptr<TSpline3> _splinePtr_{nullptr};

};


#endif //GUNDAM_SPLINEDIAL_H
