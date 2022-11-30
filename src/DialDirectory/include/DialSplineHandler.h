//
// Created by Adrien Blanchet on 30/11/2022.
//

#ifndef GUNDAM_DIALSPLINEHANDLER_H
#define GUNDAM_DIALSPLINEHANDLER_H

#include "TSpline.h"

class DialSplineHandler {

public:
  DialSplineHandler() = default;
  virtual ~DialSplineHandler() = default;

  void setSpline(const TSpline3 &spline);
  [[nodiscard]] const TSpline3 &getSpline() const;

  void copySpline(const TSpline3* splinePtr_);
  void createSpline(TGraph* grPtr_);

protected:
  TSpline3 _spline_{};

};


#endif //GUNDAM_DIALSPLINEHANDLER_H
