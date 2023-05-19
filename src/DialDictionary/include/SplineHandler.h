//
// Created by Adrien Blanchet on 30/11/2022.
//

#ifndef GUNDAM_SPLINEHANDLER_H
#define GUNDAM_SPLINEHANDLER_H

#include "DialInputBuffer.h"

#include "TSpline.h"


/* \brief SplineHandler handles every property of a spline from creation to eval.
 * It is meant to be used with DialBase derived classes.
 */

class SplineHandler {

public:
  SplineHandler() = default;
  virtual ~SplineHandler() = default;

  void setAllowExtrapolation(bool allowExtrapolation);
  void setSpline(const TSpline3 &spline);
  [[nodiscard]] const TSpline3 &getSpline() const;

  void copySpline(const TSpline3* splinePtr_);
  void createSpline(TGraph* grPtr_);

  [[nodiscard]] double evaluateSpline(const DialInputBuffer& input_) const;

protected:
  bool _allowExtrapolation_{false};
  TSpline3 _spline_{};

};


#endif //GUNDAM_SPLINEHANDLER_H
