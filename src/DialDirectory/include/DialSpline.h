//
// Created by Adrien Blanchet on 29/11/2022.
//

#ifndef GUNDAM_DIALSPLINE_H
#define GUNDAM_DIALSPLINE_H


#include "DialBaseCache.h"

class DialSpline : public DialBaseCache {

public:
  DialSpline() = default;

  double evalResponseImpl(const DialInputBuffer& input_) override;

private:
  TSpline3 _spline_;

};


#endif //GUNDAM_DIALSPLINE_H
