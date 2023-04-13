//
// Created by Adrien Blanchet on 24/01/2023.
//

#ifndef GUNDAM_MISC_H
#define GUNDAM_MISC_H

#include "TGraph.h"
#include "TSpline.h"

namespace Misc{

  bool isGraphValid(TGraph* gr);
  bool isSplineValid(TSpline3* sp);

}


#endif //GUNDAM_MISC_H
