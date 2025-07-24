//
// Created by Nadrino on 25/09/2024.
//

#ifndef GUNDAM_DIALUTILS_H
#define GUNDAM_DIALUTILS_H

#include "TSpline.h"
#include "TGraph.h"
#include "TObject.h"

#include <vector>
#include <string>
#include <limits>
#include <cmath>


namespace DialUtils{

  struct DialPoint{
    double x{std::nan("unset")};
    double y{std::nan("unset")};
    double slope{std::nan("unset")};
  };

  std::vector<DialPoint> getPointList(const TObject* src_);
  std::vector<DialPoint> getPointList(const TGraph* src_);
  std::vector<DialPoint> getPointList(const TSpline3* src_);
  std::vector<DialPoint> getPointListNoSlope(const TGraph* src_);

  void fillCatmullRomSlopes(std::vector<DialPoint>& splinePointList_);
  void fillAkimaSlopes(std::vector<DialPoint>& splinePointList_);
  void applyMonotonicCondition(std::vector<DialPoint>& splinePointList_);

  bool isFlat(const std::vector<DialPoint>& splinePointList_, double tolerance_ = 2*std::numeric_limits<float>::epsilon());
  bool isUniform(const std::vector<DialPoint>& splinePointList_, double tolerance_ = 16*std::numeric_limits<float>::epsilon());
  double getSlope(const DialPoint& start_, const DialPoint& end_);

  TSpline3 buildTSpline3(const std::vector<DialPoint>& splinePointList_);

}

#endif //GUNDAM_DIALUTILS_H
