//
// Created by Nadrino on 23/02/2025.
//

#ifndef GUNDAM_SPLINEUTILS_H
#define GUNDAM_SPLINEUTILS_H

#include "TSpline.h"
#include "TGraph.h"
#include "TObject.h"

#include <vector>
#include <string>
#include <limits>
#include <cmath>


namespace SplineUtils{

  struct SplinePoint{
    double x{std::nan("unset")};
    double y{std::nan("unset")};
    double slope{std::nan("unset")};
  };

  std::vector<SplinePoint> getSplinePointList(const TObject* src_);
  std::vector<SplinePoint> getSplinePointList(const TGraph* graph_);
  std::vector<SplinePoint> getSplinePointList(const TSpline3* spline_);

  void fillCatmullRomSlopes(std::vector<SplinePoint>& splinePointList_);
  void fillAkimaSlopes(std::vector<SplinePoint>& splinePointList_);
  void applyMonotonicCondition(std::vector<SplinePoint>& splinePointList_);

  bool isFlat(const std::vector<SplinePoint>& splinePointList_, double tolerance_ = 2*std::numeric_limits<float>::epsilon());
  bool isUniform(const std::vector<SplinePoint>& splinePointList_, double tolerance_ = 16*std::numeric_limits<float>::epsilon());
  double getSlope(const SplinePoint& start_, const SplinePoint& end_);

}


#endif // GUNDAM_SPLINEUTILS_H
