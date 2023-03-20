//
// Created by Adrien Blanchet on 24/01/2023.
//

#ifndef GUNDAM_SIMPLESPLINEHANDLER_H
#define GUNDAM_SIMPLESPLINEHANDLER_H


#include "DialInputBuffer.h"

#include "TGraph.h"
#include "TSpline.h"

#include "vector"
#include "utility"


class SimpleSplineHandler {

public:
  SimpleSplineHandler() = default;
  virtual ~SimpleSplineHandler() = default;

  void setAllowExtrapolation(bool allowExtrapolation);

  void buildSplineData(TGraph& graph_);
  void buildSplineData(const TSpline3& sp_);
  [[nodiscard]] double evaluateSpline(const DialInputBuffer& input_) const;

  bool getIsUniform() const {return _isUniform_;}
  const std::vector<double>& getSplineData() const {return _splineData_;}

protected:
  bool _allowExtrapolation_{false};
  bool _isUniform_{false};

  // A block of data to calculate the spline values.  This must be filled for
  // the Cache::Manager to work, and provides the input for spline calculation
  // functions that can be shared between the CPU and the GPU.
  std::vector<double> _splineData_{};
  std::pair<double, double> _splineBounds_{std::nan("unset"), std::nan("unset")};


};


#endif //GUNDAM_SIMPLESPLINEHANDLER_H
